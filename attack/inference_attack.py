"""
attack/inference_attack.py
==========================
Gradient-based client identity inference attack pipeline.

CRITICAL FIX (C2): Attack now trains and evaluates on weight_delta
(the defended update the server actually observes), NOT on the last-batch
gradient which the server never sees.

Attack protocol
---------------
Phase 1 — COLLECT  (rounds 1 … collect_rounds)
    Store each client's defended weight_delta via GradientStore.

Phase 2 — TRAIN  (once, after collect_rounds)
    Train all enabled attack models on collected deltas.
    Optional PCA dimensionality reduction (attack.pca_components > 0).
    Report TRAIN accuracy separately from TEST accuracy.

Phase 3 — EVALUATE  (rounds eval_start_round … T)
    Evaluate trained models on each new round's uploads.
    Log per-round test accuracy + privacy score.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from attack.attack_model import get_attack_model, ATTACK_MODELS
from client.client import ClientUpdate
from utils.experiment_tracker import ExperimentTracker
from utils.gradient_processing import normalize_gradient
from utils.metrics import compute_privacy_score, compute_normalized_attacker_advantage, random_baseline_accuracy
from utils.seed import SeedContext

logger = logging.getLogger(__name__)


class GradientInferenceAttack:
    """
    Orchestrates collect → train → evaluate.

    Parameters
    ----------
    config  : SimpleNamespace from load_config()
    tracker : ExperimentTracker
    """

    def __init__(self, config, tracker: ExperimentTracker) -> None:
        self.config = config
        self.tracker = tracker
        self._atk_cfg = config.attack

        self._collect_rounds: int = self._atk_cfg.collect_rounds
        self._eval_start: int = self._atk_cfg.eval_start_round
        self._n_clients: int = config.dataset.n_clients
        self._seed: int = getattr(config.logging, "seed", 42)

        model_names = getattr(self._atk_cfg, "model", ["mlp"])
        if isinstance(model_names, str):
            model_names = [model_names]
        self._model_names: List[str] = model_names

        self._norm_method: str = getattr(self._atk_cfg, "grad_norm", "l2")
        self._pca_components: int = getattr(self._atk_cfg, "pca_components", 0)
        self._pca = None  # fitted sklearn PCA, if used

        self._attack_models: Dict[str, object] = {}
        self._is_trained: bool = False
        self._train_accuracies: Dict[str, float] = {}

        # Accumulate eval-phase samples for evaluate_all_attack_models()
        # To limit memory growth, we store CPU tensors/numpy arrays instead of accumulating graph history
        self._eval_X: List[np.ndarray] = []
        self._eval_y: List[int] = []
        self._eval_ids: List[str] = []
        self._train_ids: List[str] = []

        logger.info(
            "GradientInferenceAttack: collect_rounds=%d, eval_start=%d, "
            "models=%s, pca_components=%d",
            self._collect_rounds, self._eval_start,
            self._model_names, self._pca_components,
        )

    # ------------------------------------------------------------------
    # Phase routing
    # ------------------------------------------------------------------

    def is_collect_phase(self, round_idx: int) -> bool:
        return 1 <= round_idx <= self._collect_rounds

    def is_train_round(self, round_idx: int) -> bool:
        return round_idx == self._collect_rounds + 1

    def is_eval_phase(self, round_idx: int) -> bool:
        return round_idx >= self._eval_start

    def should_store(self, round_idx: int) -> bool:
        return round_idx <= self._collect_rounds

    # ------------------------------------------------------------------
    # Phase 1: collect  (FIXED C2: store weight_delta not last-batch grad)
    # ------------------------------------------------------------------

    def collect(self, round_idx: int, updates: List[ClientUpdate]) -> None:
        """
        Store each client's defended weight_delta in the GradientStore.

        defended_gradients is the post-defense flattened weight_delta —
        exactly what the honest-but-curious server observes.
        """
        if not self.should_store(round_idx):
            logger.warning("Prevented storing gradients from round %d > collect_rounds", round_idx)
            return

        store = self.tracker.gradient_store
        for update in updates:
            store.register_train_hash(update.defended_gradients)
            store.store(
                round_idx=round_idx,
                client_id=update.client_id,
                flat_grad=update.defended_gradients,   # = defended weight_delta
            )

    # ------------------------------------------------------------------
    # Phase 2: train
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Train all attack models on collected weight deltas (once)."""
        if self._is_trained:
            logger.warning("Attack already trained; skipping.")
            return

        X, y = self.tracker.gradient_store.get_train_dataset()
        X_np = self._preprocess(X, fit_pca=True)
        y_np = y.numpy()

        for entry in self.tracker.gradient_store._train_index:
            self._train_ids.append(f"round_{entry['round']}_client_{entry['client_id']}")

        logger.info(
            "Attack training: %d samples, dim=%d (pca=%s), n_clients=%d",
            len(X_np), X_np.shape[1],
            f"{self._pca_components}" if self._pca_components > 0 else "off",
            self._n_clients,
        )

        with SeedContext(self._seed + 100):
            for name in self._model_names:
                model = self._build_attack_model(name, input_dim=X_np.shape[1])
                model.fit(X_np, y_np)
                train_acc = model.score(X_np, y_np)
                self._attack_models[name] = model
                self._train_accuracies[name] = train_acc
                logger.info("  %s — train_accuracy=%.4f", name, train_acc)

        self._is_trained = True

    # ------------------------------------------------------------------
    # Phase 3: evaluate (FIXED C2: eval uses update.defended_gradients)
    # ------------------------------------------------------------------

    def evaluate(
        self, round_idx: int, updates: List[ClientUpdate]
    ) -> Dict[str, float]:
        for update in updates:
            self._eval_ids.append(f"round_{round_idx}_client_{update.client_id}")

        """Evaluate on this round's defended weight deltas."""
        if not self._is_trained:
            return {}

        X_eval = torch.stack(
            [u.defended_gradients for u in updates], dim=0
        ).float()
        
        for u in updates:
            self.tracker.gradient_store.register_eval_hash(u.defended_gradients)
            
        assert len(self.tracker.gradient_store._train_hashes) > 0, "Train hashes missing"
        assert len(self.tracker.gradient_store._eval_hashes) > 0, "Eval hashes missing"
            
        assert self.tracker.gradient_store._train_hashes.isdisjoint(
            self.tracker.gradient_store._eval_hashes
        ), "CRITICAL DATA LEAKAGE: Gradient hashes overlap between train and eval"

        y_eval = np.array([u.client_id for u in updates], dtype=np.int64)

        X_np = self._preprocess(X_eval, fit_pca=False)

        # Accumulate for evaluate_all_attack_models()
        self._eval_X.append(X_np)
        self._eval_y.extend(y_eval.tolist())

        results: Dict[str, float] = {}
        for name, model in self._attack_models.items():
            acc = model.score(X_np, y_eval)
            results[f"attack_acc_{name}"] = round(acc, 6)
            results[f"attack_acc_{name}_train"] = round(
                self._train_accuracies.get(name, 0.0), 6
            )

        baseline_attacks = ["random", "majority"]
        real_attacks = [n for n in self._model_names if n not in baseline_attacks]
        baseline_models = [n for n in self._model_names if n in baseline_attacks]

        accs_real = [results[f"attack_acc_{n}"] for n in real_attacks if f"attack_acc_{n}" in results]
        accs_base = [results[f"attack_acc_{n}"] for n in baseline_models if f"attack_acc_{n}" in results]

        best_acc = float(np.max(accs_real)) if accs_real else 0.0
        
        # Check for attack overfitting
        best_train_acc = float(np.max([results[f"attack_acc_{n}_train"] for n in real_attacks if f"attack_acc_{n}_train" in results])) if accs_real else 0.0
        gap = best_train_acc - best_acc
        if gap > 0.2:
            logger.warning("ATTACK OVERFITTING WARNING: Best Attack Train Acc=%.4f vs Test Acc=%.4f (Gap=%.4f > 0.2)",
                           best_train_acc, best_acc, gap)

        mean_acc = float(np.mean(accs_real)) if accs_real else 0.0
        base_acc = float(np.mean(accs_base)) if accs_base else random_baseline_accuracy(self._n_clients)

        results["best_attack_accuracy"] = round(best_acc, 6)
        results["mean_attack_accuracy"] = round(mean_acc, 6)
        results["baseline_accuracy"] = round(base_acc, 6)
        results["privacy_score"] = round(compute_privacy_score(best_acc), 6)
        results["normalized_attacker_advantage"] = round(
            compute_normalized_attacker_advantage(best_acc, base_acc), 6
        )

        logger.info(
            "Round %d attack eval: best_acc=%.4f mean_acc=%.4f base=%.4f privacy=%.4f naa=%.4f",
            round_idx, best_acc, mean_acc, base_acc, results["privacy_score"],
            results["normalized_attacker_advantage"]
        )
        self.tracker.log_attack(round_idx, results)
        return results

    # ------------------------------------------------------------------
    # Full comparison across all 5 attack models
    # ------------------------------------------------------------------

    def evaluate_all_attack_models(self) -> Dict[str, Dict]:
        """
        Run all 5 registered attack models on the accumulated eval set.
        Called at end of experiment from run_experiment.py.

        Returns
        -------
        Dict: {model_name: {accuracy, privacy_score, normalized_attacker_advantage}}
        """
        if not self._eval_X:
            logger.warning("No eval data accumulated; skipping full comparison.")
            return {}

        X_test = np.vstack(self._eval_X)
        y_test = np.array(self._eval_y, dtype=np.int64)

        X_train, y_train = self.tracker.gradient_store.get_train_dataset()
        X_train_np = self._preprocess(X_train, fit_pca=False)

        comparison = {}
        with SeedContext(self._seed + 200):
            for name in ATTACK_MODELS:
                model = self._build_attack_model(name, input_dim=X_train_np.shape[1])
                model.fit(X_train_np, y_train.numpy())
                acc = model.score(X_test, y_test)
                comparison[name] = {
                    "accuracy": round(acc, 6),
                    "privacy_score": round(compute_privacy_score(acc), 6),
                    "normalized_attacker_advantage": round(
                        compute_normalized_attacker_advantage(acc, random_baseline_accuracy(self._n_clients)), 6
                    ),
                }
        self.tracker.log_artifact("all_attack_model_comparison", comparison)
        logger.info("All-model comparison: %s", comparison)
        return comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, X: torch.Tensor, fit_pca: bool = False) -> np.ndarray:
        """Normalize gradient vectors, then optionally apply PCA."""
        normed = torch.stack(
            [normalize_gradient(x, method=self._norm_method) for x in X], dim=0
        ).numpy()

        if self._pca_components > 0:
            if fit_pca:
                from sklearn.decomposition import PCA
                n_components = min(self._pca_components, normed.shape[0] - 1, normed.shape[1])
                self._pca = PCA(n_components=n_components, random_state=self._seed)
                normed = self._pca.fit_transform(normed)
                logger.info(
                    "PCA fitted: %d → %d dims (%.1f%% variance explained)",
                    X.shape[1], n_components,
                    100 * self._pca.explained_variance_ratio_.sum(),
                )
            elif self._pca is not None:
                normed = self._pca.transform(normed)

        return normed

    def _build_attack_model(self, name: str, input_dim: int):
        if name == "mlp":
            return get_attack_model(
                "mlp",
                input_dim=input_dim,
                num_clients=self._n_clients,
                hidden_dim=getattr(self._atk_cfg, "mlp_hidden_dim", 128),
                epochs=getattr(self._atk_cfg, "mlp_epochs", 30),
                seed=self._seed,
            )
        if name == "majority":
            return get_attack_model(name)
        return get_attack_model(name, seed=self._seed)
