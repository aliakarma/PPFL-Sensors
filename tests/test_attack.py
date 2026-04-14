"""tests/test_attack.py — unit tests for attack models and inference pipeline."""

import pytest
import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from attack.attack_model import (
    RandomBaselineAttack,
    MajorityBaselineAttack,
    AttackLogisticRegression,
    AttackRandomForest,
    AttackMLP,
    get_attack_model,
    ATTACK_MODELS,
)
from utils.seed import set_global_seed


# ── Fixtures ──────────────────────────────────────────────────────

N_CLIENTS = 5
N_SAMPLES  = 60   # 12 per client
GRAD_DIM   = 40


def _make_xy(n_clients=N_CLIENTS, n_samples=N_SAMPLES, dim=GRAD_DIM, seed=0):
    """Synthetic gradient dataset with linearly separable structure."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cid in range(n_clients):
        # Add a strong per-client mean so classifiers can succeed
        centre = rng.normal(cid * 3.0, 0.5, dim)
        Xc = rng.normal(centre, 0.3, size=(n_samples // n_clients, dim)).astype(np.float32)
        X.append(Xc)
        y.extend([cid] * (n_samples // n_clients))
    return np.vstack(X), np.array(y, dtype=np.int64)


# ── Model registry ────────────────────────────────────────────────

def test_all_model_names_in_registry():
    expected = {"random", "majority", "logistic", "rf", "mlp"}
    assert set(ATTACK_MODELS.keys()) == expected


def test_get_attack_model_unknown_raises():
    with pytest.raises(ValueError):
        get_attack_model("nonexistent_model")


# ── RandomBaselineAttack ──────────────────────────────────────────

class TestRandomBaseline:
    def test_fit_predict(self):
        X, y = _make_xy()
        model = RandomBaselineAttack(seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(np.unique(y)))

    def test_accuracy_near_random(self):
        """Over many trials, accuracy should converge to 1/n_clients."""
        X, y = _make_xy(n_clients=5, n_samples=500)
        model = RandomBaselineAttack(seed=42)
        model.fit(X, y)
        acc = model.score(X, y)
        # Allow ±15% tolerance around 0.2
        assert abs(acc - 1/N_CLIENTS) < 0.15, f"Random accuracy {acc:.3f} too far from 1/{N_CLIENTS}"

    def test_predict_before_fit_raises(self):
        model = RandomBaselineAttack()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 10), dtype=np.float32))


# ── MajorityBaselineAttack ────────────────────────────────────────

class TestMajorityBaseline:
    def test_always_predicts_majority(self):
        X, y = _make_xy()
        model = MajorityBaselineAttack()
        model.fit(X, y)
        preds = model.predict(X)
        # All predictions should be the same class
        assert len(set(preds.tolist())) == 1

    def test_predict_before_fit_raises(self):
        model = MajorityBaselineAttack()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 10), dtype=np.float32))


# ── AttackLogisticRegression ──────────────────────────────────────

class TestLogisticAttack:
    def test_fit_predict_shape(self):
        X, y = _make_xy()
        model = AttackLogisticRegression(seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_beats_random_on_separable(self):
        """LR must beat random baseline on linearly separable data."""
        X, y = _make_xy(seed=7)
        model = AttackLogisticRegression(seed=0)
        model.fit(X, y)
        acc = model.score(X, y)
        random_baseline = 1.0 / N_CLIENTS
        assert acc > random_baseline, \
            f"LR acc {acc:.3f} should beat random {random_baseline:.3f}"

    def test_score_in_valid_range(self):
        X, y = _make_xy()
        model = AttackLogisticRegression(seed=0)
        model.fit(X, y)
        acc = model.score(X, y)
        assert 0.0 <= acc <= 1.0


# ── AttackRandomForest ────────────────────────────────────────────

class TestRandomForestAttack:
    def test_fit_predict(self):
        X, y = _make_xy()
        model = AttackRandomForest(n_estimators=10, seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(preds).issubset(set(np.unique(y)))

    def test_score_bounds(self):
        X, y = _make_xy()
        model = AttackRandomForest(n_estimators=10, seed=0)
        model.fit(X, y)
        assert 0.0 <= model.score(X, y) <= 1.0


# ── AttackMLP ────────────────────────────────────────────────────

class TestAttackMLP:
    def test_fit_predict_shape(self):
        set_global_seed(0)
        X, y = _make_xy()
        model = AttackMLP(
            input_dim=GRAD_DIM, num_clients=N_CLIENTS,
            hidden_dim=32, epochs=5, seed=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_score_bounds(self):
        set_global_seed(1)
        X, y = _make_xy()
        model = AttackMLP(
            input_dim=GRAD_DIM, num_clients=N_CLIENTS,
            hidden_dim=32, epochs=5, seed=1,
        )
        model.fit(X, y)
        acc = model.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_beats_random_on_separable(self):
        """MLP must learn on clearly separable gradient data."""
        set_global_seed(2)
        X, y = _make_xy(seed=2)
        model = AttackMLP(
            input_dim=GRAD_DIM, num_clients=N_CLIENTS,
            hidden_dim=64, epochs=30, seed=2,
        )
        model.fit(X, y)
        acc = model.score(X, y)
        assert acc > 1.0 / N_CLIENTS, \
            f"MLP acc {acc:.3f} should beat random baseline"


# ── GradientInferenceAttack pipeline ─────────────────────────────

class TestInferenceAttackPipeline:
    """
    Tests the collect → train → evaluate protocol using mocked components.
    """

    def _make_config(self):
        return SimpleNamespace(
            dataset=SimpleNamespace(n_clients=N_CLIENTS),
            attack=SimpleNamespace(
                enabled=True,
                collect_rounds=3,
                eval_start_round=4,
                model=["random", "majority"],
                grad_norm="l2",
                mlp_hidden_dim=32,
                mlp_epochs=5,
                gradient_store=SimpleNamespace(
                    storage_type="raw",
                    topk_ratio=0.1,
                ),
            ),
            logging=SimpleNamespace(seed=42),
        )

    def _make_tracker(self, tmp_path):
        from utils.experiment_tracker import ExperimentTracker
        cfg = self._make_config()
        tracker = ExperimentTracker(base_log_dir=str(tmp_path))
        tracker.start(cfg)
        return tracker

    def _make_updates(self, round_idx, n_clients=N_CLIENTS, dim=GRAD_DIM):
        from client.client import ClientUpdate
        from collections import OrderedDict
        updates = []
        rng = np.random.default_rng(round_idx)
        for cid in range(n_clients):
            # Unique per-client gradient pattern
            grad = torch.from_numpy(
                rng.normal(cid * 2.0, 0.5, dim).astype(np.float32)
            )
            updates.append(ClientUpdate(
                client_id=cid,
                weight_delta=OrderedDict(),
                n_samples=20,
                raw_gradients=grad,
                defended_gradients=grad,
                local_loss=0.5,
            ))
        return updates

    def test_phase_routing(self, tmp_path):
        from attack.inference_attack import GradientInferenceAttack
        cfg = self._make_config()
        tracker = self._make_tracker(tmp_path)
        attack = GradientInferenceAttack(cfg, tracker)

        assert attack.is_collect_phase(1)
        assert attack.is_collect_phase(3)
        assert not attack.is_collect_phase(4)
        assert attack.is_train_round(4)
        assert attack.is_eval_phase(4)
        assert attack.is_eval_phase(10)

    def test_collect_stores_gradients(self, tmp_path):
        from attack.inference_attack import GradientInferenceAttack
        cfg = self._make_config()
        tracker = self._make_tracker(tmp_path)
        attack = GradientInferenceAttack(cfg, tracker)

        for r in range(1, 4):
            attack.collect(r, self._make_updates(r))

        X, y = tracker.gradient_store.get_train_dataset()
        # 3 rounds × N_CLIENTS = 15 gradient vectors
        assert X.shape[0] == 3 * N_CLIENTS
        assert y.shape[0] == 3 * N_CLIENTS

    def test_train_and_evaluate(self, tmp_path):
        from attack.inference_attack import GradientInferenceAttack
        cfg = self._make_config()
        tracker = self._make_tracker(tmp_path)
        attack = GradientInferenceAttack(cfg, tracker)

        for r in range(1, 4):
            attack.collect(r, self._make_updates(r))

        attack.train()
        assert attack._is_trained

        results = attack.evaluate(5, self._make_updates(5))
        assert "mean_attack_accuracy" in results
        assert "privacy_score" in results
        assert 0.0 <= results["mean_attack_accuracy"] <= 1.0
        assert 0.0 <= results["privacy_score"] <= 1.0
