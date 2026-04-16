"""
utils/experiment_tracker.py
============================
Lightweight, self-contained experiment tracking.

Assigns each run a UUID, saves config snapshots, logs per-round metrics to
JSON-lines files, and manages the gradient store (refinement #1).

Gradient storage policy (refinement #1)
----------------------------------------
* Gradients are stored ONLY for rounds 1 … K  (``attack.collect_rounds``).
* Two storage formats:
    - ``'raw'``  → full flat 1D tensor saved as .pt file
    - ``'topk'`` → (values, indices) sparse pair saved as .pt file
* All gradient artefacts go under ``results/logs/<run_id>/artifacts/gradients/``

Attack training protocol (refinements #2 & #3)
-----------------------------------------------
* Collect phase  : rounds 1 … N  → gradient train set
* Train phase    : after round N → attack model trained ONCE
* Eval phase     : rounds N+1 … T → attack model evaluated each round
  (per-round attack accuracy written to ``attack_metrics.jsonl``)

Run directory layout
---------------------
results/logs/<run_id>/
    config.json
    metrics.jsonl
    attack_metrics.jsonl
    artifacts/
        gradients/
            round_001_client_0.pt
            ...
    summary.json
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import hashlib
from utils.config import config_to_dict
from utils.gradient_processing import compress_gradient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# GradientStore  (handles policy: when/how to store)
# ------------------------------------------------------------------

class GradientStore:
    """
    Manages on-disk storage of client gradient uploads.

    Parameters
    ----------
    artifact_dir     : base artifacts directory for this run
    collect_rounds   : store gradients for rounds 1 … collect_rounds only
    storage_type     : ``'raw'`` (full tensor) or ``'topk'`` (sparse)
    topk_ratio       : fraction of elements to keep when storage_type='topk'
    """

    def __init__(
        self,
        artifact_dir: str,
        collect_rounds: int,
        storage_type: str = "raw",
        topk_ratio: float = 0.1,
    ) -> None:
        self.grad_dir = os.path.join(artifact_dir, "gradients")
        os.makedirs(self.grad_dir, exist_ok=True)
        self.collect_rounds = collect_rounds
        self.storage_type = storage_type
        self.topk_ratio = topk_ratio

        # In-memory index: list of (round, client_id, path) for train set
        self._train_index: List[Dict] = []
        self._train_hashes: set = set()
        self._eval_hashes: set = set()

    def should_store(self, round_idx: int) -> bool:
        """True iff this round falls within the collection window."""
        return 1 <= round_idx <= self.collect_rounds

    def store(
        self,
        round_idx: int,
        client_id: int,
        flat_grad: torch.Tensor,
    ) -> Optional[str]:
        """
        Persist one client's flat gradient for a given round.

        Returns the file path if stored, else None.
        """
        assert round_idx <= self.collect_rounds, "Attempted to store eval-phase data in training store"
        
        grad_hash = hashlib.sha256(flat_grad.cpu().numpy().tobytes()).hexdigest()
        assert grad_hash not in self._eval_hashes, "Data leakage: Gradient already seen in evaluation phase!"

        fname = f"round_{round_idx:04d}_client_{client_id:03d}.pt"
        fpath = os.path.join(self.grad_dir, fname)

        if self.storage_type == "topk":
            values, indices = compress_gradient(flat_grad, ratio=self.topk_ratio)
            payload = {
                "values": values.cpu(),
                "indices": indices.cpu(),
                "total_dim": flat_grad.numel(),
                "round": round_idx,
                "client_id": client_id,
            }
        else:  # raw
            payload = {
                "gradient": flat_grad.cpu(),
                "round": round_idx,
                "client_id": client_id,
            }

        torch.save(payload, fpath)
        self._train_index.append(
            {"round": round_idx, "client_id": client_id, "path": fpath}
        )
        self.register_train_hash(flat_grad)
        
        return fpath

    def register_train_hash(self, flat_grad: torch.Tensor) -> None:
        """Registers a training gradient to prevent leakage."""
        grad_hash = hashlib.sha256(flat_grad.cpu().numpy().tobytes()).hexdigest()
        assert grad_hash not in self._eval_hashes, "Data leakage: Train gradient already seen in evaluation phase!"
        self._train_hashes.add(grad_hash)
        assert self._train_hashes.isdisjoint(self._eval_hashes), "CRITICAL: Overlap between stored train and eval IDs"

    def register_eval_hash(self, flat_grad: torch.Tensor) -> None:
        """Registers an evaluation gradient to ensure it never leaks into training data."""
        grad_hash = hashlib.sha256(flat_grad.cpu().numpy().tobytes()).hexdigest()
        assert grad_hash not in self._train_hashes, "Data leakage: Eval gradient already present in training store!"
        self._eval_hashes.add(grad_hash)
        assert self._train_hashes.isdisjoint(self._eval_hashes), "CRITICAL: Overlap between stored train and eval IDs"

    def load(self, path: str) -> torch.Tensor:
        """
        Load a stored gradient back to a flat dense 1D tensor.
        Works for both ``'raw'`` and ``'topk'`` formats.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "gradient" in payload:
            return payload["gradient"]
        # topk
        from utils.gradient_processing import decompress_gradient
        return decompress_gradient(
            payload["values"],
            payload["indices"],
            payload["total_dim"],
        )

    def get_train_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load all stored gradients and return (X, y) tensors for attack training.

        Returns
        -------
        X : (N, D) float tensor — one gradient vector per row
        y : (N,)  long tensor  — client_id labels
        """
        if not self._train_index:
            raise RuntimeError(
                "No gradients have been stored.  "
                "Ensure collect_rounds > 0 and store() was called."
            )
        rows, labels = [], []
        for entry in self._train_index:
            grad = self.load(entry["path"])
            rows.append(grad)
            labels.append(entry["client_id"])
        X = torch.stack(rows, dim=0).float()
        y = torch.tensor(labels, dtype=torch.long)
        return X, y

    @property
    def train_index(self) -> List[Dict]:
        return list(self._train_index)


# ------------------------------------------------------------------
# ExperimentTracker
# ------------------------------------------------------------------

class ExperimentTracker:
    """
    One instance per experiment run.

    Usage
    -----
        tracker = ExperimentTracker(base_log_dir="results/logs/")
        tracker.start(config)
        tracker.log_round(1, {"fl_loss": 0.5, "fl_accuracy": 0.71})
        tracker.log_attack(6, {"attack_accuracy": 0.34, "privacy_score": 0.66})
        tracker.finish(summary_dict)
    """

    def __init__(self, base_log_dir: str = "results/logs/") -> None:
        self._run_id: str = str(uuid.uuid4())[:8]
        self._base_log_dir = base_log_dir
        self._run_dir: Optional[str] = None
        self._artifact_dir: Optional[str] = None
        self._gradient_store: Optional[GradientStore] = None
        self._start_time: Optional[float] = None
        self._config_dict: Optional[Dict] = None

    # ---- lifecycle -------------------------------------------------------

    def start(self, config: Any) -> str:
        """
        Initialise the run directory and persist the config snapshot.

        Parameters
        ----------
        config : SimpleNamespace or dict

        Returns
        -------
        run_id : str
        """
        self._run_dir = os.path.join(self._base_log_dir, self._run_id)
        self._artifact_dir = os.path.join(self._run_dir, "artifacts")
        os.makedirs(self._artifact_dir, exist_ok=True)

        self._config_dict = config_to_dict(config)
        self._start_time = time.time()

        # Persist config snapshot
        with open(os.path.join(self._run_dir, "config.json"), "w") as f:
            json.dump(self._config_dict, f, indent=2, default=str)

        # Initialise gradient store using attack config
        atk = self._config_dict.get("attack", {})
        gs_cfg = atk.get("gradient_store", {})
        self._gradient_store = GradientStore(
            artifact_dir=self._artifact_dir,
            collect_rounds=atk.get("collect_rounds", 5),
            storage_type=gs_cfg.get("storage_type", "raw"),
            topk_ratio=gs_cfg.get("topk_ratio", 0.1),
        )

        logger.info("Experiment started.  run_id=%s  dir=%s", self._run_id, self._run_dir)
        return self._run_id

    def finish(self, summary: Dict) -> None:
        """Write summary.json and log final stats."""
        summary["run_id"] = self._run_id
        summary["wall_time_s"] = round(time.time() - self._start_time, 2)
        summary_path = os.path.join(self._run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(
            "Run %s complete.  wall_time=%.1fs  summary=%s",
            self._run_id,
            summary["wall_time_s"],
            summary_path,
        )

    # ---- per-round logging -----------------------------------------------

    def log_round(self, round_idx: int, metrics: Dict) -> None:
        """Append one JSON line to ``metrics.jsonl``."""
        record = {"round": round_idx, **metrics}
        self._append_jsonl("metrics.jsonl", record)

    def log_attack(self, round_idx: int, attack_metrics: Dict) -> None:
        """Append one JSON line to ``attack_metrics.jsonl``."""
        record = {"round": round_idx, **attack_metrics}
        self._append_jsonl("attack_metrics.jsonl", record)

    def log_artifact(self, name: str, data: Any) -> None:
        """Save an arbitrary JSON-serialisable object to artifacts/."""
        path = os.path.join(self._artifact_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ---- gradient store proxy -------------------------------------------

    @property
    def gradient_store(self) -> GradientStore:
        if self._gradient_store is None:
            raise RuntimeError("Call tracker.start(config) before accessing gradient_store.")
        return self._gradient_store

    # ---- accessors -------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> str:
        return self._run_dir

    @property
    def artifact_dir(self) -> str:
        return self._artifact_dir

    # ---- internal -------------------------------------------------------

    def _append_jsonl(self, filename: str, record: Dict) -> None:
        path = os.path.join(self._run_dir, filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ------------------------------------------------------------------
# ExperimentRegistry — load & compare past runs
# ------------------------------------------------------------------

class ExperimentRegistry:
    """
    Scan ``results/logs/`` and build a comparison DataFrame of all runs.

    Usage
    -----
        from utils.experiment_tracker import ExperimentRegistry
        registry = ExperimentRegistry("results/logs/")
        df = registry.to_dataframe()
        df[["run_id", "final_fl_accuracy", "mean_attack_accuracy", "final_privacy_score"]]
    """

    def __init__(self, base_log_dir: str = "results/logs/") -> None:
        self.base_log_dir = base_log_dir

    def _load_summary(self, run_dir: str) -> Optional[Dict]:
        path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _load_config(self, run_dir: str) -> Optional[Dict]:
        path = os.path.join(run_dir, "config.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def list_runs(self) -> List[str]:
        """Return list of run_ids (directory names) in base_log_dir."""
        if not os.path.exists(self.base_log_dir):
            return []
        return [
            d for d in os.listdir(self.base_log_dir)
            if os.path.isdir(os.path.join(self.base_log_dir, d))
        ]

    def to_dataframe(self):
        """Return a pandas DataFrame with one row per completed run."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for ExperimentRegistry.to_dataframe()")

        rows = []
        for run_id in self.list_runs():
            run_dir = os.path.join(self.base_log_dir, run_id)
            summary = self._load_summary(run_dir)
            cfg = self._load_config(run_dir)
            if summary is None:
                continue
            row = {"run_id": run_id}
            row.update(summary)
            if cfg:
                row["defense_noise"] = cfg.get("defense", {}).get("noise", {}).get("enabled", False)
                row["defense_clipping"] = cfg.get("defense", {}).get("clipping", {}).get("enabled", False)
                row["n_clients"] = cfg.get("dataset", {}).get("n_clients", "?")
                row["partition"] = cfg.get("dataset", {}).get("partition_strategy", "?")
            rows.append(row)
        return pd.DataFrame(rows)
