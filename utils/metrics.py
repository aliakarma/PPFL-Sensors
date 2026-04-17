"""
utils/metrics.py
================
Compute and accumulate metrics for FL task and inference attack.

Key additions
-------------
compute_privacy_score(attack_acc)            → 1 − attack_acc
compute_normalized_attacker_advantage(...)   → (attack_acc − random) / (1 − random)
    Comparable across experiments with different n_clients.
    0 = no better than random; 1 = perfect attack.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Stateless helpers
# ------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_privacy_score(best_attack_accuracy: float) -> float:
    """privacy_score = 1 − best_attack_accuracy.  Higher = more private."""
    return max(0.0, min(1.0, 1.0 - float(best_attack_accuracy)))


def compute_normalized_attacker_advantage(
    attack_accuracy: float, baseline_accuracy: float
) -> float:
    """
    Normalized Attacker Advantage (NAA).

    NAA = (attack_acc − random_baseline) / (1 − random_baseline)
    """
    assert 0.0 <= baseline_accuracy <= 1.0, f"Invalid baseline accuracy: {baseline_accuracy}"
    denom = 1.0 - float(baseline_accuracy)
    if denom < 1e-10:
        return 0.0
    naa = (float(attack_accuracy) - float(baseline_accuracy)) / denom
    return max(0.0, min(1.0, naa))


def random_baseline_accuracy(n_clients: int) -> float:
    return 1.0 / max(1, n_clients)


# ------------------------------------------------------------------
# Accumulator
# ------------------------------------------------------------------

@dataclass
class _RoundRecord:
    round_idx: int
    fl_loss: Optional[float] = None
    fl_accuracy: Optional[float] = None
    best_attack_accuracy: Optional[float] = None
    mean_attack_accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    attack_accuracy_train: Optional[float] = None   # memorisation check
    privacy_score: Optional[float] = None
    normalized_attacker_advantage: Optional[float] = None
    defense_active: bool = False


@dataclass
class MetricsTracker:
    n_clients: int = 1
    records: List[_RoundRecord] = field(default_factory=list)
    _index: Dict[int, _RoundRecord] = field(default_factory=dict, repr=False)

    def _get_or_create(self, round_idx: int) -> _RoundRecord:
        if round_idx not in self._index:
            rec = _RoundRecord(round_idx=round_idx)
            self.records.append(rec)
            self._index[round_idx] = rec
        return self._index[round_idx]

    def update_fl(
        self,
        round_idx: int,
        loss: float,
        accuracy: float,
        defense_active: bool = False,
    ) -> None:
        rec = self._get_or_create(round_idx)
        rec.fl_loss = round(loss, 6)
        rec.fl_accuracy = round(accuracy, 6)
        rec.defense_active = defense_active

    def update_attack(
        self,
        round_idx: int,
        best_attack_accuracy: float,
        mean_attack_accuracy: float,
        baseline_accuracy: float,
        train_accuracy: Optional[float] = None,
    ) -> None:
        rec = self._get_or_create(round_idx)
        rec.best_attack_accuracy = round(best_attack_accuracy, 6)
        rec.mean_attack_accuracy = round(mean_attack_accuracy, 6)
        rec.baseline_accuracy = round(baseline_accuracy, 6)
        rec.privacy_score = round(compute_privacy_score(best_attack_accuracy), 6)
        rec.normalized_attacker_advantage = round(
            compute_normalized_attacker_advantage(best_attack_accuracy, baseline_accuracy), 6
        )
        if train_accuracy is not None:
            rec.attack_accuracy_train = round(train_accuracy, 6)

    def to_list(self) -> List[Dict]:
        return [
            {k: v for k, v in vars(r).items() if v is not None}
            for r in sorted(self.records, key=lambda r: r.round_idx)
        ]

    def summary(self) -> Dict:
        fl_accs = [r.fl_accuracy for r in self.records if r.fl_accuracy is not None]
        best_atk_accs = [r.best_attack_accuracy for r in self.records if r.best_attack_accuracy is not None]
        mean_atk_accs = [r.mean_attack_accuracy for r in self.records if r.mean_attack_accuracy is not None]
        naa_vals = [r.normalized_attacker_advantage for r in self.records
                    if r.normalized_attacker_advantage is not None]

        best_fl_acc = max(fl_accs) if fl_accs else None
        final_fl_acc = fl_accs[-1] if fl_accs else None
        mean_best_atk_acc = float(np.mean(best_atk_accs)) if best_atk_accs else None
        mean_atk_acc = float(np.mean(mean_atk_accs)) if mean_atk_accs else None
        mean_naa = float(np.mean(naa_vals)) if naa_vals else None
        final_privacy = compute_privacy_score(mean_best_atk_acc) if mean_best_atk_acc is not None else None
        random_bl = random_baseline_accuracy(self.n_clients)

        return {
            "best_fl_accuracy": best_fl_acc,
            "final_fl_accuracy": final_fl_acc,
            "mean_best_attack_accuracy": mean_best_atk_acc,
            "mean_attack_accuracy": mean_atk_acc,
            "mean_normalized_attacker_advantage": mean_naa,
            "final_privacy_score": final_privacy,
            "random_baseline_attack_accuracy": random_bl,
            "total_rounds": len(self.records),
        }
