"""attack — gradient-based client identity inference attack."""

from attack.attack_model import (
    AttackMLP,
    AttackRandomForest,
    AttackLogisticRegression,
    RandomBaselineAttack,
    MajorityBaselineAttack,
    get_attack_model,
    ATTACK_MODELS,
)
from attack.inference_attack import GradientInferenceAttack

__all__ = [
    "AttackMLP",
    "AttackRandomForest",
    "AttackLogisticRegression",
    "RandomBaselineAttack",
    "MajorityBaselineAttack",
    "get_attack_model",
    "ATTACK_MODELS",
    "GradientInferenceAttack",
]
