"""
attack/attack_model.py
======================
Attack classifiers used to predict client identity from gradient vectors.

Five models ordered from weakest to strongest:
    1. RandomBaselineAttack   — uniform random (hard lower bound)
    2. MajorityBaselineAttack — always predicts majority class (soft lower bound)
    3. AttackLogisticRegression — linear classifier (separability test)
    4. AttackRandomForest       — non-linear sklearn ensemble
    5. AttackMLP                — neural network (strongest / most expressive)

Research interpretation
-----------------------
* If RandomBaseline ~= any other model → gradients carry no identity signal.
* If LogisticRegression is strong → gradients are linearly separable per client
  (very weak privacy; even linear attackers succeed).
* If only MLP/RF succeed → non-linear structure in gradient space reveals identity.
* If all models fail after defense → defense is effective.

All sklearn-based models wrap a common interface so inference_attack.py
can treat them uniformly.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  sklearn-compatible base protocol (duck-typed)
# ══════════════════════════════════════════════════════════════════
# All attack models expose:
#   .fit(X: np.ndarray, y: np.ndarray) → self
#   .predict(X: np.ndarray) → np.ndarray
#   .score(X, y) → float  (accuracy)


# ------------------------------------------------------------------
# 1. Random Baseline
# ------------------------------------------------------------------

class RandomBaselineAttack:
    """
    Predicts a uniformly random client ID regardless of input.
    Accuracy = 1 / n_clients — the theoretical lower bound.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.classes_: Optional[np.ndarray] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomBaselineAttack":
        self.classes_ = np.unique(y)
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._rng.choice(self.classes_, size=len(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def __repr__(self) -> str:
        return f"RandomBaselineAttack(n_classes={len(self.classes_) if self.classes_ is not None else '?'})"


# ------------------------------------------------------------------
# 2. Majority Baseline
# ------------------------------------------------------------------

class MajorityBaselineAttack:
    """
    Always predicts the most frequent class in the training set.
    Guards against inflated accuracy from class imbalance.
    """

    def __init__(self, **kwargs) -> None:
        self.majority_class_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaselineAttack":
        values, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(values[counts.argmax()])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(len(X), self.majority_class_, dtype=np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def __repr__(self) -> str:
        return f"MajorityBaselineAttack(majority_class={self.majority_class_})"


# ------------------------------------------------------------------
# 3. Logistic Regression
# ------------------------------------------------------------------

class AttackLogisticRegression:
    """
    L2-regularised multinomial logistic regression.

    If this achieves high accuracy the gradients are *linearly separable*
    — the weakest meaningful privacy violation.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 500, seed: int = 0) -> None:
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackLogisticRegression":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self._model.score(X, y))

    def __repr__(self) -> str:
        return f"AttackLogisticRegression(C={self._model.C})"


# ------------------------------------------------------------------
# 4. Random Forest
# ------------------------------------------------------------------

class AttackRandomForest:
    """
    Random Forest classifier on flattened gradient vectors.
    Strong non-linear baseline without the training instability of MLP.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackRandomForest":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self._model.score(X, y))

    def __repr__(self) -> str:
        return f"AttackRandomForest(n_estimators={self._model.n_estimators})"


# ------------------------------------------------------------------
# 5. Attack MLP (PyTorch)
# ------------------------------------------------------------------

class AttackMLP(nn.Module):
    """
    3-layer MLP attack model implemented in PyTorch.

    Designed for large gradient vectors where sklearn models may OOM.
    Input  : (N, D) — flattened gradient vectors
    Output : (N, n_clients) — logits over client identities

    Provides the same .fit() / .predict() / .score() sklearn-like interface.
    """

    def __init__(
        self,
        input_dim: int,
        num_clients: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._device = torch.device(device)
        self._lr = lr
        self._is_fitted = False

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_clients),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackMLP":
        """Train the MLP on (X, y) using cross-entropy loss."""
        from utils.seed import SeedContext
        with SeedContext(self.seed):
            self.to(self._device)
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
            criterion = nn.CrossEntropyLoss()

            X_t = torch.from_numpy(X).float().to(self._device)
            y_t = torch.from_numpy(y.astype(np.int64)).long().to(self._device)

            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for X_b, y_b in loader:
                    optimizer.zero_grad()
                    loss = criterion(self(X_b), y_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if epoch % 10 == 0:
                    logger.debug("AttackMLP epoch %d loss=%.4f", epoch, epoch_loss)

        self._is_fitted = True
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X_t = torch.from_numpy(X).float().to(self._device)
        logits = self(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())

    def __repr__(self) -> str:
        return (
            f"AttackMLP(input_dim={self.input_dim}, "
            f"num_clients={self.num_clients})"
        )


# ------------------------------------------------------------------
# Registry + factory
# ------------------------------------------------------------------

ATTACK_MODELS = {
    "random":    RandomBaselineAttack,
    "majority":  MajorityBaselineAttack,
    "logistic":  AttackLogisticRegression,
    "rf":        AttackRandomForest,
    "mlp":       AttackMLP,
}


def get_attack_model(name: str, **kwargs):
    """
    Instantiate an attack model by name.

    Parameters
    ----------
    name   : one of 'random', 'majority', 'logistic', 'rf', 'mlp'
    kwargs : forwarded to the model constructor

    Returns
    -------
    Attack model instance with .fit() / .predict() / .score() interface.
    """
    name = name.lower()
    if name not in ATTACK_MODELS:
        raise ValueError(
            f"Unknown attack model: '{name}'.  "
            f"Choose from: {list(ATTACK_MODELS.keys())}"
        )
    return ATTACK_MODELS[name](**kwargs)
