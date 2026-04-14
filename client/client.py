"""
client/client.py
================
Federated Learning client.

CRITICAL FIX (C1 + C2):
    Defense (clip + noise) is now applied to weight_delta — the object that
    is actually transmitted to the server and observed by the attacker.
    Previously, defense was applied to the last-batch gradient, which was
    never sent to the server (cosmetic defense, phantom attack signal).

    raw_gradients   = weight_delta BEFORE defense  (for audit/analysis)
    defended_gradients = weight_delta AFTER defense (sent to server)
    The attack trains on defended_gradients, matching the honest-but-curious
    server threat model.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset_loader import ClientDataset
from defense.clipping import clip_gradients
from defense.noise import add_gaussian_noise
from models import get_model
from models.base_model import BaseModel
from utils.device import get_device, to_device
from utils.gradient_processing import flatten_gradients
from utils.seed import worker_init_fn

logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """
    Everything a client sends to the server after one local training round.

    Attributes
    ----------
    client_id          : int
    weight_delta       : OrderedDict  — defended Δw sent to server
    n_samples          : int
    raw_gradients      : Tensor  — flat weight_delta BEFORE defense
    defended_gradients : Tensor  — flat weight_delta AFTER defense (= server input)
    local_loss         : float
    """
    client_id: int
    weight_delta: "OrderedDict[str, torch.Tensor]"
    n_samples: int
    raw_gradients: torch.Tensor        # 1-D, CPU, pre-defense weight_delta
    defended_gradients: torch.Tensor   # 1-D, CPU, post-defense weight_delta
    local_loss: float


class FLClient:
    """One federated learning participant."""

    def __init__(self, client_id: int, dataset: ClientDataset, config) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.device = get_device()
        self._log = logging.getLogger(f"client.{client_id}")

        model_cfg = config.model
        self.model: BaseModel = get_model(
            name=model_cfg.arch,
            input_dim=model_cfg.input_dim,
            num_classes=model_cfg.num_classes,
            hidden_dims=getattr(model_cfg, "hidden_dims", [256, 128]),
            dropout=getattr(model_cfg, "dropout", 0.3),
        )
        to_device(self.model, self.device)

        train_cfg = config.training
        self._loader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )
        self._criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weights(self, weights: "OrderedDict[str, torch.Tensor]") -> None:
        self.model.set_weights(weights)

    def get_weights(self) -> "OrderedDict[str, torch.Tensor]":
        return self.model.get_weights()

    # ------------------------------------------------------------------
    # Local training  (FIXED: defense applied to weight_delta)
    # ------------------------------------------------------------------

    def local_train(self) -> ClientUpdate:
        """
        Run local epochs then apply defense to the weight delta.

        The weight delta (w_after - w_before) is what the server receives
        and what the attacker observes.  Defense is therefore applied here,
        not to the last-batch gradient.
        """
        train_cfg = self.config.training
        defense_cfg = self.config.defense

        # Snapshot weights BEFORE training
        initial_weights = self.model.get_weights()
        optimizer = self._build_optimizer()

        self.model.train()
        total_loss, n_batches = 0.0, 0

        for _ in range(train_cfg.local_epochs):
            for X_batch, y_batch in self._loader:
                X_batch = to_device(X_batch, self.device)
                y_batch = to_device(y_batch, self.device)
                optimizer.zero_grad()
                loss = self._criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # --- Compute weight delta (the actual upload) -----------------
        raw_delta = self.model.get_weight_delta(initial_weights)
        raw_flat = flatten_gradients(raw_delta)

        # --- Apply defense to weight_delta (C1 fix) -------------------
        defended_delta = self._apply_defense(raw_delta, defense_cfg)
        defended_flat = flatten_gradients(defended_delta)

        self._log.debug(
            "local_train: epochs=%d loss=%.4f raw_norm=%.4f defended_norm=%.4f",
            train_cfg.local_epochs, avg_loss,
            raw_flat.norm(p=2).item(), defended_flat.norm(p=2).item(),
        )

        return ClientUpdate(
            client_id=self.client_id,
            weight_delta=defended_delta,   # defended delta sent to server
            n_samples=len(self.dataset),
            raw_gradients=raw_flat,        # pre-defense weight_delta (for audit)
            defended_gradients=defended_flat,  # post-defense weight_delta (attacker sees this)
            local_loss=avg_loss,
        )

    # ------------------------------------------------------------------
    # Defense
    # ------------------------------------------------------------------

    def _apply_defense(
        self,
        delta_dict: "OrderedDict[str, torch.Tensor]",
        defense_cfg,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Clip then noise the weight delta (standard DP-SGD order)."""
        defended = delta_dict

        clip_cfg = defense_cfg.clipping
        if getattr(clip_cfg, "enabled", False):
            defended = clip_gradients(defended, max_norm=float(clip_cfg.max_norm))

        noise_cfg = defense_cfg.noise
        if getattr(noise_cfg, "enabled", False):
            clip_norm = (
                float(clip_cfg.max_norm)
                if getattr(clip_cfg, "enabled", False) else None
            )
            defended = add_gaussian_noise(
                defended, sigma=float(noise_cfg.sigma), clip_norm=clip_norm
            )

        return defended

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(
            torch.utils.data.TensorDataset(self.dataset.X_test, self.dataset.y_test),
            batch_size=256, shuffle=False,
        )
        total_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in loader:
            X_b, y_b = to_device(X_b, self.device), to_device(y_b, self.device)
            logits = self.model(X_b)
            total_loss += self._criterion(logits, y_b).item() * len(y_b)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += len(y_b)
        return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train_cfg = self.config.training
        lr = float(train_cfg.lr)
        opt = getattr(train_cfg, "optimizer", "sgd").lower()
        if opt == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        return torch.optim.SGD(
            self.model.parameters(), lr=lr,
            momentum=getattr(train_cfg, "momentum", 0.9),
        )

    def __repr__(self) -> str:
        return f"FLClient(id={self.client_id}, n_train={len(self.dataset)})"
