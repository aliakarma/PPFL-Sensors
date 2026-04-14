"""
server/ensemble.py
==================
Ensemble Federated Learning server.

Defense rationale
-----------------
Standard FedAvg merges all client updates into a single global model, making
each client's gradient directly observable by the server (honest-but-curious
attacker).  Ensemble FL instead trains K sub-models, each on a disjoint subset
of clients, and combines their predictions at inference time.

This means:
* No single model sees all clients' gradients.
* An attacker observing one sub-model can only infer a subset of client IDs.
* Privacy improves at the cost of some accuracy (evaluated experimentally).

Architecture
------------
    clients → grouped into K SubModelGroups
    each group trains its own sub-model independently
    inference: ensemble_predict() aggregates across sub-models

Usage
-----
Activated via ``training.aggregation: ensemble`` in YAML config.
``EnsembleServer`` is a drop-in replacement for ``FLServer``.
"""

import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from client.client import FLClient, ClientUpdate
from data.dataset_loader import ClientDataset
from models.base_model import BaseModel
from models import get_model
from server.aggregation import fedavg
from server.server import FLServer
from utils.device import get_device, to_device

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------

@dataclass
class SubModelGroup:
    """
    One sub-model and its assigned clients.

    Attributes
    ----------
    group_id   : int
    model      : BaseModel — the sub-model (independent weights)
    client_ids : list of assigned client IDs
    """
    group_id: int
    model: BaseModel
    client_ids: List[int] = field(default_factory=list)


# ------------------------------------------------------------------
# Group assignment strategies
# ------------------------------------------------------------------

def create_submodel_groups(
    clients: List[FLClient],
    n_groups: int,
    model_template: BaseModel,
    strategy: str = "round_robin",
    seed: int = 42,
) -> List[SubModelGroup]:
    """
    Assign clients to sub-model groups.

    Parameters
    ----------
    clients        : all FLClient instances
    n_groups       : number of sub-models to maintain
    model_template : a model instance whose architecture is copied per group
    strategy       : ``'round_robin'``, ``'random'``
    seed           : RNG seed for random strategy

    Returns
    -------
    List[SubModelGroup]
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n_clients = len(clients)

    if n_groups > n_clients:
        logger.warning(
            "n_groups (%d) > n_clients (%d); clamping to n_clients.",
            n_groups, n_clients,
        )
        n_groups = n_clients

    # Create independent sub-model copies
    groups = [
        SubModelGroup(
            group_id=gid,
            model=copy.deepcopy(model_template),
        )
        for gid in range(n_groups)
    ]

    # Assign client IDs to groups
    if strategy == "round_robin":
        for i, client in enumerate(clients):
            groups[i % n_groups].client_ids.append(client.client_id)
    elif strategy == "random":
        perm = rng.permutation(n_clients)
        splits = [arr.tolist() for arr in np.array_split(perm, n_groups)]
        for gid, idx_list in enumerate(splits):
            groups[gid].client_ids = [clients[i].client_id for i in idx_list]
    else:
        raise ValueError(f"Unknown group strategy: '{strategy}'")

    for g in groups:
        logger.info(
            "SubModelGroup %d: clients %s", g.group_id, g.client_ids
        )

    return groups


# ------------------------------------------------------------------
# Ensemble prediction
# ------------------------------------------------------------------

def ensemble_predict(
    x: torch.Tensor,
    sub_models: List[BaseModel],
    aggregation: str = "average_logits",
) -> torch.Tensor:
    """
    Combine predictions from multiple sub-models.

    Parameters
    ----------
    x            : input tensor (batch, ...)
    sub_models   : list of trained sub-models
    aggregation  : ``'average_logits'``, ``'majority_vote'``,
                   ``'max_confidence'``

    Returns
    -------
    Predicted class indices: (batch,) LongTensor
    """
    logits_list = []
    for model in sub_models:
        model.eval()
        with torch.no_grad():
            logits_list.append(model(x))  # (batch, C)

    if aggregation == "average_logits":
        avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return avg_logits.argmax(dim=1)

    if aggregation == "majority_vote":
        preds = torch.stack(
            [lg.argmax(dim=1) for lg in logits_list], dim=0
        )  # (n_models, batch)
        # Mode along model dimension
        mode_result = preds.mode(dim=0)
        return mode_result.values

    if aggregation == "max_confidence":
        # Pick the prediction from the most confident sub-model per sample
        probs_list = [torch.softmax(lg, dim=1) for lg in logits_list]
        max_probs = torch.stack(
            [p.max(dim=1).values for p in probs_list], dim=0
        )  # (n_models, batch)
        best_model_idx = max_probs.argmax(dim=0)  # (batch,)
        preds = torch.stack(
            [lg.argmax(dim=1) for lg in logits_list], dim=0
        )  # (n_models, batch)
        return preds[best_model_idx, torch.arange(x.size(0))]

    raise ValueError(f"Unknown ensemble aggregation: '{aggregation}'")


# ------------------------------------------------------------------
# EnsembleServer
# ------------------------------------------------------------------

class EnsembleServer(FLServer):
    """
    Federated server that maintains K independent sub-models.

    Inherits broadcast/collect/evaluate from FLServer.
    Overrides ``aggregate()`` and ``run_round()`` to work per sub-model group.

    Parameters
    ----------
    Same as FLServer, plus:
        config.training.n_ensemble_groups (int, default 3)
        config.training.ensemble_strategy (str, default 'round_robin')
        config.training.ensemble_predict  (str, default 'average_logits')
    """

    def __init__(
        self,
        model: BaseModel,
        clients: List[FLClient],
        config,
        test_dataset: ClientDataset,
    ) -> None:
        super().__init__(model, clients, config, test_dataset)

        n_groups = getattr(config.training, "n_ensemble_groups", 3)
        strategy = getattr(config.training, "ensemble_strategy", "round_robin")
        seed = getattr(config.logging, "seed", 42)

        self._groups = create_submodel_groups(
            clients=clients,
            n_groups=n_groups,
            model_template=model,
            strategy=strategy,
            seed=seed,
        )
        self._ensemble_predict_mode = getattr(
            config.training, "ensemble_predict", "average_logits"
        )
        # Move sub-models to device
        for g in self._groups:
            to_device(g.model, self.device)

        self._log.info(
            "EnsembleServer: %d groups, predict_mode=%s",
            n_groups, self._ensemble_predict_mode,
        )

    # ------------------------------------------------------------------
    # Per-group helpers
    # ------------------------------------------------------------------

    def _get_group_clients(self, group: SubModelGroup) -> List[FLClient]:
        """Return FLClient objects belonging to this group."""
        id_set = set(group.client_ids)
        return [c for c in self.clients if c.client_id in id_set]

    def _broadcast_to_group(self, group: SubModelGroup) -> None:
        """Send this group's sub-model weights to its clients."""
        w = group.model.get_weights()
        for client in self._get_group_clients(group):
            client.set_weights(w)

    def _collect_group_updates(
        self, group: SubModelGroup
    ) -> List[ClientUpdate]:
        """Collect updates from this group's clients only."""
        updates = []
        for client in self._get_group_clients(group):
            update = client.local_train()
            updates.append(update)
        return updates

    # ------------------------------------------------------------------
    # Overridden round logic
    # ------------------------------------------------------------------

    def run_round(
        self, round_idx: int
    ) -> Tuple[List[ClientUpdate], Dict]:
        """
        Train each sub-model group independently.

        Returns all updates (across all groups) so the GradientStore can
        still record them for attack evaluation.
        """
        self._log.info(
            "─── Ensemble Round %d / %d  (%d groups) ───",
            round_idx, self.config.training.rounds, len(self._groups),
        )

        all_updates: List[ClientUpdate] = []

        for group in self._groups:
            # Broadcast sub-model weights to group clients
            self._broadcast_to_group(group)

            # Collect updates from group clients
            group_updates = self._collect_group_updates(group)
            all_updates.extend(group_updates)

            # Aggregate within group
            delta_updates = [(u.weight_delta, u.n_samples) for u in group_updates]
            aggregated_delta = fedavg(delta_updates)

            # Apply delta to sub-model
            current_w = group.model.get_weights()
            from collections import OrderedDict
            new_w = OrderedDict()
            for key in current_w:
                new_w[key] = current_w[key].float() + aggregated_delta[key].float()
            group.model.set_weights(new_w)

        # Evaluate ensemble
        eval_metrics = self.evaluate_ensemble()
        avg_client_loss = (
            sum(u.local_loss for u in all_updates) / len(all_updates)
            if all_updates else 0.0
        )

        metrics = {
            "fl_loss": eval_metrics["loss"],
            "fl_accuracy": eval_metrics["accuracy"],
            "avg_client_loss": avg_client_loss,
        }
        self._log.info(
            "Ensemble Round %d: acc=%.4f  loss=%.4f",
            round_idx, metrics["fl_accuracy"], metrics["fl_loss"],
        )
        return all_updates, metrics

    # ------------------------------------------------------------------
    # Ensemble evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_ensemble(self) -> Dict[str, float]:
        """
        Evaluate the ensemble (all sub-models combined) on the test set.
        """
        ds = self._test_dataset
        loader = DataLoader(
            TensorDataset(ds.X_test, ds.y_test),
            batch_size=512,
            shuffle=False,
        )
        sub_models = [g.model for g in self._groups]
        for m in sub_models:
            m.eval()

        correct, total = 0, 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        for X_batch, y_batch in loader:
            X_batch = to_device(X_batch, self.device)
            y_batch = to_device(y_batch, self.device)

            preds = ensemble_predict(
                X_batch, sub_models, aggregation=self._ensemble_predict_mode
            )
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

            # Loss from average logits sub-model
            logits_list = [m(X_batch) for m in sub_models]
            avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
            loss = criterion(avg_logits, y_batch)
            total_loss += loss.item() * len(y_batch)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    # Override evaluate() to use ensemble
    def evaluate(self) -> Dict[str, float]:
        return self.evaluate_ensemble()

    @property
    def sub_models(self) -> List[BaseModel]:
        return [g.model for g in self._groups]
