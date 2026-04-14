"""
server/server.py
================
FLServer orchestrates federated learning rounds.

Responsibilities
----------------
1. Maintain the global model.
2. Broadcast global weights to all clients each round.
3. Collect ClientUpdate objects from all clients.
4. Aggregate updates into new global weights (FedAvg or FedMedian).
5. Evaluate the global model on the held-out test set.
6. Return per-round metrics and client updates (for attack pipeline).

The server is intentionally stateless w.r.t. clients — it receives updates
and returns a new model; it does NOT store raw client data.

Factory
-------
Use ``get_server(config, model, clients, test_dataset)`` to instantiate
the correct server class based on ``config.training.aggregation``.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from client.client import FLClient, ClientUpdate
from data.dataset_loader import ClientDataset
from models.base_model import BaseModel
from server.aggregation import fedavg, fedmedian
from utils.device import get_device, to_device

logger = logging.getLogger(__name__)


class FLServer:
    """
    Standard federated learning server using FedAvg or FedMedian.

    Parameters
    ----------
    model    : global BaseModel instance
    clients  : list of FLClient instances
    config   : SimpleNamespace from load_config()
    test_dataset : ClientDataset  (uses .X_test / .y_test for global eval)
    """

    def __init__(
        self,
        model: BaseModel,
        clients: List[FLClient],
        config,
        test_dataset: ClientDataset,
    ) -> None:
        self.model = model
        self.clients = clients
        self.config = config
        self.device = get_device()
        self._test_dataset = test_dataset
        self._criterion = nn.CrossEntropyLoss()
        self._log = logging.getLogger("server")

        to_device(self.model, self.device)

        # Select aggregation function
        agg_name = getattr(config.training, "aggregation", "fedavg").lower()
        if agg_name in ("fedavg", "weighted_fedavg"):
            self._aggregate_fn = fedavg
        elif agg_name == "fedmedian":
            self._aggregate_fn = fedmedian
        else:
            raise ValueError(
                f"Unknown aggregation: '{agg_name}'.  "
                "Choose: fedavg, fedmedian, ensemble"
            )
        self._log.info(
            "FLServer ready: %d clients, aggregation=%s", len(clients), agg_name
        )

    # ------------------------------------------------------------------
    # Core round logic
    # ------------------------------------------------------------------

    def broadcast_weights(self) -> "OrderedDict[str, torch.Tensor]":
        """Copy global weights to all clients. Returns the weight dict."""
        global_weights = self.model.get_weights()
        for client in self.clients:
            client.set_weights(global_weights)
        return global_weights

    def collect_updates(self) -> List[ClientUpdate]:
        """
        Trigger local training on all clients and collect their updates.

        Returns
        -------
        List of ClientUpdate objects (one per client).
        """
        updates: List[ClientUpdate] = []
        for client in self.clients:
            update = client.local_train()
            updates.append(update)
            self._log.debug(
                "Received update from client %d  (loss=%.4f, n=%d)",
                client.client_id, update.local_loss, update.n_samples,
            )
        return updates

    def aggregate(
        self, updates: List[ClientUpdate]
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Run the aggregation function on client weight deltas, then apply the
        resulting delta to the current global weights.

        Using weight deltas (not full weights) is equivalent to FedAvg when
        all clients start from the same global weights, and is easier to
        analyse for the attack.

        Returns
        -------
        New global weight dict (also applied to self.model in-place).
        """
        global_weights = self.model.get_weights()

        # Aggregate deltas
        delta_updates = [(u.weight_delta, u.n_samples) for u in updates]
        aggregated_delta = self._aggregate_fn(delta_updates)

        # Apply delta: w_global ← w_global + aggregated_delta
        new_weights: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for key in global_weights:
            new_weights[key] = (
                global_weights[key].float() + aggregated_delta[key].float()
            )

        self.model.set_weights(new_weights)
        return new_weights

    def run_round(self, round_idx: int) -> Tuple[List[ClientUpdate], Dict]:
        """
        Execute one complete FL round:
            broadcast → collect → aggregate → evaluate

        Parameters
        ----------
        round_idx : 1-based round index

        Returns
        -------
        (updates, metrics)
            updates : List[ClientUpdate]
            metrics : dict with keys loss, accuracy, avg_client_loss
        """
        self._log.info("─── Round %d / %d ───", round_idx,
                        self.config.training.rounds)

        # 1. Broadcast
        self.broadcast_weights()

        # 2. Collect
        updates = self.collect_updates()

        # 3. Aggregate
        self.aggregate(updates)

        # 4. Evaluate global model
        eval_metrics = self.evaluate()
        avg_client_loss = sum(u.local_loss for u in updates) / len(updates)

        metrics = {
            "fl_loss": eval_metrics["loss"],
            "fl_accuracy": eval_metrics["accuracy"],
            "avg_client_loss": avg_client_loss,
        }
        self._log.info(
            "Round %d: global_acc=%.4f  global_loss=%.4f  client_loss=%.4f",
            round_idx,
            metrics["fl_accuracy"],
            metrics["fl_loss"],
            metrics["avg_client_loss"],
        )
        return updates, metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current global model on the held-out test set.

        Returns
        -------
        dict: ``{'loss': float, 'accuracy': float}``
        """
        self.model.eval()
        ds = self._test_dataset
        loader = DataLoader(
            TensorDataset(ds.X_test, ds.y_test),
            batch_size=512,
            shuffle=False,
        )
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in loader:
            X_batch = to_device(X_batch, self.device)
            y_batch = to_device(y_batch, self.device)
            logits = self.model(X_batch)
            loss = self._criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def get_server(
    config,
    model: BaseModel,
    clients: List[FLClient],
    test_dataset: ClientDataset,
) -> FLServer:
    """
    Instantiate the correct server class based on config.

    ``config.training.aggregation`` options:
        ``'fedavg'``    → FLServer with FedAvg
        ``'fedmedian'`` → FLServer with FedMedian
        ``'ensemble'``  → EnsembleServer

    Returns
    -------
    FLServer or EnsembleServer
    """
    agg = getattr(config.training, "aggregation", "fedavg").lower()
    if agg == "ensemble":
        from server.ensemble import EnsembleServer
        return EnsembleServer(
            model=model,
            clients=clients,
            config=config,
            test_dataset=test_dataset,
        )
    return FLServer(
        model=model,
        clients=clients,
        config=config,
        test_dataset=test_dataset,
    )
