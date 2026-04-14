"""tests/test_server.py — unit tests for FLServer and aggregation."""

import pytest
import numpy as np
import torch
from collections import OrderedDict
from types import SimpleNamespace

from server.aggregation import fedavg, fedmedian
from server.server import FLServer
from client.client import FLClient
from data.dataset_loader import ClientDataset
from models import get_model
from utils.seed import set_global_seed


def _make_weight_dict(val: float, shapes=None) -> OrderedDict:
    if shapes is None:
        shapes = {"layer.weight": (4, 4), "layer.bias": (4,)}
    return OrderedDict(
        {k: torch.full(s, val) for k, s in shapes.items()}
    )


def _make_config():
    return SimpleNamespace(
        model=SimpleNamespace(
            arch="mlp", input_dim=20, num_classes=3,
            hidden_dims=[16, 8], dropout=0.0,
        ),
        training=SimpleNamespace(
            rounds=3, local_epochs=1, batch_size=8,
            lr=0.01, optimizer="sgd", momentum=0.9,
            aggregation="fedavg", device="cpu",
        ),
        defense=SimpleNamespace(
            noise=SimpleNamespace(enabled=False, sigma=0.0),
            clipping=SimpleNamespace(enabled=False, max_norm=1.0),
        ),
        attack=SimpleNamespace(enabled=False, collect_rounds=1, eval_start_round=2),
        logging=SimpleNamespace(seed=42),
    )


def _make_dataset(cid=0, n=60, d=20, c=3):
    set_global_seed(42)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, c, n).astype(np.int64)
    return ClientDataset(X[:50], y[:50], X[50:], y[50:], client_id=cid)


# ── Aggregation tests ──────────────────────────────────────────────

def test_fedavg_equal_weights():
    """FedAvg of identical weights with equal samples → same weights."""
    w = _make_weight_dict(2.0)
    updates = [(w, 10), (w, 10)]
    result = fedavg(updates)
    for key in w:
        assert torch.allclose(result[key], w[key]), \
            f"FedAvg changed equal weights at {key}"


def test_fedavg_weighted_average():
    """FedAvg should weight by sample count."""
    w1 = _make_weight_dict(0.0)
    w2 = _make_weight_dict(4.0)
    # 25% from client1 (n=10), 75% from client2 (n=30) → expected = 3.0
    result = fedavg([(w1, 10), (w2, 30)])
    for key in result:
        expected = torch.full_like(result[key], 3.0)
        assert torch.allclose(result[key], expected, atol=1e-5), \
            f"FedAvg weighted average failed at {key}"


def test_fedmedian_three_clients():
    """FedMedian of [0, 2, 4]: odd count gives unambiguous median = 2."""
    w1 = _make_weight_dict(0.0)
    w2 = _make_weight_dict(2.0)
    w3 = _make_weight_dict(4.0)
    result = fedmedian([(w1, 10), (w2, 10), (w3, 10)])
    for key in result:
        expected = torch.full_like(result[key], 2.0)
        assert torch.allclose(result[key], expected, atol=1e-5), \
            f"FedMedian failed at {key}"


def test_fedavg_empty_raises():
    with pytest.raises(ValueError):
        fedavg([])


def test_fedmedian_empty_raises():
    with pytest.raises(ValueError):
        fedmedian([])


# ── FLServer tests ────────────────────────────────────────────────

def test_server_run_round_returns_metrics():
    set_global_seed(10)
    cfg = _make_config()
    ds_list = [_make_dataset(cid=i) for i in range(2)]
    test_ds = ds_list[0]
    global_model = get_model("mlp", input_dim=20, num_classes=3, hidden_dims=[16, 8])
    clients = [FLClient(ds.client_id, ds, cfg) for ds in ds_list]
    server = FLServer(global_model, clients, cfg, test_ds)

    updates, metrics = server.run_round(round_idx=1)

    assert len(updates) == 2, "Expected one update per client"
    assert "fl_accuracy" in metrics
    assert "fl_loss" in metrics
    assert 0.0 <= metrics["fl_accuracy"] <= 1.0
    assert metrics["fl_loss"] >= 0.0


def test_server_broadcast_updates_clients():
    """All clients must receive identical global weights after broadcast."""
    set_global_seed(11)
    cfg = _make_config()
    ds_list = [_make_dataset(cid=i) for i in range(3)]
    global_model = get_model("mlp", input_dim=20, num_classes=3, hidden_dims=[16, 8])
    clients = [FLClient(ds.client_id, ds, cfg) for ds in ds_list]
    server = FLServer(global_model, clients, cfg, ds_list[0])

    global_weights = server.broadcast_weights()

    for client in clients:
        client_weights = client.get_weights()
        for key in global_weights:
            assert torch.allclose(global_weights[key], client_weights[key]), \
                f"Client {client.client_id} has different weights at {key}"
