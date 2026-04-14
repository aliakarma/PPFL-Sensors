"""tests/test_client.py — unit tests for FLClient."""

import pytest
import numpy as np
import torch
from types import SimpleNamespace

from client.client import FLClient
from data.dataset_loader import ClientDataset
from utils.seed import set_global_seed


def _make_config(noise=False, clipping=False):
    return SimpleNamespace(
        model=SimpleNamespace(
            arch="mlp", input_dim=20, num_classes=3,
            hidden_dims=[16, 8], dropout=0.0,
        ),
        training=SimpleNamespace(
            rounds=2, local_epochs=1, batch_size=8,
            lr=0.01, optimizer="sgd", momentum=0.9,
        ),
        defense=SimpleNamespace(
            noise=SimpleNamespace(enabled=noise, sigma=0.1),
            clipping=SimpleNamespace(enabled=clipping, max_norm=1.0),
        ),
        attack=SimpleNamespace(
            enabled=False, collect_rounds=1, eval_start_round=2,
        ),
        logging=SimpleNamespace(seed=42),
    )


def _make_dataset(cid=0, n=50, d=20, c=3):
    set_global_seed(42)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randint(0, c, size=n).astype(np.int64)
    return ClientDataset(X[:40], y[:40], X[40:], y[40:], client_id=cid)


# ── Tests ──────────────────────────────────────────────────────────

def test_local_train_returns_correct_shape():
    set_global_seed(0)
    cfg = _make_config()
    ds = _make_dataset()
    client = FLClient(0, ds, cfg)
    update = client.local_train()

    # weight_delta keys must match model state dict
    model_keys = set(client.model.state_dict().keys())
    delta_keys = set(update.weight_delta.keys())
    assert model_keys == delta_keys, "weight_delta keys do not match model keys"

    # n_samples
    assert update.n_samples == len(ds)

    # gradient tensors are 1-D
    assert update.raw_gradients.dim() == 1
    assert update.defended_gradients.dim() == 1

    # raw and defended should be same shape
    assert update.raw_gradients.shape == update.defended_gradients.shape


def test_no_defense_raw_equals_defended():
    """With no defense, raw (weight_delta) and defended must be identical."""
    set_global_seed(1)
    cfg = _make_config(noise=False, clipping=False)
    ds = _make_dataset()
    client = FLClient(0, ds, cfg)
    update = client.local_train()
    assert torch.allclose(update.raw_gradients, update.defended_gradients), \
        "raw and defended should be equal when no defense is active"


def test_clipping_respects_max_norm():
    """After clipping, defended gradient norm must be ≤ max_norm."""
    set_global_seed(2)
    cfg = _make_config(noise=False, clipping=True)
    cfg.defense.clipping.max_norm = 0.5
    ds = _make_dataset()
    client = FLClient(0, ds, cfg)
    update = client.local_train()
    norm = float(update.defended_gradients.norm(p=2).item())
    assert norm <= cfg.defense.clipping.max_norm + 1e-5, \
        f"Clipped norm {norm:.6f} exceeds max_norm {cfg.defense.clipping.max_norm}"


def test_noise_changes_gradients():
    """Gaussian noise must change the defended gradients."""
    set_global_seed(3)
    cfg = _make_config(noise=True, clipping=False)
    ds = _make_dataset()
    client = FLClient(0, ds, cfg)
    update = client.local_train()
    diff = (update.raw_gradients - update.defended_gradients).abs().sum().item()
    assert diff > 0, "Noise injection did not change the gradient"


def test_set_weights_and_get_weights_roundtrip():
    """set_weights(get_weights()) must be a no-op."""
    set_global_seed(4)
    cfg = _make_config()
    ds = _make_dataset()
    client = FLClient(0, ds, cfg)
    w_before = client.get_weights()
    client.set_weights(w_before)
    w_after = client.get_weights()
    for key in w_before:
        assert torch.allclose(w_before[key], w_after[key]), \
            f"Weight mismatch at key: {key}"


def test_evaluate_returns_valid_metrics():
    set_global_seed(5)
    cfg = _make_config()
    ds = _make_dataset()
    client = FLClient(0, ds, cfg)
    metrics = client.evaluate()
    assert "loss" in metrics and "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] >= 0.0
