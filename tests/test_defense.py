"""tests/test_defense.py — unit tests for noise and clipping defenses."""

import pytest
import torch
import numpy as np
from collections import OrderedDict

from defense.noise import add_gaussian_noise, compute_sensitivity
from defense.clipping import clip_gradients, adaptive_clip
from utils.gradient_processing import (
    flatten_gradients,
    reconstruct_grad_dict,
    normalize_gradient,
    compress_gradient,
    decompress_gradient,
    gradient_similarity,
)
from utils.seed import set_global_seed


# ── Fixtures ──────────────────────────────────────────────────────

def _grad_dict(scale: float = 1.0, dim: int = 50) -> OrderedDict:
    """Create a simple named gradient dict for testing."""
    set_global_seed(0)
    return OrderedDict({
        "layer1.weight": torch.randn(10, 10) * scale,
        "layer1.bias":   torch.randn(10) * scale,
        "layer2.weight": torch.randn(5, 5) * scale,
        "layer2.bias":   torch.randn(5) * scale,
    })


# ── defense/clipping.py ───────────────────────────────────────────

class TestClipping:
    def test_clip_reduces_norm(self):
        """After clipping, L2 norm must not exceed max_norm."""
        gd = _grad_dict(scale=10.0)
        max_norm = 1.0
        clipped = clip_gradients(gd, max_norm=max_norm)
        flat = flatten_gradients(clipped)
        norm = float(flat.norm(p=2).item())
        assert norm <= max_norm + 1e-5, f"Norm {norm:.6f} exceeds max_norm {max_norm}"

    def test_clip_no_change_below_threshold(self):
        """Gradient already below max_norm must be unchanged."""
        set_global_seed(0)
        gd = _grad_dict(scale=0.01)
        flat_before = flatten_gradients(gd)
        norm_before = float(flat_before.norm(p=2).item())
        max_norm = 100.0   # well above the tiny gradient
        clipped = clip_gradients(gd, max_norm=max_norm)
        flat_after = flatten_gradients(clipped)
        assert torch.allclose(flat_before, flat_after, atol=1e-6), \
            "Gradient should not change when norm < max_norm"

    def test_clip_preserves_shape(self):
        gd = _grad_dict()
        clipped = clip_gradients(gd, max_norm=1.0)
        for key in gd:
            assert gd[key].shape == clipped[key].shape

    def test_clip_invalid_max_norm_raises(self):
        gd = _grad_dict()
        with pytest.raises(ValueError):
            clip_gradients(gd, max_norm=0.0)
        with pytest.raises(ValueError):
            clip_gradients(gd, max_norm=-1.0)

    def test_adaptive_clip_with_history(self):
        gd = _grad_dict(scale=2.0)
        history = [0.5, 0.8, 1.2, 0.6, 0.9]
        clipped = adaptive_clip(gd, target_quantile=0.5, history_norms=history)
        flat = flatten_gradients(clipped)
        median_norm = float(np.median(history))
        assert float(flat.norm(p=2).item()) <= median_norm + 1e-5


# ── defense/noise.py ─────────────────────────────────────────────

class TestNoise:
    def test_noise_changes_gradient(self):
        set_global_seed(1)
        gd = _grad_dict(scale=1.0)
        flat_before = flatten_gradients(gd)
        noised = add_gaussian_noise(gd, sigma=0.1)
        flat_after = flatten_gradients(noised)
        diff = (flat_before - flat_after).abs().sum().item()
        assert diff > 0.0, "Noise injection had no effect"

    def test_zero_sigma_is_noop(self):
        gd = _grad_dict()
        flat_before = flatten_gradients(gd)
        noised = add_gaussian_noise(gd, sigma=0.0)
        flat_after = flatten_gradients(noised)
        assert torch.allclose(flat_before, flat_after), \
            "sigma=0 should not change gradients"

    def test_noise_preserves_shape(self):
        gd = _grad_dict()
        noised = add_gaussian_noise(gd, sigma=0.05)
        for key in gd:
            assert gd[key].shape == noised[key].shape

    def test_negative_sigma_raises(self):
        gd = _grad_dict()
        with pytest.raises(ValueError):
            add_gaussian_noise(gd, sigma=-0.1)

    def test_compute_sensitivity_equals_max_norm(self):
        assert compute_sensitivity(1.5) == pytest.approx(1.5)

    def test_larger_sigma_more_noise(self):
        """Mean absolute noise should increase with sigma."""
        set_global_seed(42)
        gd = _grad_dict()
        flat = flatten_gradients(gd)

        set_global_seed(10)
        n1 = flatten_gradients(add_gaussian_noise(gd, sigma=0.01))
        set_global_seed(10)
        n2 = flatten_gradients(add_gaussian_noise(gd, sigma=1.0))

        diff1 = (flat - n1).abs().mean().item()
        diff2 = (flat - n2).abs().mean().item()
        assert diff2 > diff1, "Larger sigma must produce more noise"


# ── utils/gradient_processing.py ─────────────────────────────────

class TestGradientProcessing:
    def test_flatten_reconstruct_roundtrip(self):
        gd = _grad_dict()
        flat = flatten_gradients(gd)
        reconstructed = reconstruct_grad_dict(flat, gd)
        for key in gd:
            assert torch.allclose(gd[key], reconstructed[key], atol=1e-6), \
                f"Roundtrip failed at {key}"

    def test_flatten_length(self):
        gd = _grad_dict()
        expected_len = sum(v.numel() for v in gd.values())
        flat = flatten_gradients(gd)
        assert flat.numel() == expected_len

    def test_reconstruct_wrong_length_raises(self):
        gd = _grad_dict()
        flat = flatten_gradients(gd)
        bad_flat = torch.zeros(flat.numel() + 5)  # wrong length
        with pytest.raises(ValueError):
            reconstruct_grad_dict(bad_flat, gd)

    def test_normalize_l2_unit_norm(self):
        flat = torch.randn(100)
        normed = normalize_gradient(flat, method="l2")
        norm = float(normed.norm(p=2).item())
        assert abs(norm - 1.0) < 1e-5, f"L2 norm should be 1.0, got {norm}"

    def test_normalize_linf_max_one(self):
        flat = torch.randn(100)
        normed = normalize_gradient(flat, method="linf")
        max_val = float(normed.abs().max().item())
        assert abs(max_val - 1.0) < 1e-5, f"Linf max should be 1.0, got {max_val}"

    def test_normalize_zscore_zero_mean(self):
        flat = torch.randn(100) * 5 + 3
        normed = normalize_gradient(flat, method="zscore")
        mean = float(normed.mean().item())
        assert abs(mean) < 1e-4, f"Z-score mean should be ~0, got {mean}"

    def test_normalize_none_is_identity(self):
        flat = torch.randn(50)
        normed = normalize_gradient(flat, method="none")
        assert torch.allclose(flat, normed)

    def test_normalize_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_gradient(torch.randn(10), method="l7")

    def test_compress_decompress_roundtrip(self):
        set_global_seed(0)
        flat = torch.randn(200)
        values, indices = compress_gradient(flat, ratio=0.3)
        reconstructed = decompress_gradient(values, indices, total_dim=200)

        # Only top-k elements are preserved; the rest are zero
        assert reconstructed.numel() == 200
        # Top-k positions must be preserved exactly
        assert torch.allclose(reconstructed[indices], values, atol=1e-6)

    def test_compress_ratio_invalid_raises(self):
        flat = torch.randn(50)
        with pytest.raises(ValueError):
            compress_gradient(flat, ratio=0.0)
        with pytest.raises(ValueError):
            compress_gradient(flat, ratio=1.5)

    def test_gradient_similarity_cosine_self(self):
        flat = torch.randn(50)
        sim = gradient_similarity(flat, flat, metric="cosine")
        assert abs(sim - 1.0) < 1e-5, "Cosine similarity with self should be 1.0"

    def test_gradient_similarity_l2_self(self):
        flat = torch.randn(50)
        sim = gradient_similarity(flat, flat, metric="l2")
        assert abs(sim - 0.0) < 1e-5, "L2 distance with self should be 0 (returned as 0)"

    def test_gradient_similarity_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            gradient_similarity(torch.randn(10), torch.randn(20))


# ── utils/metrics.py ─────────────────────────────────────────────

class TestPrivacyScore:
    def test_privacy_score_formula(self):
        from utils.metrics import compute_privacy_score, compute_normalized_attacker_advantage
        assert compute_privacy_score(0.0) == pytest.approx(1.0)
        assert compute_privacy_score(1.0) == pytest.approx(0.0)
        assert compute_privacy_score(0.4) == pytest.approx(0.6)

    def test_privacy_score_clamped(self):
        from utils.metrics import compute_privacy_score, compute_normalized_attacker_advantage
        assert compute_privacy_score(1.5) == pytest.approx(0.0)
        assert compute_privacy_score(-0.5) == pytest.approx(1.0)

    def test_random_baseline(self):
        from utils.metrics import random_baseline_accuracy
        assert random_baseline_accuracy(10) == pytest.approx(0.1)
        assert random_baseline_accuracy(1) == pytest.approx(1.0)

    def test_normalized_attacker_advantage(self):
        from utils.metrics import compute_normalized_attacker_advantage
        # NAA=0 when attack equals random
        assert compute_normalized_attacker_advantage(0.1, 0.1) == pytest.approx(0.0)
        # NAA=1 when attack is perfect
        assert compute_normalized_attacker_advantage(1.0, 0.1) == pytest.approx(1.0)
        # NAA=0.5 when attack is halfway between random and perfect
        expected = (0.55 - 0.1) / (1.0 - 0.1)
        assert compute_normalized_attacker_advantage(0.55, 0.1) == pytest.approx(expected, abs=1e-5)
