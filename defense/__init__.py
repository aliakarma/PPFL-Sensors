"""defense — privacy defense mechanisms."""

from defense.noise import add_gaussian_noise, compute_sensitivity
from defense.clipping import clip_gradients, adaptive_clip

__all__ = [
    "add_gaussian_noise",
    "compute_sensitivity",
    "clip_gradients",
    "adaptive_clip",
]
