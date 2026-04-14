"""
defense/noise.py
================
Noise-based privacy defense.

The standard Gaussian mechanism adds N(0, sigma²) noise to each element of
the (already clipped) gradient, where sigma is calibrated relative to the
L2 sensitivity (= max_norm after clipping).

Reference: Dwork & Roth (2014), Abadi et al. (2016).
"""

import logging
from collections import OrderedDict

import torch

from utils.gradient_processing import (
    flatten_gradients,
    reconstruct_grad_dict,
)

logger = logging.getLogger(__name__)


def compute_sensitivity(max_norm: float) -> float:
    """
    Global L2 sensitivity of the gradient mechanism.

    After clipping to ``max_norm``, the sensitivity of any single client's
    update is exactly ``max_norm``.

    Parameters
    ----------
    max_norm : float — the clipping threshold applied before noise

    Returns
    -------
    float — sensitivity (= max_norm)
    """
    return float(max_norm)


def add_gaussian_noise(
    grad_dict: "OrderedDict[str, torch.Tensor]",
    sigma: float,
    clip_norm: float = None,
) -> "OrderedDict[str, torch.Tensor]":
    """
    Add zero-mean Gaussian noise with std = sigma to all gradient tensors.

    Operates in flat space (via gradient_processing) and reconstructs the
    original dict shape.

    Parameters
    ----------
    grad_dict  : OrderedDict of named gradient tensors
    sigma      : noise standard deviation.
                 If ``clip_norm`` is provided, the effective noise std is
                 ``sigma * clip_norm`` (standard noise parameterisation).
    clip_norm  : optional L2 clip threshold; multiplies ``sigma`` when given.

    Returns
    -------
    Noised OrderedDict (same keys, same shapes; new CPU tensors)
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0.0:
        logger.debug("sigma=0: no noise added")
        return grad_dict

    effective_sigma = sigma * clip_norm if clip_norm is not None else sigma

    flat = flatten_gradients(grad_dict)
    noise = torch.randn_like(flat) * effective_sigma
    noised_flat = flat + noise

    logger.debug(
        "Gaussian noise added: sigma=%.4f, effective_sigma=%.4f, "
        "grad_norm=%.4f, noise_norm=%.4f",
        sigma,
        effective_sigma,
        float(flat.norm(p=2).item()),
        float(noise.norm(p=2).item()),
    )

    return reconstruct_grad_dict(noised_flat, grad_dict)
