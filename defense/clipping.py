"""
defense/clipping.py
====================
Gradient clipping defense.

Clipping bounds the sensitivity of each client's update, which is a
prerequisite for differential privacy noise calibration and also a standalone
defense that limits how much any single update can shift the global model.

All flattening/reconstruction delegates to ``utils/gradient_processing.py``.
"""

import logging
from collections import OrderedDict
from typing import Union

import numpy as np
import torch

from utils.gradient_processing import (
    flatten_gradients,
    reconstruct_grad_dict,
    gradient_norm,
)

logger = logging.getLogger(__name__)


def clip_gradients(
    grad_dict: "OrderedDict[str, torch.Tensor]",
    max_norm: float,
) -> "OrderedDict[str, torch.Tensor]":
    """
    Clip a named-gradient dict so its L2 norm ≤ ``max_norm``.

    This is the standard per-update clipping used in DP-SGD and FL defenses.

    Parameters
    ----------
    grad_dict : OrderedDict of named gradient tensors
    max_norm  : L2 norm upper bound (must be > 0)

    Returns
    -------
    Clipped OrderedDict (same keys, same shapes; new tensors on CPU)
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")

    flat = flatten_gradients(grad_dict)
    current_norm = float(flat.norm(p=2).item())

    if current_norm > max_norm:
        scale = max_norm / (current_norm + 1e-8)
        flat = flat * scale
        logger.debug(
            "Gradient clipped: norm %.4f → %.4f (scale=%.4f)",
            current_norm, float(flat.norm(p=2).item()), scale,
        )
    else:
        logger.debug("Gradient norm %.4f ≤ max_norm %.4f; no clipping applied",
                     current_norm, max_norm)

    return reconstruct_grad_dict(flat, grad_dict)


def adaptive_clip(
    grad_dict: "OrderedDict[str, torch.Tensor]",
    target_quantile: float = 0.5,
    history_norms: list = None,
) -> "OrderedDict[str, torch.Tensor]":
    """
    Adaptive clipping: set max_norm to the ``target_quantile`` of recent
    gradient norms.  If no history is provided, falls back to a single-update
    estimate (current norm × target_quantile).

    Parameters
    ----------
    grad_dict        : OrderedDict of named gradient tensors
    target_quantile  : quantile of history to use as clip threshold (0–1)
    history_norms    : list of recent gradient L2 norms for quantile estimation

    Returns
    -------
    Clipped OrderedDict
    """
    if not (0.0 < target_quantile <= 1.0):
        raise ValueError(f"target_quantile must be in (0, 1], got {target_quantile}")

    if history_norms and len(history_norms) > 0:
        max_norm = float(np.quantile(history_norms, target_quantile))
    else:
        current_norm = gradient_norm(grad_dict, p=2)
        max_norm = current_norm * target_quantile

    max_norm = max(max_norm, 1e-6)  # safety floor
    return clip_gradients(grad_dict, max_norm=max_norm)
