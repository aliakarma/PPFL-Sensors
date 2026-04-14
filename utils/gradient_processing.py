"""
utils/gradient_processing.py
=============================
Single source of truth for every operation on gradient tensors.

All consumers — client.py, noise.py, clipping.py, inference_attack.py —
import from here so that the gradient representation is always consistent.

Key functions
-------------
flatten_gradients      : OrderedDict → 1D Tensor
reconstruct_grad_dict  : 1D Tensor + reference dict → OrderedDict  (inverse)
normalize_gradient     : l2 / linf / zscore normalisation
compress_gradient      : top-k sparsification  (for storage_type='topk')
decompress_gradient    : reconstruct dense tensor from sparse representation
gradient_similarity    : cosine or L2 distance between two flat gradients
"""

import logging
from collections import OrderedDict
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Flatten / reconstruct  (must be exact inverses)
# ------------------------------------------------------------------

def flatten_gradients(
    grad_dict: "OrderedDict[str, torch.Tensor]",
) -> torch.Tensor:
    """
    Concatenate all gradient tensors in ``grad_dict`` into a single 1D tensor.

    Parameters
    ----------
    grad_dict : OrderedDict[str, Tensor]
        Named gradients as returned by ``model.named_parameters()`` after a
        backward pass, or a weight-delta dict.

    Returns
    -------
    torch.Tensor  shape (D,)  where D = sum of all parameter sizes
    """
    parts = []
    for name, tensor in grad_dict.items():
        if tensor is None:
            logger.debug("Gradient for '%s' is None — skipping", name)
            continue
        parts.append(tensor.detach().cpu().reshape(-1))
    if not parts:
        raise ValueError("grad_dict contains no non-None tensors.")
    return torch.cat(parts, dim=0)


def reconstruct_grad_dict(
    flat_grad: torch.Tensor,
    reference_dict: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """
    Inverse of ``flatten_gradients``.  Reshape ``flat_grad`` back to the
    shapes defined by ``reference_dict``.

    Parameters
    ----------
    flat_grad      : 1D Tensor of length D
    reference_dict : OrderedDict whose values define the target shapes.

    Returns
    -------
    OrderedDict[str, Tensor]
    """
    result = OrderedDict()
    offset = 0
    for name, ref_tensor in reference_dict.items():
        if ref_tensor is None:
            result[name] = None
            continue
        numel = ref_tensor.numel()
        chunk = flat_grad[offset: offset + numel]
        result[name] = chunk.reshape(ref_tensor.shape).to(ref_tensor.device)
        offset += numel

    if offset != flat_grad.numel():
        raise ValueError(
            f"flat_grad length ({flat_grad.numel()}) does not match "
            f"reference_dict total elements ({offset})."
        )
    return result


# ------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------

NormMethod = Literal["l2", "linf", "zscore", "none"]


def normalize_gradient(
    flat_grad: torch.Tensor,
    method: NormMethod = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalise a flat gradient vector.

    Parameters
    ----------
    flat_grad : 1D Tensor
    method    : one of ``'l2'``, ``'linf'``, ``'zscore'``, ``'none'``
    eps       : small constant to avoid division by zero

    Returns
    -------
    Normalised 1D Tensor (same shape, same device as input)
    """
    if method == "none":
        return flat_grad

    g = flat_grad.float()

    if method == "l2":
        norm = g.norm(p=2)
        return g / (norm + eps)

    if method == "linf":
        norm = g.abs().max()
        return g / (norm + eps)

    if method == "zscore":
        mean = g.mean()
        std = g.std()
        return (g - mean) / (std + eps)

    raise ValueError(f"Unknown normalisation method: '{method}'")


# ------------------------------------------------------------------
# Compression / decompression  (top-k sparsification)
# ------------------------------------------------------------------

def compress_gradient(
    flat_grad: torch.Tensor,
    ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k sparsification: keep only the ``ratio`` fraction of elements with
    the largest absolute values.

    Parameters
    ----------
    flat_grad : 1D Tensor of length D
    ratio     : fraction to keep, in (0, 1].  Default 0.1 → keep top 10 %.

    Returns
    -------
    (values, indices) — two 1D tensors of length k = ceil(D * ratio).
    Store these pairs instead of the full dense tensor to save disk space.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    k = max(1, int(np.ceil(flat_grad.numel() * ratio)))
    abs_vals = flat_grad.abs()
    indices = torch.topk(abs_vals, k, largest=True, sorted=False).indices
    values = flat_grad[indices]
    return values, indices


def decompress_gradient(
    values: torch.Tensor,
    indices: torch.Tensor,
    total_dim: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Reconstruct a dense gradient tensor from a (values, indices) sparse pair.

    Parameters
    ----------
    values    : 1D Tensor of kept values
    indices   : 1D LongTensor of kept indices
    total_dim : D (original gradient dimension)
    device    : target device (defaults to values.device)

    Returns
    -------
    Dense 1D Tensor of shape (total_dim,); un-kept elements are 0.
    """
    if device is None:
        device = values.device
    dense = torch.zeros(total_dim, dtype=values.dtype, device=device)
    dense[indices] = values
    return dense


# ------------------------------------------------------------------
# Similarity
# ------------------------------------------------------------------

SimilarityMetric = Literal["cosine", "l2"]


def gradient_similarity(
    g1: torch.Tensor,
    g2: torch.Tensor,
    metric: SimilarityMetric = "cosine",
    eps: float = 1e-8,
) -> float:
    """
    Scalar similarity between two flat gradient vectors.

    Parameters
    ----------
    g1, g2 : 1D Tensors of the same length
    metric : ``'cosine'`` (higher = more similar) or
             ``'l2'`` (lower = more similar, returned as *negative* L2 norm
             so that higher is always more similar)

    Returns
    -------
    float
    """
    if g1.shape != g2.shape:
        raise ValueError(
            f"Gradient shapes do not match: {g1.shape} vs {g2.shape}"
        )
    a = g1.float()
    b = g2.float()

    if metric == "cosine":
        dot = (a * b).sum()
        denom = a.norm() * b.norm() + eps
        return float(dot / denom)

    if metric == "l2":
        return -float((a - b).norm(p=2).item())

    raise ValueError(f"Unknown similarity metric: '{metric}'")


# ------------------------------------------------------------------
# Utility: compute gradient norm  (used by clipping.py)
# ------------------------------------------------------------------

def gradient_norm(
    grad_dict: "OrderedDict[str, Optional[torch.Tensor]]",
    p: int = 2,
) -> float:
    """
    Compute the p-norm of all gradients in a named-gradient dict.

    Parameters
    ----------
    grad_dict : OrderedDict of named gradient tensors (may contain Nones)
    p         : norm order (default L2)

    Returns
    -------
    float
    """
    flat = flatten_gradients(
        OrderedDict((k, v) for k, v in grad_dict.items() if v is not None)
    )
    return float(flat.norm(p=p).item())
