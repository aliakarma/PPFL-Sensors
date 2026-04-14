"""
utils/device.py
===============
Centralised torch.device management.

All model/tensor operations in the project route device selection through
this module so that switching between CPU and GPU requires changing exactly
one config key: ``training.device``.

Usage
-----
    from utils.device import get_device, to_device

    device = get_device()          # resolved once, cached
    tensor = to_device(tensor)     # moves any tensor/module to active device
"""

import logging
from typing import Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Module-level cached device (None until initialised)
_DEVICE: torch.device = None


def init_device(device_str: str = "auto") -> torch.device:
    """
    Initialise and cache the global device.

    Parameters
    ----------
    device_str : str
        ``"auto"``  — use CUDA if available, else CPU  (default)
        ``"cpu"``   — force CPU
        ``"cuda"``  — force CUDA (raises if unavailable)
        ``"cuda:N"``— specific GPU index

    Returns
    -------
    torch.device
    """
    global _DEVICE

    if device_str == "auto":
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available; falling back to CPU."
            )
            _DEVICE = torch.device("cpu")
        else:
            _DEVICE = torch.device(device_str)

    logger.info("Active device: %s", _DEVICE)
    return _DEVICE


def get_device() -> torch.device:
    """
    Return the cached device.  Initialises to ``"auto"`` on first call if
    ``init_device`` has not been called explicitly.
    """
    global _DEVICE
    if _DEVICE is None:
        init_device("auto")
    return _DEVICE


def to_device(
    obj: Union[torch.Tensor, nn.Module],
    device: torch.device = None,
) -> Union[torch.Tensor, nn.Module]:
    """
    Move a tensor or nn.Module to the active (or specified) device.

    Parameters
    ----------
    obj    : torch.Tensor or nn.Module
    device : torch.device, optional
        Defaults to the cached global device.

    Returns
    -------
    The same object moved to the target device.
    """
    if device is None:
        device = get_device()
    return obj.to(device)


def is_cuda_available() -> bool:
    """Convenience wrapper."""
    return torch.cuda.is_available()


def device_info() -> dict:
    """
    Return a dict of device diagnostics for logging / experiment tracking.
    """
    dev = get_device()
    info = {
        "device": str(dev),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available() and dev.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(dev)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(dev).total_memory / 1e9, 2
        )
    return info
