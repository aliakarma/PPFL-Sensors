"""
utils/seed.py
=============
Global, centralised, deterministic seed control.

Ensures Python random, NumPy, PyTorch (CPU + CUDA), and DataLoader workers
all share the same reproducible RNG state.  One call → full reproducibility.

Usage
-----
    from utils.seed import set_global_seed, SeedContext, worker_init_fn

    set_global_seed(42)

    # Temporarily isolate a block's randomness:
    with SeedContext(seed=99):
        ...  # uses seed 99; previous seed restored on exit

    # Pass to DataLoader for deterministic data loading:
    DataLoader(dataset, worker_init_fn=worker_init_fn, generator=...)
"""

import os
import random
import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Module-level state: the currently active seed (-1 = not set)
_CURRENT_SEED: int = -1


def set_global_seed(seed: int) -> None:
    """
    Seed every RNG used in this project.

    Parameters
    ----------
    seed : int
        The master seed.  Stored internally so ``get_seed()`` can retrieve it.
    """
    global _CURRENT_SEED
    _CURRENT_SEED = seed

    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN ops (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hash randomisation (affects dict iteration order in some Python versions)
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug("Global seed set to %d", seed)


def get_seed() -> int:
    """Return the currently active seed, or -1 if ``set_global_seed`` has not
    been called yet."""
    return _CURRENT_SEED


@contextmanager
def SeedContext(seed: int):
    """
    Context manager that temporarily overrides the global seed, then restores
    the previous seed state on exit.

    Useful for isolating the attack model's training randomness from the FL
    training seed sequence so the two do not interfere.

    Parameters
    ----------
    seed : int
        Temporary seed to use inside the ``with`` block.

    Example
    -------
        with SeedContext(seed=99):
            attack_model.fit(X_train, y_train)
        # FL seed sequence is unaffected
    """
    global _CURRENT_SEED
    previous_seed = _CURRENT_SEED
    set_global_seed(seed)
    try:
        yield
    finally:
        if previous_seed != -1:
            set_global_seed(previous_seed)
        else:
            _CURRENT_SEED = -1


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader ``worker_init_fn``.

    Each worker gets a unique but deterministic seed derived from the global
    seed and its worker index.  Pass this to every ``DataLoader``:

        DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)

    Parameters
    ----------
    worker_id : int
        Automatically supplied by PyTorch (0-indexed worker index).
    """
    base_seed = get_seed()
    if base_seed == -1:
        # No global seed set; use worker_id alone for some determinism
        base_seed = 0
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_torch_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a seeded ``torch.Generator`` for use with DataLoader's
    ``generator`` argument (PyTorch ≥ 1.9).

    Parameters
    ----------
    seed : int, optional
        Seed for the generator.  Defaults to the current global seed.
    """
    if seed is None:
        seed = get_seed()
    if seed == -1:
        seed = 0
    g = torch.Generator()
    g.manual_seed(seed)
    return g
