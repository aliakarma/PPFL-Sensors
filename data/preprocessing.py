"""
data/preprocessing.py
======================
Stateless preprocessing utilities for sensor data.

All functions operate on NumPy arrays and return NumPy arrays so they
compose cleanly with both the HAR loader and the synthetic fallback.
"""

import logging

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "standard",
) -> tuple:
    """
    Fit a scaler on X_train and apply to both splits.

    Parameters
    ----------
    X_train : (N_train, D)
    X_test  : (N_test,  D)
    method  : ``'standard'`` (z-score) or ``'minmax'``

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalisation method: '{method}'")

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc.astype(np.float32), X_test_sc.astype(np.float32), scaler


def sliding_window(
    X: np.ndarray,
    window_size: int = 128,
    stride: int = 64,
) -> np.ndarray:
    """
    Apply a sliding window over the time axis of a 2D signal array.

    Parameters
    ----------
    X           : (T, C) — T timesteps, C channels
    window_size : number of timesteps per window
    stride      : step between windows

    Returns
    -------
    (N_windows, C, window_size)
    """
    windows = []
    T = X.shape[0]
    start = 0
    while start + window_size <= T:
        windows.append(X[start: start + window_size].T)  # → (C, window_size)
        start += stride
    if not windows:
        raise ValueError(
            f"Signal length {T} is shorter than window_size {window_size}."
        )
    return np.stack(windows, axis=0).astype(np.float32)


def encode_labels(y: np.ndarray) -> tuple:
    """
    Encode string or integer labels to contiguous 0-based integers.

    Returns
    -------
    (y_encoded, label_encoder)
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y).astype(np.int64)
    return y_enc, le


def add_feature_noise(
    X: np.ndarray,
    noise_std: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add Gaussian feature noise (used for feature-skew non-IID partition).

    Parameters
    ----------
    X         : (N, D) float array
    noise_std : standard deviation of the Gaussian noise
    rng       : optional NumPy Generator for reproducibility

    Returns
    -------
    Noised array of same shape
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
    return X + noise
