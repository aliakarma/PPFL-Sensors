"""
data/dataset_loader.py
=======================
Load the UCI HAR dataset (or a synthetic fallback) and partition it across
federated clients using one of five strategies.

Partition strategies
--------------------
1. iid              — uniform random split                  (McMahan 2017)
2. dirichlet        — label-proportion skew via Dir(alpha)  (most used today)
3. pathological     — each client gets only K classes       (McMahan 2017 non-IID)
4. quantity_skew    — unequal sample counts via Dir(beta)
5. feature_skew     — per-client Gaussian feature noise

Public API
----------
    datasets = get_client_datasets(config)
    # → List of ClientDataset, one per client

    Each ClientDataset exposes:
        .X_train, .y_train  (np.ndarray)
        .X_test,  .y_test   (np.ndarray)
        .client_id          (int)
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing import normalize, add_feature_noise, encode_labels

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# ClientDataset  (thin PyTorch Dataset wrapper)
# ------------------------------------------------------------------

class ClientDataset(Dataset):
    """
    Holds one client's training partition.
    Compatible with torch.utils.data.DataLoader out of the box.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_id: int,
    ) -> None:
        self.X_train = torch.from_numpy(X_train).float()
        self.y_train = torch.from_numpy(y_train).long()
        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).long()
        self.client_id = client_id

    # Dataset interface operates on the training split
    def __len__(self) -> int:
        return len(self.X_train)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_train[idx], self.y_train[idx]

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y_train.max().item()) + 1

    def __repr__(self) -> str:
        return (
            f"ClientDataset(id={self.client_id}, "
            f"train={len(self.X_train)}, test={len(self.X_test)}, "
            f"input_dim={self.input_dim})"
        )


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

class HARDatasetLoader:
    """
    Load the UCI Human Activity Recognition dataset.

    Expected directory layout (``data_path``)::

        data/raw/
            UCI HAR Dataset/
                train/
                    X_train.txt
                    y_train.txt
                    subject_train.txt
                test/
                    X_test.txt
                    y_test.txt
                    subject_test.txt

    Download: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    """

    N_FEATURES = 561
    N_CLASSES = 6

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self._har_root = os.path.join(data_path, "UCI HAR Dataset")

    def _is_available(self) -> bool:
        required = [
            os.path.join(self._har_root, "train", "X_train.txt"),
            os.path.join(self._har_root, "train", "y_train.txt"),
            os.path.join(self._har_root, "test", "X_test.txt"),
            os.path.join(self._har_root, "test", "y_test.txt"),
        ]
        return all(os.path.exists(p) for p in required)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        (X_train, y_train, X_test, y_test)  all np.ndarray
        y labels are 0-indexed (original dataset uses 1-based labels)
        """
        if not self._is_available():
            raise FileNotFoundError(
                f"HAR dataset not found at {self._har_root}.  "
                "Please download it or set dataset.name='synthetic' in config."
            )

        def _load_file(rel_path: str) -> np.ndarray:
            full = os.path.join(self._har_root, rel_path)
            return np.loadtxt(full)

        X_train = _load_file("train/X_train.txt").astype(np.float32)
        y_train = _load_file("train/y_train.txt").astype(np.int64) - 1  # 0-based
        X_test = _load_file("test/X_test.txt").astype(np.float32)
        y_test = _load_file("test/y_test.txt").astype(np.int64) - 1    # 0-based

        # Normalise (fit on train, apply to both)
        X_train, X_test, _ = normalize(X_train, X_test, method="standard")

        logger.info(
            "HAR loaded: train=%d, test=%d, features=%d, classes=%d",
            len(X_train), len(X_test), X_train.shape[1], len(np.unique(y_train)),
        )
        return X_train, y_train, X_test, y_test


class SyntheticLoader:
    """
    Generate a synthetic classification dataset when HAR is unavailable.
    Uses ``sklearn.datasets.make_classification`` with sensor-like parameters.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_features: int = 561,
        n_classes: int = 6,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        logger.warning(
            "HAR dataset not found.  Using synthetic data "
            "(n_samples=%d, n_features=%d, n_classes=%d).",
            self.n_samples, self.n_features, self.n_classes,
        )

        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=max(20, self.n_features // 5),
            n_redundant=self.n_features // 10,
            n_classes=self.n_classes,
            n_clusters_per_class=1,
            random_state=self.seed,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        X_train, X_test, _ = normalize(X_train, X_test, method="standard")
        return X_train, y_train, X_test, y_test


# ------------------------------------------------------------------
# Partition strategies
# ------------------------------------------------------------------

def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Shuffle and split uniformly across n_clients."""
    indices = rng.permutation(len(X))
    splits = np.array_split(indices, n_clients)
    return [(X[idx], y[idx]) for idx in splits]


def partition_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_samples: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Label-skew via Dirichlet(alpha).

    Lower alpha → higher heterogeneity.
    alpha=100 approximates IID; alpha=0.1 creates strong label skew.
    """
    classes = np.unique(y)
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        # Draw proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        # exact allocation
        splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)
        splits = np.insert(splits, 0, 0)
        splits[-1] = len(cls_idx)
        
        for cid in range(n_clients):
            client_indices[cid].extend(cls_idx[splits[cid]:splits[cid+1]].tolist())

    return [(X[np.array(idx)], y[np.array(idx)]) for idx in client_indices]

def partition_pathological(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    classes_per_client: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Each client receives data from exactly ``classes_per_client`` classes.
    Original McMahan (2017) non-IID setup.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    if classes_per_client > n_classes:
        logger.warning(
            "classes_per_client=%d > n_classes=%d; clamping.",
            classes_per_client, n_classes,
        )
        classes_per_client = n_classes

    # Build per-class index lists
    class_indices = {cls: np.where(y == cls)[0].tolist() for cls in classes}
    for cls in classes:
        rng.shuffle(class_indices[cls])

    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    # Assign classes to clients and divide chunks cleanly without duplication
    client_classes = [rng.choice(classes, classes_per_client, replace=False) for _ in range(n_clients)]
    
    # Calculate how many clients need each class to split fairly
    class_splits = {cls: 0 for cls in classes}
    for cid in range(n_clients):
        for cls in client_classes[cid]:
            class_splits[cls] += 1
            
    # Track current split offset for each class
    class_current_offset = {cls: 0 for cls in classes}

    for cid in range(n_clients):
        for cls in client_classes[cid]:
            pool = class_indices[cls]
            num_splits = class_splits[cls]
            
            # Divide into non-overlapping chunks
            chunk_size = len(pool) // num_splits
            start_idx = class_current_offset[cls] * chunk_size
            end_idx = start_idx + chunk_size
            
            # If it's the last split for this class, give remainder
            if class_current_offset[cls] == num_splits - 1:
                end_idx = len(pool)
                
            client_indices[cid].extend(pool[start_idx:end_idx])
            class_current_offset[cls] += 1

    return [(X[np.array(idx)], y[np.array(idx)]) for idx in client_indices]


def partition_quantity_skew(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    beta: float,
    rng: np.random.Generator,
    min_samples: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Unequal sample counts drawn from Dirichlet(beta).
    beta=1 → moderate skew; beta=0.1 → extreme size differences.
    """
    proportions = rng.dirichlet(np.repeat(beta, n_clients))
    sizes = (proportions * len(X)).astype(int)
    sizes[-1] = len(X) - sizes[:-1].sum()  # fix rounding

    # Ensure minimum
    for i in range(n_clients):
        if sizes[i] < min_samples:
            deficit = min_samples - sizes[i]
            donor = sizes.argmax()
            sizes[i] += deficit
            sizes[donor] -= deficit

    indices = rng.permutation(len(X))
    result = []
    start = 0
    for size in sizes:
        end = start + size
        idx = indices[start:end]
        result.append((X[idx], y[idx]))
        start = end

    return result


def partition_feature_skew(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    noise_std_per_client: Optional[List[float]],
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    IID label distribution but with per-client Gaussian feature noise.
    Simulates sensor calibration differences across devices.
    """
    if noise_std_per_client is None:
        noise_std_per_client = [0.1 * (i + 1) for i in range(n_clients)]

    base_partitions = partition_iid(X, y, n_clients, rng)
    result = []
    for cid, (Xc, yc) in enumerate(base_partitions):
        std = noise_std_per_client[cid % len(noise_std_per_client)]
        Xc_noisy = add_feature_noise(Xc, std, rng)
        result.append((Xc_noisy, yc))
    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_client_datasets(config) -> List[ClientDataset]:
    """
    Top-level factory: load data and partition into per-client datasets.

    Parameters
    ----------
    config : SimpleNamespace from load_config()
        Reads: config.dataset.{name, path, n_clients, partition_strategy,
               partition_params, test_split}

    Returns
    -------
    List[ClientDataset]  length = config.dataset.n_clients
    """
    ds_cfg = config.dataset
    n_clients: int = ds_cfg.n_clients
    strategy: str = getattr(ds_cfg, "partition_strategy", "iid")
    params: dict = getattr(ds_cfg, "partition_params", {})
    if not isinstance(params, dict):
        from utils.config import config_to_dict
        params = config_to_dict(params)

    seed: int = getattr(config.logging, "seed", 42)
    rng = np.random.default_rng(seed)

    # 1. Load raw data
    dataset_name = getattr(ds_cfg, "name", "synthetic")
    if dataset_name == "har":
        try:
            X_all, y_all, X_test_global, y_test_global = HARDatasetLoader(
                ds_cfg.path
            ).load()
        except FileNotFoundError:
            raise RuntimeError("HAR dataset required. Please run scripts/download_har.py to download it.")
    else:
        raise RuntimeError("HAR dataset required. Set dataset.name='har' in config and run scripts/download_har.py to download it.")

    # 2. Partition training data
    logger.info("Partitioning %d samples across %d clients using '%s'",
                len(X_all), n_clients, strategy)

    if strategy == "iid":
        partitions = partition_iid(X_all, y_all, n_clients, rng)

    elif strategy == "dirichlet":
        alpha = params.get("alpha", 0.5)
        partitions = partition_dirichlet(X_all, y_all, n_clients, alpha, rng)

    elif strategy == "pathological":
        cpc = params.get("classes_per_client", 2)
        partitions = partition_pathological(X_all, y_all, n_clients, cpc, rng)

    elif strategy == "quantity_skew":
        beta = params.get("beta", 0.5)
        partitions = partition_quantity_skew(X_all, y_all, n_clients, beta, rng)

    elif strategy == "feature_skew":
        noise_stds = params.get("noise_std_per_client", None)
        partitions = partition_feature_skew(X_all, y_all, n_clients, noise_stds, rng)

    else:
        raise ValueError(
            f"Unknown partition strategy: '{strategy}'.  "
            "Choose from: iid, dirichlet, pathological, quantity_skew, feature_skew"
        )

    # 3. Wrap in ClientDataset objects
    test_split = getattr(ds_cfg, "test_split", 0.2)
    datasets = []
    for cid, (Xc, yc) in enumerate(partitions):
        # Each client uses the global test set for FL evaluation
        # (standard FL evaluation protocol)
        ds = ClientDataset(
            X_train=Xc,
            y_train=yc,
            X_test=X_test_global,
            y_test=y_test_global,
            client_id=cid,
        )
        datasets.append(ds)
        logger.debug("Client %d: %s", cid, ds)

    return datasets
