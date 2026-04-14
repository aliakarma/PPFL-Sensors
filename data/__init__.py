"""data — dataset loading and preprocessing for fl-privacy-project."""

from data.dataset_loader import (
    get_client_datasets,
    HARDatasetLoader,
    SyntheticLoader,
    ClientDataset,
)
from data.preprocessing import normalize, sliding_window, encode_labels

__all__ = [
    "get_client_datasets",
    "HARDatasetLoader",
    "SyntheticLoader",
    "ClientDataset",
    "normalize",
    "sliding_window",
    "encode_labels",
]
