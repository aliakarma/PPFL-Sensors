"""
server/aggregation.py
=====================
Stateless aggregation functions.

All functions take a list of (weight_dict, n_samples) pairs and return a
single aggregated weight dict.  The server calls these; they have no side
effects and no knowledge of clients.

Functions
---------
fedavg          — weighted average (McMahan 2017)
weighted_fedavg — alias of fedavg with explicit sample-count weighting
fedmedian       — coordinate-wise median (Yin et al. 2018, robust baseline)
"""

import logging
from collections import OrderedDict
from typing import List, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

WeightList = List[Tuple["OrderedDict[str, torch.Tensor]", int]]


def fedavg(updates: WeightList) -> "OrderedDict[str, torch.Tensor]":
    """
    Federated Averaging (McMahan et al., 2017).

    Computes a weighted average of client weight dicts, weighted by each
    client's number of training samples.

    Parameters
    ----------
    updates : list of (weight_dict, n_samples) tuples

    Returns
    -------
    Aggregated OrderedDict on CPU.
    """
    if not updates:
        raise ValueError("fedavg received an empty update list.")

    total_samples = sum(n for _, n in updates)
    if total_samples == 0:
        raise ValueError("Total sample count is 0 — cannot compute weighted average.")

    # Initialise accumulator with zeros matching the first weight dict
    reference_dict, _ = updates[0]
    aggregated: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, tensor in reference_dict.items():
        aggregated[key] = torch.zeros_like(tensor, dtype=torch.float32)

    for weight_dict, n_samples in updates:
        weight = n_samples / total_samples
        for key in aggregated:
            aggregated[key] += weight_dict[key].float() * weight

    logger.debug(
        "FedAvg: aggregated %d clients, total_samples=%d",
        len(updates), total_samples,
    )
    return aggregated


# Alias — identical logic, different name for clarity in config
weighted_fedavg = fedavg


def fedmedian(updates: WeightList) -> "OrderedDict[str, torch.Tensor]":
    """
    Coordinate-wise Median aggregation (Yin et al., 2018).

    More robust to Byzantine / outlier updates than FedAvg.
    Ignores sample counts (uniform aggregation).

    Parameters
    ----------
    updates : list of (weight_dict, n_samples) tuples
              (n_samples is ignored for median)

    Returns
    -------
    Aggregated OrderedDict on CPU.
    """
    if not updates:
        raise ValueError("fedmedian received an empty update list.")

    reference_dict, _ = updates[0]
    aggregated: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    for key in reference_dict:
        # Stack all client tensors along a new first axis: (n_clients, *shape)
        stacked = torch.stack(
            [wd[key].float() for wd, _ in updates], dim=0
        )
        aggregated[key] = stacked.median(dim=0).values

    logger.debug("FedMedian: aggregated %d clients", len(updates))
    return aggregated
