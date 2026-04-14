"""
models/base_model.py
====================
Abstract base class enforcing the FL model interface.

All FL models must implement ``forward()``.  The base class provides
``get_weights()``, ``set_weights()``, and ``get_gradients()`` so the FL
infrastructure (client, aggregation, attack) never calls model-specific APIs.
"""

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base for all task models used in federated training.

    Sub-classes must implement ``forward(x)``.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim) or (batch, channels, seq_len)

        Returns
        -------
        logits : (batch, num_classes)
        """
        ...

    # ------------------------------------------------------------------
    # Weight management — used by FLServer for aggregation
    # ------------------------------------------------------------------

    def get_weights(self) -> "OrderedDict[str, torch.Tensor]":
        """Return a deep copy of the model's state dict (CPU tensors)."""
        return OrderedDict(
            {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}
        )

    def set_weights(self, weights: "OrderedDict[str, torch.Tensor]") -> None:
        """
        Load weights into the model.

        Parameters
        ----------
        weights : OrderedDict matching ``self.state_dict()``
        """
        self.load_state_dict(weights, strict=True)

    def get_weight_delta(
        self,
        initial_weights: "OrderedDict[str, torch.Tensor]",
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Compute the weight update delta = current_weights − initial_weights.
        Used by FLClient to send pseudo-gradients to the server.

        Parameters
        ----------
        initial_weights : weights at the start of local training (CPU tensors)

        Returns
        -------
        OrderedDict of delta tensors (CPU)
        """
        current = self.get_weights()
        delta = OrderedDict()
        for key in current:
            delta[key] = current[key] - initial_weights[key].to(current[key].device)
        return delta

    # ------------------------------------------------------------------
    # Gradient management — used by attack + defense modules
    # ------------------------------------------------------------------

    def get_gradients(self) -> "OrderedDict[str, Optional[torch.Tensor]]":
        """
        Collect .grad tensors for all named parameters after a backward pass.

        Returns
        -------
        OrderedDict[str, Tensor | None]
            None entries indicate parameters that did not receive a gradient
            (e.g. frozen layers).
        """
        grads = OrderedDict()
        for name, param in self.named_parameters():
            grads[name] = (
                param.grad.detach().clone().cpu() if param.grad is not None else None
            )
        return grads

    def zero_gradients(self) -> None:
        """Zero all parameter gradients (convenience wrapper)."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def clone(self) -> "BaseModel":
        """Return a deep copy of this model (weights + architecture)."""
        return copy.deepcopy(self)
