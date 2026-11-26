"""
Abstract base class for all HGNN models in the benchmark.

Every model must subclass ``BaseHGNN`` and implement:
- ``forward(data) -> Tensor``  (logits for target node type)
- ``reset_parameters()``
"""
from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData


class BaseHGNN(nn.Module, metaclass=abc.ABCMeta):
    """
    Unified interface for heterogeneous graph neural networks.

    Attributes
    ----------
    target_type : str
        The node type for which predictions are made.
    num_classes : int
        Number of target classes.
    """

    def __init__(self, target_type: str, num_classes: int, **kwargs):
        super().__init__()
        self.target_type = target_type
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, data: HeteroData) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.

        Returns
        -------
        Tensor, shape (N_target, num_classes)
            Logits for the target node type.
        """
        ...

    def reset_parameters(self):
        """Re-initialise all learnable parameters."""
        for name, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def get_target_type(self) -> str:
        return self.target_type

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target={self.target_type!r}, "
            f"classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
