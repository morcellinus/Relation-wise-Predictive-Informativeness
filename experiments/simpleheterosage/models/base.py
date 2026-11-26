from __future__ import annotations

import abc

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData


class BaseHGNN(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, target_type: str, num_classes: int, **kwargs):
        super().__init__()
        self.target_type = target_type
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, data: HeteroData) -> Tensor:
        ...

    def reset_parameters(self):
        for name, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def get_target_type(self) -> str:
        return self.target_type

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target={self.target_type!r}, "
            f"classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
