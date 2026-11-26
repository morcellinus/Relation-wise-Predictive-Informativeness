"""
Common layers shared across HGNN models.

- ``HeteroLinear``: per-type linear projection to a common dimension
- ``ClassifierHead``: dropout + linear → logits
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData


class HeteroLinear(nn.Module):
    """
    Project per-type features to a common hidden dimension.

    For node types whose feature dimension already equals ``out_channels``,
    the projection is skipped (identity).
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.linears = nn.ModuleDict()
        for ntype, in_c in in_channels_dict.items():
            self.linears[ntype] = nn.Linear(in_c, out_channels)

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = {}
        for ntype, x in x_dict.items():
            if ntype in self.linears:
                out[ntype] = self.linears[ntype](x)
            else:
                out[ntype] = x
        return out

    def reset_parameters(self):
        for lin in self.linears.values():
            lin.reset_parameters()


class ClassifierHead(nn.Module):
    """Final dropout + linear layer producing class logits."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.dropout(x))

    def reset_parameters(self):
        self.linear.reset_parameters()


def get_in_channels_dict(data: HeteroData) -> Dict[str, int]:
    """Extract ``{node_type: feature_dim}`` from a HeteroData object."""
    d = {}
    for ntype in data.node_types:
        if hasattr(data[ntype], "x") and data[ntype].x is not None:
            d[ntype] = data[ntype].x.size(-1)
        elif hasattr(data[ntype], "num_nodes"):
            d[ntype] = 0  # featureless — needs embedding
    return d
