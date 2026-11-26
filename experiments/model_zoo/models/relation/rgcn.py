"""
R-GCN — Relational Graph Convolutional Network
================================================

Source: PyG built-in ``RGCNConv`` (Schlichtkrull et al., ESWC 2018)

Architecture:
- Converts HeteroData to homogeneous graph with relation type ids
- Stacks RGCNConv layers with configurable basis/block decomposition
- Extracts target-node logits from the homogeneous output
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

from ..base import BaseHGNN
from ..common.layers import ClassifierHead, get_in_channels_dict
from ..common.relation_utils import (
    get_homogeneous_features,
    hetero_to_homogeneous,
)


class RelationalGCN(BaseHGNN):
    """
    Relational Graph Convolutional Network using PyG's RGCNConv.

    Operates on a homogenized graph where each edge carries a relation-type index.
    Supports basis and block decomposition for parameter efficiency.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_relations: int,
        target_type: str,
        target_offset: int,
        target_num_nodes: int,
        num_nodes_total: int,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        dropout: float = 0.5,
    ):
        super().__init__(target_type=target_type, num_classes=num_classes)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.target_offset = target_offset
        self.target_num_nodes = target_num_nodes
        self.num_nodes_total = num_nodes_total
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(in_channels, hidden_channels, num_relations,
                     num_bases=num_bases)
        )
        for _ in range(1, num_layers):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations,
                         num_bases=num_bases)
            )

        self.classifier = ClassifierHead(hidden_channels, num_classes, dropout)

        # Cache homogeneous conversion results
        self._cached_edge_index = None
        self._cached_edge_type = None

    def _convert_graph(self, data: HeteroData):
        """Convert and cache the homogeneous graph."""
        if self._cached_edge_index is not None:
            return

        ei, _, et, _, _, _ = hetero_to_homogeneous(data)
        device = next(self.parameters()).device
        self._cached_edge_index = ei.to(device)
        self._cached_edge_type = et.to(device)

    def forward(self, data: HeteroData) -> Tensor:
        self._convert_graph(data)
        device = next(self.parameters()).device

        # Build homogeneous feature matrix
        offset = 0
        offsets = {}
        for nt in data.node_types:
            offsets[nt] = offset
            offset += data[nt].num_nodes

        x = get_homogeneous_features(
            data, offsets, offset, self.in_channels
        ).to(device)

        edge_index = self._cached_edge_index
        edge_type = self._cached_edge_type

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract target node embeddings
        target_x = x[self.target_offset: self.target_offset + self.target_num_nodes]
        return self.classifier(target_x)

    def invalidate_cache(self):
        """Clear cached homogeneous conversion."""
        self._cached_edge_index = None
        self._cached_edge_type = None

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        target_type: str,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_bases: Optional[int] = 30,
        dropout: float = 0.5,
        **kwargs,
    ) -> "RelationalGCN":
        # Compute metadata
        num_relations = len(data.edge_types)

        # Figure out feature dim (max non-zero across types, or hidden_channels)
        in_channels_dict = get_in_channels_dict(data)
        nonzero_dims = [v for v in in_channels_dict.values() if v > 0]
        in_channels = max(nonzero_dims) if nonzero_dims else hidden_channels

        # Find target offset
        offset = 0
        target_offset = 0
        for nt in data.node_types:
            if nt == target_type:
                target_offset = offset
            offset += data[nt].num_nodes

        return cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_relations=num_relations,
            target_type=target_type,
            target_offset=target_offset,
            target_num_nodes=data[target_type].num_nodes,
            num_nodes_total=offset,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout,
        )
