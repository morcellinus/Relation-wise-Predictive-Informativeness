"""
RSHN — Relation Structure-Aware Heterogeneous Graph Neural Network
====================================================================

Source: Official repo (Zhu et al., ICDM 2019)
https://github.com/CheriseZhu/RSHN

Architecture:
- Build coarsened line graph from relation structures
- Initialize relation features via aggregation from incident edges
- Apply GNN (GCN) on the augmented graph with relation-aware features
- Message passing considers both node-level and relation-structure-level info

In this port, we simplify the coarsened line graph to a per-relation GCN with
relation-aware feature augmentation, preserving the core design of incorporating
relation structure information into node representations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from ..base import BaseHGNN
from ..common.layers import ClassifierHead, HeteroLinear, get_in_channels_dict


class RelationFeatureEncoder(nn.Module):
    """
    Encode relation structure features.

    For each relation type, compute a relation embedding by aggregating
    the features of nodes incident to that relation's edges.
    """

    def __init__(self, hidden_channels: int, num_relations: int):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, hidden_channels)
        self.proj = nn.Linear(hidden_channels * 2, hidden_channels)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        num_relations: int,
    ) -> Tensor:
        """
        Returns edge-level features incorporating relation structure.

        Returns
        -------
        Tensor (E, D) — relation-aware edge features.
        """
        src, dst = edge_index
        # Concatenate source features with relation embedding
        rel_feat = self.relation_emb(edge_type)  # (E, D)
        h_src = x[src]  # (E, D)
        edge_feat = self.proj(torch.cat([h_src, rel_feat], dim=-1))
        return edge_feat


class RSHNConv(nn.Module):
    """
    Single RSHN convolution layer.

    Performs GCN-style aggregation with relation-aware edge features.
    """

    def __init__(self, in_channels: int, out_channels: int, num_relations: int):
        super().__init__()
        self.W_node = nn.Linear(in_channels, out_channels)
        self.W_edge = nn.Linear(in_channels, out_channels)
        self.num_relations = num_relations

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : (N, D)
        edge_index : (2, E)
        edge_feat : (E, D)
        """
        src, dst = edge_index
        N = x.size(0)
        D_out = self.W_node.out_features

        # Transform
        h_node = self.W_node(x)  # (N, D_out)
        h_edge = self.W_edge(edge_feat)  # (E, D_out)

        # Aggregate: for each destination node, sum edge-weighted source messages
        msg = h_node[src] + h_edge  # (E, D_out) — relation-enhanced message

        # Degree normalization
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1.0)

        out = torch.zeros(N, D_out, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, D_out), msg)
        out = out / deg.unsqueeze(-1)

        # Add self-loop contribution
        out = out + h_node

        return out


class RSHN(BaseHGNN):
    """
    Relation Structure-aware Heterogeneous Graph Neural Network.

    Operates on a homogenized graph with relation-aware edge features.
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
        dropout: float = 0.5,
    ):
        super().__init__(target_type=target_type, num_classes=num_classes)
        self.in_channels = in_channels
        self.target_offset = target_offset
        self.target_num_nodes = target_num_nodes
        self.num_nodes_total = num_nodes_total
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Relation feature encoder
        self.rel_encoder = RelationFeatureEncoder(hidden_channels, num_relations)

        # RSHN conv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RSHNConv(hidden_channels, hidden_channels, num_relations))

        # Classifier
        self.classifier = ClassifierHead(hidden_channels, num_classes, dropout)

        self._cached_edge_index = None
        self._cached_edge_type = None

    def forward(self, data: HeteroData) -> Tensor:
        device = next(self.parameters()).device

        # Homogeneous conversion (cached)
        if self._cached_edge_index is None:
            from ..common.relation_utils import hetero_to_homogeneous
            ei, _, et, _, _, _ = hetero_to_homogeneous(data)
            self._cached_edge_index = ei.to(device)
            self._cached_edge_type = et.to(device)

        # Feature matrix
        offset = 0
        offsets = {}
        for nt in data.node_types:
            offsets[nt] = offset
            offset += data[nt].num_nodes

        from ..common.relation_utils import get_homogeneous_features
        x = get_homogeneous_features(data, offsets, offset, self.in_channels).to(device)
        x = self.input_proj(x)

        edge_index = self._cached_edge_index
        edge_type = self._cached_edge_type

        # Compute relation-aware edge features
        edge_feat = self.rel_encoder(
            x, edge_index, edge_type, len(data.edge_types)
        )

        # RSHN convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_feat)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                # Recompute edge features with updated node features
                edge_feat = self.rel_encoder(
                    x, edge_index, edge_type, len(data.edge_types)
                )

        target_x = x[self.target_offset: self.target_offset + self.target_num_nodes]
        return self.classifier(target_x)

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        target_type: str,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ) -> "RSHN":
        num_relations = len(data.edge_types)
        in_dict = get_in_channels_dict(data)
        in_channels = max(in_dict.values()) if in_dict else hidden_channels
        if in_channels == 0:
            in_channels = hidden_channels

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
            dropout=dropout,
        )
