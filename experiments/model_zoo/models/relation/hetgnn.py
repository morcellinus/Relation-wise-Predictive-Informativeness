"""
HetGNN — Heterogeneous Graph Neural Network
=============================================

Source: Official repo (Zhang et al., KDD 2019)
https://github.com/chuxuzhang/KDD2019_HetGNN

Architecture:
- Per-type content encoder: transforms raw features into type-specific embeddings
- Type-level neighbor aggregation: for each edge type, aggregate neighbor features
  using attention-weighted mean
- Cross-type fusion: attention-weighted combination of per-type aggregated representations

This is a faithful port that adapts the architecture to work with PyG HeteroData,
using per-relation message passing instead of the original custom batching.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

from ..base import BaseHGNN
from ..common.layers import ClassifierHead, HeteroLinear, get_in_channels_dict


class TypeContentEncoder(nn.Module):
    """
    Per-type content encoder.

    Original HetGNN uses BiLSTM for content encoding. For simplicity and
    compatibility, we use a 2-layer MLP which is equivalent for fixed-length
    feature vectors (BiLSTM is needed only for variable-length content like
    word sequences).
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TypeNeighborAggregator(nn.Module):
    """
    Aggregate neighbor features from a specific edge type using attention.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x_src : Tensor (N_src, D)
        x_dst : Tensor (N_dst, D)
        edge_index : Tensor (2, E)

        Returns
        -------
        Tensor (N_dst, D) — aggregated neighbor features.
        """
        src_idx, dst_idx = edge_index
        N_dst = x_dst.size(0)
        D = x_src.size(1)

        # Attention: [h_neighbor || h_self]
        h_pair = torch.cat([x_src[src_idx], x_dst[dst_idx]], dim=-1)
        alpha = self.attn(h_pair).squeeze(-1)  # (E,)
        alpha = softmax(alpha, dst_idx, num_nodes=N_dst)

        # Weighted aggregation
        msg = x_src[src_idx] * alpha.unsqueeze(-1)
        out = torch.zeros(N_dst, D, device=x_src.device)
        out.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, D), msg)

        return out


class CrossTypeFusion(nn.Module):
    """
    Attention-weighted fusion of per-type aggregated representations.
    """

    def __init__(self, hidden_channels: int, num_types: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, z_list: List[Tensor]) -> Tensor:
        """Fuse a list of per-type aggregated tensors (N, D)."""
        if len(z_list) == 1:
            return z_list[0]
        stacked = torch.stack(z_list, dim=1)  # (N, T, D)
        scores = self.attn(stacked).squeeze(-1)  # (N, T)
        weights = F.softmax(scores, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)


class HetGNN(BaseHGNN):
    """
    Heterogeneous Graph Neural Network.

    Per-type content encoding → per-relation neighbor aggregation
    → cross-type attention fusion → classification.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        num_classes: int,
        target_type: str,
        edge_types: List[Tuple[str, str, str]],
        dropout: float = 0.5,
    ):
        super().__init__(target_type=target_type, num_classes=num_classes)
        self.hidden_channels = hidden_channels
        self.edge_types_list = edge_types

        # Per-type content encoders
        self.content_encoders = nn.ModuleDict()
        for nt, in_c in in_channels_dict.items():
            self.content_encoders[nt] = TypeContentEncoder(in_c, hidden_channels)

        # Per-relation neighbor aggregators (only for edges targeting target type)
        self.neighbor_aggs = nn.ModuleDict()
        target_neighbor_types = set()
        for src_t, rel, dst_t in edge_types:
            if dst_t == target_type:
                key = f"{src_t}__{rel}__{dst_t}"
                self.neighbor_aggs[key] = TypeNeighborAggregator(hidden_channels)
                target_neighbor_types.add(src_t)

        # Cross-type fusion
        num_views = len(self.neighbor_aggs) + 1  # +1 for self (target content)
        self.fusion = CrossTypeFusion(hidden_channels, num_views)

        # Classifier
        self.classifier = ClassifierHead(hidden_channels, num_classes, dropout)

    def forward(self, data: HeteroData) -> Tensor:
        device = next(self.parameters()).device

        # Content encoding for all types
        h_dict = {}
        for nt in data.node_types:
            if nt in self.content_encoders and hasattr(data[nt], "x"):
                h_dict[nt] = self.content_encoders[nt](data[nt].x.to(device))

        # Per-relation neighbor aggregation (for edges targeted at target_type)
        z_list = [h_dict[self.target_type]]  # Self-view

        for src_t, rel, dst_t in self.edge_types_list:
            if dst_t == self.target_type:
                key = f"{src_t}__{rel}__{dst_t}"
                if key in self.neighbor_aggs and src_t in h_dict:
                    ei = data[(src_t, rel, dst_t)].edge_index.to(device)
                    z = self.neighbor_aggs[key](h_dict[src_t], h_dict[dst_t], ei)
                    z_list.append(z)

        # Cross-type fusion
        z_fused = self.fusion(z_list)

        return self.classifier(z_fused)

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        target_type: str,
        num_classes: int,
        hidden_channels: int = 128,
        dropout: float = 0.5,
        **kwargs,
    ) -> "HetGNN":
        in_channels_dict = get_in_channels_dict(data)
        # Replace 0-dim with embedding-like dimension
        for nt in list(in_channels_dict.keys()):
            if in_channels_dict[nt] == 0:
                in_channels_dict[nt] = hidden_channels

        return cls(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            target_type=target_type,
            edge_types=list(data.edge_types),
            dropout=dropout,
        )
