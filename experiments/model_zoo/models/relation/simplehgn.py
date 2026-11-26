"""
SimpleHGN — Simple Heterogeneous Graph Neural Network
======================================================

Source: OpenHGNN (Lv et al., KDD 2021)
Ported from DGL to PyG MessagePassing.

Architecture:
- Edge-type-aware GAT attention: α_ij = softmax(a^T [Wh_i || Wh_j || W_r r_e])
- Learnable edge-type embeddings
- Node residual: h_i^l = σ(Σ α_ij W h_j + h_i^(l-1))
- Edge residual: α^l = (1-β)α^l + β·α^(l-1)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from ..base import BaseHGNN
from ..common.layers import ClassifierHead, get_in_channels_dict
from ..common.relation_utils import (
    get_homogeneous_features,
    hetero_to_homogeneous,
)


class SimpleHGNConv(MessagePassing):
    """
    Single SimpleHGN convolution layer (PyG port of OpenHGNN DGL implementation).

    Computes edge-type-aware attention with 3-way scoring:
    source features, target features, and edge-type embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        num_edge_types: int,
        edge_dim: int = 64,
        feat_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
        beta: float = 0.05,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.beta = beta

        self.W = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, num_heads, out_channels))
        self.a_dst = nn.Parameter(torch.empty(1, num_heads, out_channels))
        self.a_edge = nn.Parameter(torch.empty(1, num_heads, edge_dim))

        self.edge_emb = nn.Embedding(num_edge_types, edge_dim)
        self.edge_proj = nn.Linear(edge_dim, edge_dim * num_heads, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.feat_drop = nn.Dropout(feat_drop)

        if residual:
            self.residual_proj = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        else:
            self.residual_proj = None

        self._alpha = None  # Store for edge residual
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.a_edge)
        nn.init.xavier_uniform_(self.edge_emb.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        last_alpha: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        H, C = self.num_heads, self.out_channels

        x_drop = self.feat_drop(x)
        x_proj = self.W(x_drop).view(-1, H, C)  # (N, H, C)

        # Source and target attention scores
        alpha_src = (x_proj * self.a_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_proj * self.a_dst).sum(dim=-1)  # (N, H)

        # Edge type embeddings
        e_emb = self.edge_emb(edge_type)  # (E, edge_dim)
        e_proj = self.edge_proj(e_emb).view(-1, H, self.edge_dim)
        alpha_edge = (e_proj * self.a_edge).sum(dim=-1)  # (E, H)

        # Store for message passing
        self._x_proj = x_proj
        self._alpha_src = alpha_src
        self._alpha_dst = alpha_dst
        self._alpha_edge = alpha_edge
        self._last_alpha = last_alpha

        out = self.propagate(edge_index, size=None)  # (N, H, C)

        # Flatten heads
        out = out.view(-1, H * C)

        # Node residual
        if self.residual_proj is not None:
            out = out + self.residual_proj(x_drop)

        alpha = self._alpha  # cached from message()
        return out, alpha

    def message(self, x_proj_j: Tensor, index: Tensor, size_i: int) -> Tensor:
        # This method is called by propagate but we need a custom approach
        # because we need edge-level alpha. Use message_and_aggregate instead.
        raise NotImplementedError("Use forward directly")

    def propagate(self, edge_index, size=None, **kwargs):
        """Custom propagation with edge-type attention."""
        H, C = self.num_heads, self.out_channels
        src, dst = edge_index

        # Attention coefficients
        alpha = (
            self._alpha_src[src] +  # (E, H)
            self._alpha_dst[dst] +  # (E, H)
            self._alpha_edge        # (E, H)
        )
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, dst)

        # Edge residual
        if self._last_alpha is not None and self.beta > 0:
            alpha = (1 - self.beta) * alpha + self.beta * self._last_alpha
        self._alpha = alpha

        # Message: weighted source features
        msg = self._x_proj[src] * alpha.unsqueeze(-1)  # (E, H, C)

        # Aggregate
        num_nodes = self._x_proj.size(0)
        out = torch.zeros(num_nodes, H, C, device=msg.device)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, H, C)
        out.scatter_add_(0, dst_expanded, msg)

        return out


class SimpleHGN(BaseHGNN):
    """
    Simple Heterogeneous Graph Neural Network.

    Stacks ``SimpleHGNConv`` layers on a homogenized graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_edge_types: int,
        target_type: str,
        target_offset: int,
        target_num_nodes: int,
        num_nodes_total: int,
        num_layers: int = 3,
        num_heads: int = 4,
        edge_dim: int = 64,
        feat_drop: float = 0.5,
        beta: float = 0.05,
    ):
        super().__init__(target_type=target_type, num_classes=num_classes)
        self.in_channels = in_channels
        self.target_offset = target_offset
        self.target_num_nodes = target_num_nodes
        self.num_nodes_total = num_nodes_total
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(
            SimpleHGNConv(
                in_channels, hidden_channels, num_heads, num_edge_types,
                edge_dim=edge_dim, feat_drop=feat_drop, residual=False, beta=beta,
            )
        )
        # Hidden layers
        for _ in range(1, num_layers - 1):
            self.layers.append(
                SimpleHGNConv(
                    hidden_channels * num_heads, hidden_channels, num_heads,
                    num_edge_types, edge_dim=edge_dim, feat_drop=feat_drop,
                    residual=True, beta=beta,
                )
            )
        # Output layer (single head)
        self.layers.append(
            SimpleHGNConv(
                hidden_channels * num_heads, num_classes, 1, num_edge_types,
                edge_dim=edge_dim, feat_drop=feat_drop, residual=True, beta=beta,
            )
        )

        self._cached_edge_index = None
        self._cached_edge_type = None

    def forward(self, data: HeteroData) -> Tensor:
        device = next(self.parameters()).device

        # Homogeneous conversion (cached)
        if self._cached_edge_index is None:
            ei, _, et, _, _, _ = hetero_to_homogeneous(data)
            self._cached_edge_index = ei.to(device)
            self._cached_edge_type = et.to(device)

        # Build feature matrix
        offset = 0
        offsets = {}
        for nt in data.node_types:
            offsets[nt] = offset
            offset += data[nt].num_nodes

        x = get_homogeneous_features(data, offsets, offset, self.in_channels).to(device)
        edge_index = self._cached_edge_index
        edge_type = self._cached_edge_type

        last_alpha = None
        for i, layer in enumerate(self.layers):
            # Reset edge residual if head count changes (e.g. hidden→output)
            if last_alpha is not None and last_alpha.size(1) != layer.num_heads:
                last_alpha = None
            x, last_alpha = layer(x, edge_index, edge_type, last_alpha)
            if i < self.num_layers - 1:
                x = F.elu(x)

        target_x = x[self.target_offset: self.target_offset + self.target_num_nodes]
        return target_x

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        target_type: str,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        edge_dim: int = 64,
        feat_drop: float = 0.5,
        beta: float = 0.05,
        **kwargs,
    ) -> "SimpleHGN":
        num_edge_types = len(data.edge_types)
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
            num_edge_types=num_edge_types,
            target_type=target_type,
            target_offset=target_offset,
            target_num_nodes=data[target_type].num_nodes,
            num_nodes_total=offset,
            num_layers=num_layers,
            num_heads=num_heads,
            edge_dim=edge_dim,
            feat_drop=feat_drop,
            beta=beta,
        )
