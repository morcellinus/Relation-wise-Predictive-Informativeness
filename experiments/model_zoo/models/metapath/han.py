"""
HAN — Heterogeneous Graph Attention Network
=============================================

Source: PyG built-in ``HANConv`` (Wang et al., WWW 2019)

Architecture:
- Per-metapath semantic view with node-level GAT attention
- Cross-metapath semantic-level attention fusion
- Final classification head

This wrapper stacks multiple ``HANConv`` layers and adds a classification head.
HANConv internally handles metapath construction from the HeteroData metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv

from ..base import BaseHGNN
from ..common.layers import ClassifierHead, get_in_channels_dict


class HAN(BaseHGNN):
    """
    Heterogeneous Graph Attention Network using PyG's HANConv.

    Parameters
    ----------
    metadata : tuple
        ``(node_types, edge_types)`` from the HeteroData.
    in_channels : int or dict
        Input feature dimensions per node type.
    hidden_channels : int
    num_classes : int
    target_type : str
    num_layers : int
    heads : int
    dropout : float
    """

    def __init__(
        self,
        metadata: Tuple,
        in_channels,
        hidden_channels: int,
        num_classes: int,
        target_type: str,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__(target_type=target_type, num_classes=num_classes)
        self.metadata = metadata
        self.active_node_types = list(metadata[0])
        self.active_edge_types = list(metadata[1])
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer
        self.convs.append(
            HANConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
                dropout=dropout,
            )
        )

        # Hidden layers
        for _ in range(1, num_layers):
            self.convs.append(
                HANConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                    dropout=dropout,
                )
            )

        self.classifier = ClassifierHead(hidden_channels, num_classes, dropout)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            nt: data[nt].x
            for nt in self.active_node_types
            if nt in data.node_types and hasattr(data[nt], "x") and data[nt].x is not None
        }
        edge_index_dict = {
            et: data[et].edge_index
            for et in self.active_edge_types
            if et in data.edge_types
        }

        for i, conv in enumerate(self.convs):
            raw_out = conv(x_dict, edge_index_dict)
            x_dict = {k: v for k, v in raw_out.items() if v is not None}
            if i < self.num_layers - 1:
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        if self.target_type not in x_dict:
            raise RuntimeError(
                f"HAN target type '{self.target_type}' became unreachable under current graph/metapath structure."
            )
        return self.classifier(x_dict[self.target_type])

    @classmethod
    def from_data(
        cls,
        data: HeteroData,
        target_type: str,
        num_classes: int,
        metapaths: Optional[List[List[Tuple[str, str, str]]]] = None,
        hidden_channels: int = 128,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        **kwargs,
    ) -> "HAN":
        if metapaths:
            edge_types = []
            seen = set()
            node_types = []
            node_seen = set()
            for mp in metapaths:
                for et in mp:
                    et = tuple(et)
                    if et not in seen:
                        seen.add(et)
                        edge_types.append(et)
                    for nt in (et[0], et[2]):
                        if nt not in node_seen:
                            node_seen.add(nt)
                            node_types.append(nt)
            if target_type not in node_seen:
                node_types.append(target_type)
            metadata = (node_types, edge_types)
        else:
            metadata = data.metadata()

        in_channels_all = get_in_channels_dict(data)
        in_channels = {
            k: v for k, v in in_channels_all.items()
            if v > 0 and k in metadata[0]
        }

        return cls(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            target_type=target_type,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
