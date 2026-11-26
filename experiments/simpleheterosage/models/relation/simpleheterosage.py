from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from simpleheterosage.models.base import BaseHGNN
from simpleheterosage.models.common.layers import get_in_channels_dict


class SimpleHeteroSAGE(BaseHGNN):
    def __init__(self, in_channels_dict, hidden_channels, num_classes, target_type, edge_types, num_layers=2, dropout=0.5):
        super().__init__(target_type=target_type, num_classes=num_classes)
        self.dropout = dropout
        self.node_types = list(in_channels_dict)
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_ch, hidden_channels) for nt, in_ch in in_channels_dict.items()
        })
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs = {et: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for et in edge_types}
            self.convs.append(HeteroConv(convs, aggr="sum"))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x_dict = {nt: F.relu(self.input_proj[nt](data[nt].x)) for nt in self.node_types}
        full_edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        for conv in self.convs:
            edge_index_dict = {
                et: ei for et, ei in full_edge_index_dict.items()
                if et[0] in x_dict and et[2] in x_dict
            }
            if not edge_index_dict:
                break
            raw_out = conv(x_dict, edge_index_dict)
            x_dict = {
                k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                for k, v in raw_out.items()
                if v is not None
            }
        if self.target_type not in x_dict:
            raise RuntimeError(
                f"SimpleHeteroSAGE target type '{self.target_type}' became unreachable under current graph structure."
            )
        return self.classifier(x_dict[self.target_type])

    @classmethod
    def from_data(cls, data, target_type: str, num_classes: int, **config):
        return cls(
            get_in_channels_dict(data),
            hidden_channels=config.get("hidden_channels", 128),
            num_classes=num_classes,
            target_type=target_type,
            edge_types=list(data.edge_types),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.5),
        )
