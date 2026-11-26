"""
Utilities for converting HeteroData to homogeneous graphs with type annotations.

Used by relation-based models (R-GCN, SimpleHGN) that operate on a single
adjacency matrix with per-edge relation type indices.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def hetero_to_homogeneous(
    data: HeteroData,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, int], Dict[Tuple, int], int]:
    """
    Convert a HeteroData to a single homogeneous graph with type indices.

    Returns
    -------
    edge_index : Tensor (2, E_total)
        All edges with global node indices.
    node_type_ids : Tensor (N_total,)
        Integer type ID per node.
    edge_type_ids : Tensor (E_total,)
        Integer relation ID per edge.
    node_type_offset : dict
        ``{node_type: global_start_index}``.
    edge_type_map : dict
        ``{(src,rel,dst): relation_id}``.
    num_nodes_total : int
    """
    # Assign global offsets per node type
    node_types = data.node_types
    node_type_offset: Dict[str, int] = {}
    offset = 0
    for nt in node_types:
        node_type_offset[nt] = offset
        offset += data[nt].num_nodes
    num_nodes_total = offset

    # Node type IDs
    node_type_ids_list = []
    for i, nt in enumerate(node_types):
        node_type_ids_list.append(
            torch.full((data[nt].num_nodes,), i, dtype=torch.long)
        )
    node_type_ids = torch.cat(node_type_ids_list)

    # Edges
    edge_type_map: Dict[Tuple, int] = {}
    all_src, all_dst, all_etype = [], [], []
    for et_idx, (src_t, rel, dst_t) in enumerate(data.edge_types):
        edge_type_map[(src_t, rel, dst_t)] = et_idx
        ei = data[(src_t, rel, dst_t)].edge_index
        if ei is None or ei.numel() == 0 or ei.size(1) == 0:
            continue
        all_src.append(ei[0] + node_type_offset[src_t])
        all_dst.append(ei[1] + node_type_offset[dst_t])
        all_etype.append(torch.full((ei.size(1),), et_idx, dtype=torch.long))

    if not all_src:
        raise RuntimeError("Homogeneous conversion received a graph with no edges.")

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
    edge_type_ids = torch.cat(all_etype)

    return (
        edge_index,
        node_type_ids,
        edge_type_ids,
        node_type_offset,
        edge_type_map,
        num_nodes_total,
    )


def get_homogeneous_features(
    data: HeteroData,
    node_type_offset: Dict[str, int],
    num_nodes_total: int,
    hidden_dim: int,
) -> Tensor:
    """
    Stack per-type features into a single (N_total, D) tensor.

    Types with different feature dims are zero-padded to ``hidden_dim``.
    Featureless types get zeros.
    """
    # Detect device from features
    device = torch.device("cpu")
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            device = data[nt].x.device
            break

    x = torch.zeros(num_nodes_total, hidden_dim, device=device)
    for nt in data.node_types:
        off = node_type_offset[nt]
        n = data[nt].num_nodes
        if hasattr(data[nt], "x") and data[nt].x is not None:
            feat = data[nt].x
            d = min(feat.size(1), hidden_dim)
            x[off: off + n, :d] = feat[:, :d]
    return x
