"""
Utilities for meta-path-based models.

Provides:
- Metapath adjacency matrix construction from HeteroData
- Sparse neighbor feature precomputation (for SeHGNN)
- Metapath validation helpers
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

EdgeType = Tuple[str, str, str]


def _resolve_edge_type(
    data: HeteroData,
    step_src: str,
    step_rel: str,
    step_dst: str,
) -> Tuple[str, str, str]:
    """
    Resolve an edge type triplet against the data's actual edge types.

    If the exact (src, rel, dst) exists, return it.
    Otherwise, find the first edge type matching (src, *, dst).
    """
    # Exact match
    if (step_src, step_rel, step_dst) in data.edge_types:
        return (step_src, step_rel, step_dst)

    # Fuzzy match by (src, dst)
    for s, r, d in data.edge_types:
        if s == step_src and d == step_dst:
            return (s, r, d)

    raise KeyError(
        f"No edge type found for ({step_src}, *, {step_dst}). "
        f"Available: {data.edge_types}"
    )


def build_metapath_adjacency(
    data: HeteroData,
    metapath: List[EdgeType],
    symmetric: bool = True,
    return_stats: bool = False,
) -> Tensor | tuple[Tensor, dict]:
    """
    Compose a meta-path induced adjacency as a sparse (N_target, N_target) matrix.

    Multiplies per-hop sparse adjacency matrices along the meta-path and
    returns a boolean connectivity matrix (self-loops removed).

    Edge types are auto-resolved: if the exact (src, rel, dst) doesn't exist
    in the data, the first edge matching (src, *, dst) is used.

    Parameters
    ----------
    data : HeteroData
    metapath : list of (src, rel, dst)
    symmetric : bool
        If True, make the result undirected.

    Returns
    -------
    edge_index : Tensor (2, E)
        Meta-path induced edges between target nodes.
    """
    src_type = metapath[0][0]
    dst_type = metapath[-1][2]
    assert src_type == dst_type, f"Metapath must start and end at same type, got {src_type}→{dst_type}"

    t_total = time.perf_counter()
    relation_build_s = 0.0
    sparse_mm_s = 0.0
    postprocess_s = 0.0
    sparse_mm_calls = 0

    # Build per-hop sparse adjacency
    adj = None
    for step_src, step_rel, step_dst in metapath:
        t_rel = time.perf_counter()
        resolved = _resolve_edge_type(data, step_src, step_rel, step_dst)
        ei = data[resolved].edge_index.cpu()
        n_src = data[step_src].num_nodes
        n_dst = data[step_dst].num_nodes

        vals = torch.ones(ei.size(1), dtype=torch.float)
        step_adj = torch.sparse_coo_tensor(
            ei, vals, (n_src, n_dst),
        ).coalesce()
        relation_build_s += time.perf_counter() - t_rel

        if adj is None:
            adj = step_adj
        else:
            t_mm = time.perf_counter()
            adj = torch.sparse.mm(adj, step_adj)
            sparse_mm_s += time.perf_counter() - t_mm
            sparse_mm_calls += 1

    # Convert to boolean edge_index (remove self-loops)
    t_post = time.perf_counter()
    adj = adj.coalesce()
    idx = adj.indices()
    mask = idx[0] != idx[1]  # remove self-loops
    edge_index = idx[:, mask]

    if symmetric and edge_index.size(1) > 0:
        # Add reverse edges and dedup
        rev = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, rev], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
    postprocess_s += time.perf_counter() - t_post

    if not return_stats:
        return edge_index

    stats = {
        "metapath": list(metapath),
        "num_steps": len(metapath),
        "sparse_mm_calls": int(sparse_mm_calls),
        "relation_build_s": float(relation_build_s),
        "sparse_mm_s": float(sparse_mm_s),
        "postprocess_s": float(postprocess_s),
        "non_sparse_overhead_s": float(relation_build_s + postprocess_s),
        "total_s": float(time.perf_counter() - t_total),
    }
    return edge_index, stats


def build_all_metapath_adjacencies(
    data: HeteroData,
    metapaths: List[List[EdgeType]],
    symmetric: bool = True,
) -> List[Tensor]:
    """Build edge_index for each metapath in the list."""
    return [build_metapath_adjacency(data, mp, symmetric) for mp in metapaths]


def precompute_metapath_features(
    data: HeteroData,
    metapaths: List[List[EdgeType]],
    target_type: str,
    num_hops: int = 1,
    return_stats: bool = False,
) -> Dict[str, Tensor] | tuple[Dict[str, Tensor], dict]:
    """
    Precompute metapath neighbor aggregated features (for SeHGNN).

    For each metapath, average the target features of metapath-reachable nodes.

    Returns
    -------
    Dict mapping metapath description → aggregated feature tensor (N_target, D).
    """
    x_target = data[target_type].x
    N = data[target_type].num_nodes
    D = x_target.size(1)
    result = {}
    stats = {
        "per_metapath": {},
        "feature_sparse_mm_s": 0.0,
        "feature_non_sparse_overhead_s": 0.0,
        "feature_sparse_mm_calls": 0,
    }

    for mp in metapaths:
        mp_desc = "→".join(e[0][0].upper() for e in [mp]) + "→".join(
            step[2][0].upper() for step in mp
        )
        built = build_metapath_adjacency(data, mp, symmetric=False, return_stats=return_stats)
        if return_stats:
            ei, mp_stats = built
        else:
            ei = built
            mp_stats = None
        if ei.size(1) == 0:
            result[mp_desc] = torch.zeros(N, D)
            if return_stats:
                stats["per_metapath"][mp_desc] = {
                    **mp_stats,
                    "feature_sparse_mm_s": 0.0,
                    "feature_non_sparse_overhead_s": 0.0,
                    "feature_sparse_mm_calls": 0,
                }
            continue

        # For each target node, average features of metapath neighbors
        t_non_sparse = time.perf_counter()
        vals = torch.ones(ei.size(1), dtype=torch.float)
        adj = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce()

        # Row-normalize
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        idx = adj.indices()
        new_vals = adj.values() / deg[idx[0]]
        adj_norm = torch.sparse_coo_tensor(idx, new_vals, (N, N)).coalesce()
        feature_non_sparse = time.perf_counter() - t_non_sparse

        t_mm = time.perf_counter()
        agg = torch.sparse.mm(adj_norm, x_target)
        feature_sparse_mm = time.perf_counter() - t_mm
        result[mp_desc] = agg
        if return_stats:
            stats["feature_sparse_mm_s"] += float(feature_sparse_mm)
            stats["feature_non_sparse_overhead_s"] += float(feature_non_sparse)
            stats["feature_sparse_mm_calls"] += 1
            stats["per_metapath"][mp_desc] = {
                **mp_stats,
                "feature_sparse_mm_s": float(feature_sparse_mm),
                "feature_non_sparse_overhead_s": float(feature_non_sparse),
                "feature_sparse_mm_calls": 1,
            }

    if not return_stats:
        return result
    return result, stats
