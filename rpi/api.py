from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData

from . import _core


@dataclass
class RPIResult:
    """Scores by directed relation, by destination type, and for the target type."""

    relations: list[dict]
    node_types: list[dict]
    target: dict | None


def compute_rpi(
    graph: HeteroData,
    target_type: str,
    *,
    train_mask: torch.Tensor,
    alpha: float = 0.9,
    steps: int = 50,
) -> RPIResult:
    """Compute operational RPI on a PyG heterogeneous graph on CPU.

    Supply a one-dimensional integer label tensor as `graph[target_type].y`
    and a one-dimensional boolean training mask. Class IDs can be noncontiguous.
    Only labels selected by the mask are read for propagation. The graph is
    unchanged. Relations with zero destination entropy have a NaN score.
    """
    if not isinstance(graph, HeteroData):
        raise TypeError("graph must be a PyG HeteroData")
    if target_type not in graph.node_types:
        raise ValueError(f"Unknown target node type: {target_type!r}")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    node_counts = {}
    for node_type in graph.node_types:
        count = graph[node_type].num_nodes
        if count is None or count < 0:
            raise ValueError(f"Set num_nodes for node type {node_type!r}")
        node_counts[node_type] = count

    target_shape = (node_counts[target_type],)
    labels = getattr(graph[target_type], "y", None)
    if (
        not isinstance(labels, torch.Tensor)
        or labels.shape != target_shape
        or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    ):
        raise ValueError("Target y must be a one-dimensional integer tensor with one label per node")
    if (
        not isinstance(train_mask, torch.Tensor)
        or train_mask.shape != target_shape
        or train_mask.dtype != torch.bool
    ):
        raise ValueError("train_mask must be a one-dimensional boolean tensor with one entry per target node")

    mask = train_mask.cpu()
    training_labels = labels.cpu()[mask]
    if training_labels.numel() == 0 or (training_labels < 0).any():
        raise ValueError("train_mask must select at least one node and only nodes with nonnegative labels")

    relations = sorted(graph.edge_types)
    for relation in relations:
        src, _, dst = relation
        if src not in node_counts or dst not in node_counts:
            raise ValueError(f"Define both endpoint node types for relation {relation!r}")
        edge_index = getattr(graph[relation], "edge_index", None)
        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.dtype != torch.long
            or edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(f"Relation {relation!r} needs a long edge_index tensor of shape [2, E]")
        if edge_index.numel() and (
            (edge_index < 0).any()
            or edge_index[0].max() >= node_counts[src]
            or edge_index[1].max() >= node_counts[dst]
        ):
            raise ValueError(f"Relation {relation!r} has node indices outside its endpoint types")

    # Encode training classes densely and leave every held-out label unknown.
    _, encoded = torch.unique(training_labels, sorted=True, return_inverse=True)
    seed_labels = torch.full(target_shape, -1, dtype=torch.long)
    seed_labels[mask] = encoded
    payload = _core.propagate_profiles(
        graph, target_type, relations, alpha, steps, y=seed_labels, train_mask=mask
    )
    relation_rows = _core.score_relations(graph, payload["hard_labels"], relations)
    node_type_rows = _core.aggregate_by_destination(relation_rows)
    target_row = next((row for row in node_type_rows if row["node_type"] == target_type), None)
    return RPIResult(relation_rows, node_type_rows, target_row)
