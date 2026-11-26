from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch


def stable_int_seed(*parts: object, modulo: int = 1_000_000_000) -> int:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:16], 16) % modulo


def edge_type_name(edge_type: tuple[str, str, str]) -> str:
    return "-".join(edge_type)


def relation_distance_to_target(
    data, target_type: str
) -> dict[tuple[str, str, str], int | None]:
    reverse = {node_type: set() for node_type in data.node_types}
    for source, _, destination in data.edge_types:
        reverse[destination].add(source)
    distance = {target_type: 0}
    frontier = [target_type]
    while frontier:
        destination = frontier.pop(0)
        for source in reverse[destination]:
            if source not in distance:
                distance[source] = distance[destination] + 1
                frontier.append(source)
    return {
        tuple(edge_type): distance.get(edge_type[2]) for edge_type in data.edge_types
    }


def _edge_keys(edge_index: torch.Tensor, num_dst: int) -> torch.Tensor:
    return edge_index[0].long().cpu() * int(num_dst) + edge_index[1].long().cpu()


def _is_exact_reverse(data, forward, reverse) -> bool:
    left = data[forward].edge_index.cpu()
    right = data[reverse].edge_index.cpu()
    if int(left.size(1)) != int(right.size(1)):
        return False
    left_keys = torch.sort(_edge_keys(left, int(data[forward[2]].num_nodes))).values
    right_keys = torch.sort(
        right[1].long().cpu() * int(data[forward[2]].num_nodes)
        + right[0].long().cpu()
    ).values
    return bool(torch.equal(left_keys, right_keys))


def infer_exact_reverse_groups(data) -> list[dict]:
    edge_types = sorted(tuple(value) for value in data.edge_types)
    used: set[tuple[str, str, str]] = set()
    groups = []
    for edge_type in edge_types:
        if edge_type in used:
            continue
        source, _, destination = edge_type
        candidates = [
            candidate
            for candidate in edge_types
            if candidate not in used
            and candidate != edge_type
            and candidate[0] == destination
            and candidate[2] == source
            and _is_exact_reverse(data, edge_type, candidate)
        ]
        if not candidates:
            continue
        pair = sorted([edge_type, sorted(candidates)[0]])
        groups.append(
            {
                "name": " || ".join(edge_type_name(value) for value in pair),
                "forward": pair[0],
                "reverse": pair[1],
            }
        )
        used.update(pair)
    return sorted(groups, key=lambda group: group["name"])


def _paired_local_indices(data, forward, reverse) -> tuple[np.ndarray, np.ndarray]:
    left = data[forward].edge_index.cpu()
    right = data[reverse].edge_index.cpu()
    left_keys = _edge_keys(left, int(data[forward[2]].num_nodes))
    right_keys = (
        right[1].long().cpu() * int(data[forward[2]].num_nodes)
        + right[0].long().cpu()
    )
    left_order = torch.argsort(left_keys, stable=True)
    right_order = torch.argsort(right_keys, stable=True)
    if not torch.equal(left_keys[left_order], right_keys[right_order]):
        raise RuntimeError(f"Reverse-pair mismatch: {forward} vs {reverse}")
    return left_order.numpy(), right_order.numpy()


def eligible_scored_groups(
    data,
    target_type: str,
    rpi_by_relation: dict[str, tuple[float, int]],
    num_layers: int,
) -> list[dict]:
    distances = relation_distance_to_target(data, target_type)
    max_distance = max(0, int(num_layers) - 1)
    output = []
    for group in infer_exact_reverse_groups(data):
        edge_types = [group["forward"], group["reverse"]]
        if not any(
            distances.get(edge_type) is not None
            and int(distances[edge_type]) <= max_distance
            for edge_type in edge_types
        ):
            continue
        weighted = []
        for edge_type in edge_types:
            score, edge_count = rpi_by_relation.get(
                edge_type_name(edge_type), (float("nan"), 0)
            )
            if math.isfinite(float(score)) and int(edge_count) > 0:
                weighted.append((float(score), int(edge_count)))
        if not weighted:
            continue
        denominator = float(sum(weight for _, weight in weighted))
        forward_local, reverse_local = _paired_local_indices(
            data, group["forward"], group["reverse"]
        )
        output.append(
            {
                **group,
                "rpi": sum(value * weight for value, weight in weighted)
                / denominator,
                "capacity_units": int(len(forward_local)),
                "forward_local": forward_local,
                "reverse_local": reverse_local,
            }
        )
    return sorted(output, key=lambda group: group["name"])


def build_pair_order(
    groups: list[dict], arm: str, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    group_indices = list(range(len(groups)))
    if arm == "rpi_low_first":
        group_indices.sort(
            key=lambda index: (groups[index]["rpi"], groups[index]["name"])
        )
    elif arm != "global_random_edges":
        raise ValueError(f"Unknown pruning arm: {arm}")
    group_parts = []
    unit_parts = []
    if arm == "global_random_edges":
        for group_index, group in enumerate(groups):
            capacity = int(group["capacity_units"])
            group_parts.append(np.full(capacity, group_index, dtype=np.int32))
            unit_parts.append(np.arange(capacity, dtype=np.int64))
        ordered_groups = np.concatenate(group_parts)
        ordered_units = np.concatenate(unit_parts)
        permutation = rng.permutation(len(ordered_units))
        return (
            ordered_groups[permutation],
            ordered_units[permutation],
            ["GLOBAL_RANDOM"],
        )
    for group_index in group_indices:
        capacity = int(groups[group_index]["capacity_units"])
        group_parts.append(np.full(capacity, group_index, dtype=np.int32))
        unit_parts.append(rng.permutation(capacity).astype(np.int64))
    return (
        np.concatenate(group_parts),
        np.concatenate(unit_parts),
        [groups[index]["name"] for index in group_indices],
    )


def _mask_edge_store(store, keep: torch.Tensor, original_edges: int) -> None:
    store.edge_index = store.edge_index.cpu()[:, keep]
    for key, value in list(store.items()):
        if key == "edge_index" or not torch.is_tensor(value) or value.ndim == 0:
            continue
        if int(value.size(0)) == int(original_edges):
            store[key] = value.cpu()[keep]


def apply_pair_budget(
    data,
    groups: list[dict],
    ordered_groups: np.ndarray,
    ordered_units: np.ndarray,
    budget_units: int,
) -> tuple[object, dict[str, int]]:
    graph = copy.deepcopy(data)
    chosen_groups = ordered_groups[:budget_units]
    chosen_units = ordered_units[:budget_units]
    removed = {}
    for group_index, group in enumerate(groups):
        unit_ids = chosen_units[chosen_groups == group_index]
        forward_remove = group["forward_local"][unit_ids]
        reverse_remove = group["reverse_local"][unit_ids]
        for edge_type, local_indices in [
            (group["forward"], forward_remove),
            (group["reverse"], reverse_remove),
        ]:
            original_edges = int(graph[edge_type].edge_index.size(1))
            keep = torch.ones(original_edges, dtype=torch.bool)
            if len(local_indices):
                keep[torch.from_numpy(local_indices).long()] = False
            _mask_edge_store(graph[edge_type], keep, original_edges)
        removed[group["name"]] = int(len(unit_ids))
    if sum(removed.values()) != int(budget_units):
        raise RuntimeError("Exact reverse-paired budget was not satisfied")
    return graph, removed
