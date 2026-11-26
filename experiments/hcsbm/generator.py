from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .config import HCSBMConfig, NodeTypeConfig
from .math_utils import (
    choose_feature_dim,
    effective_relation_matrix,
    empirical_label_joint,
    empirical_mutual_information,
    make_reverse_edge_type,
    mutual_information_from_relation,
    normalize_probability_vector,
    relation_name_to_edge_type,
    reverse_relation_name,
    reverse_relation_matrix_from_joint,
    theoretical_joint_from_relation,
)


def _torch_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return gen


def _generate_prototypes(
    node_cfg: NodeTypeConfig,
    common_dim: int,
    seed: int,
) -> torch.Tensor:
    dim = choose_feature_dim(common_dim, node_cfg.feature_dim)
    num_classes = node_cfg.num_classes
    gen = _torch_generator(seed)
    mode = node_cfg.prototype_mode.lower()

    if mode == "orthonormal" and dim >= num_classes:
        base = torch.randn(dim, num_classes, generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(base, mode="reduced")
        return q.transpose(0, 1).contiguous()

    if mode in {"orthonormal", "simplex", "normalized_gaussian"}:
        base = torch.randn(num_classes, dim, generator=gen, dtype=torch.float32)
        base = torch.nn.functional.normalize(base, p=2, dim=1)
        return base

    raise ValueError(f"Unsupported prototype_mode {node_cfg.prototype_mode!r}")


def _sample_labels(prior: np.ndarray, num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    num_classes = prior.size
    if num_nodes < num_classes:
        return rng.choice(num_classes, size=num_nodes, replace=True, p=prior).astype(np.int64)

    counts = np.ones(num_classes, dtype=np.int64)
    remaining = num_nodes - num_classes
    if remaining > 0:
        counts += rng.multinomial(remaining, prior)
    labels = np.repeat(np.arange(num_classes, dtype=np.int64), counts)
    rng.shuffle(labels)
    return labels


def _build_features(
    labels: np.ndarray,
    prototypes: torch.Tensor,
    gamma: float,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    gen = _torch_generator(seed)
    num_nodes = labels.shape[0]
    dim = prototypes.size(1)
    proto = prototypes[torch.from_numpy(labels)]
    noise = torch.randn(num_nodes, dim, generator=gen, dtype=torch.float32) * float(sigma)
    return float(gamma) * proto + noise


def _indices_by_class(labels: np.ndarray, num_classes: int) -> List[np.ndarray]:
    return [np.where(labels == cls)[0] for cls in range(num_classes)]


def _validate_relation_feasibility(
    relation_name: str,
    source_type: str,
    destination_type: str,
    out_degree: int,
    allow_multi_edges: bool,
    destination_class_members: Sequence[np.ndarray],
) -> None:
    class_sizes = np.array([members.size for members in destination_class_members], dtype=np.int64)
    if np.any(class_sizes == 0):
        empty = np.where(class_sizes == 0)[0].tolist()
        raise ValueError(
            f"Relation {relation_name!r} cannot sample destination labels because "
            f"{destination_type!r} has empty classes {empty}"
        )
    if not allow_multi_edges and out_degree > int(class_sizes.sum()):
        raise ValueError(
            f"Relation {relation_name!r} requests out_degree={out_degree}, but destination type "
            f"{destination_type!r} only has {int(class_sizes.sum())} nodes"
        )


def _sample_without_multiedges(
    source_indices: np.ndarray,
    source_labels: np.ndarray,
    relation_matrix: np.ndarray,
    destination_class_members: Sequence[np.ndarray],
    out_degree: int,
    rng: np.random.Generator,
    relation_name: str,
    max_attempts: int = 256,
) -> np.ndarray:
    edges: List[np.ndarray] = []
    class_sizes = np.array([members.size for members in destination_class_members], dtype=np.int64)
    if out_degree > int(class_sizes.sum()):
        raise ValueError(
            f"Relation {relation_name!r} cannot sample {out_degree} unique destinations per source "
            f"from only {int(class_sizes.sum())} destination nodes"
        )

    for src_idx, src_label in zip(source_indices.tolist(), source_labels.tolist()):
        probs = relation_matrix[src_label]
        sampled_dest_labels: Optional[np.ndarray] = None
        for _ in range(max_attempts):
            candidate = rng.choice(probs.size, size=out_degree, replace=True, p=probs)
            counts = np.bincount(candidate, minlength=probs.size)
            if np.all(counts <= class_sizes):
                sampled_dest_labels = candidate
                break
        if sampled_dest_labels is None:
            raise RuntimeError(
                f"Failed to sample unique destinations for relation {relation_name!r} "
                f"after {max_attempts} attempts; try increasing destination class sizes "
                f"or enabling multi-edges"
            )
        destinations = []
        for dst_label, count in enumerate(np.bincount(sampled_dest_labels, minlength=probs.size)):
            if count == 0:
                continue
            picked = rng.choice(destination_class_members[dst_label], size=int(count), replace=False)
            destinations.append(picked.astype(np.int64))
        sampled_destinations = np.concatenate(destinations, axis=0)
        rng.shuffle(sampled_destinations)
        src_column = np.full(sampled_destinations.shape[0], src_idx, dtype=np.int64)
        edges.append(np.stack([src_column, sampled_destinations], axis=0))
    return np.concatenate(edges, axis=1) if edges else np.empty((2, 0), dtype=np.int64)


def _sample_with_multiedges(
    source_indices: np.ndarray,
    source_labels: np.ndarray,
    relation_matrix: np.ndarray,
    destination_class_members: Sequence[np.ndarray],
    out_degree: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_edges = int(source_indices.size) * int(out_degree)
    repeated_sources = np.repeat(source_indices.astype(np.int64), out_degree)
    source_repeated_labels = np.repeat(source_labels.astype(np.int64), out_degree)
    sampled_dest_labels = np.empty(total_edges, dtype=np.int64)

    for cls in np.unique(source_repeated_labels):
        mask = source_repeated_labels == cls
        sampled_dest_labels[mask] = rng.choice(
            relation_matrix.shape[1],
            size=int(mask.sum()),
            replace=True,
            p=relation_matrix[int(cls)],
        )

    sampled_destinations = np.empty(total_edges, dtype=np.int64)
    for dst_label, members in enumerate(destination_class_members):
        mask = sampled_dest_labels == dst_label
        sampled_destinations[mask] = rng.choice(members, size=int(mask.sum()), replace=True)
    return np.stack([repeated_sources, sampled_destinations], axis=0)


def _build_relation_edges(
    relation_name: str,
    source_type: str,
    destination_type: str,
    source_labels: np.ndarray,
    destination_labels: np.ndarray,
    relation_matrix: np.ndarray,
    out_degree: int,
    allow_multi_edges: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    source_indices = np.arange(source_labels.shape[0], dtype=np.int64)
    destination_class_members = _indices_by_class(destination_labels, relation_matrix.shape[1])
    _validate_relation_feasibility(
        relation_name=relation_name,
        source_type=source_type,
        destination_type=destination_type,
        out_degree=out_degree,
        allow_multi_edges=allow_multi_edges,
        destination_class_members=destination_class_members,
    )
    if allow_multi_edges:
        return _sample_with_multiedges(
            source_indices=source_indices,
            source_labels=source_labels,
            relation_matrix=relation_matrix,
            destination_class_members=destination_class_members,
            out_degree=out_degree,
            rng=rng,
        )
    return _sample_without_multiedges(
        source_indices=source_indices,
        source_labels=source_labels,
        relation_matrix=relation_matrix,
        destination_class_members=destination_class_members,
        out_degree=out_degree,
        rng=rng,
        relation_name=relation_name,
    )


def _generate_split_masks(num_nodes: int, labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    rng = np.random.default_rng(seed)
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_cls = cls_idx.size
        n_train = max(1, int(n_cls * train_ratio))
        n_val = max(1, int(n_cls * val_ratio))
        n_train = min(n_train, n_cls)
        n_val = min(n_val, n_cls - n_train)
        train_idx = cls_idx[:n_train]
        val_idx = cls_idx[n_train:n_train + n_val]
        test_idx = cls_idx[n_train + n_val:]
        train[train_idx] = True
        val[val_idx] = True
        test[test_idx] = True
    return train, val, test


def generate_hcsbm(config: HCSBMConfig) -> Tuple[HeteroData, Dict[str, object]]:
    config.validate()
    rng = np.random.default_rng(config.seed)

    data = HeteroData()
    metadata: Dict[str, object] = {
        "seed": config.seed,
        "feature_dim": config.feature_dim,
        "allow_multi_edges": config.allow_multi_edges,
        "add_reverse_relations": config.add_reverse_relations,
        "edge_set_mode": config.edge_set_mode,
        "node_types": {},
        "relations": {},
        "global_summary": {},
        "destination_summary": {},
    }

    type_labels: Dict[str, np.ndarray] = {}
    type_priors: Dict[str, np.ndarray] = {}
    for type_offset, (type_name, node_cfg) in enumerate(config.node_types.items()):
        prior = normalize_probability_vector(np.asarray(node_cfg.class_prior, dtype=np.float64), name=f"{type_name}.class_prior")
        labels = _sample_labels(prior=prior, num_nodes=node_cfg.num_nodes, rng=rng)
        prototypes = _generate_prototypes(node_cfg=node_cfg, common_dim=config.feature_dim, seed=config.seed + 1009 * (type_offset + 1))
        features = _build_features(
            labels=labels,
            prototypes=prototypes,
            gamma=node_cfg.gamma,
            sigma=node_cfg.sigma,
            seed=config.seed + 4001 * (type_offset + 1),
        )
        data[type_name].x = features
        data[type_name].y = torch.from_numpy(labels).long()
        data[type_name].num_nodes = int(node_cfg.num_nodes)

        type_labels[type_name] = labels
        type_priors[type_name] = prior
        metadata["node_types"][type_name] = {
            "num_nodes": int(node_cfg.num_nodes),
            "num_classes": int(node_cfg.num_classes),
            "class_prior": prior.tolist(),
            "gamma": float(node_cfg.gamma),
            "sigma": float(node_cfg.sigma),
            "feature_dim": int(features.size(1)),
            "prototype_mode": node_cfg.prototype_mode,
            "prototype_vectors": prototypes.tolist(),
        }

    for relation_offset, (relation_name, rel_cfg) in enumerate(config.relations.items()):
        src_type = rel_cfg.source_type
        dst_type = rel_cfg.destination_type
        edge_type = relation_name_to_edge_type(
            relation_name=relation_name,
            source_type=src_type,
            destination_type=dst_type,
            relation_aliases=dict(config.relation_aliases),
        )
        matrices = effective_relation_matrix(
            source_prior=type_priors[src_type],
            destination_prior=type_priors[dst_type],
            tau=rel_cfg.tau,
            aligned_matrix=None if rel_cfg.aligned_matrix is None else np.asarray(rel_cfg.aligned_matrix, dtype=np.float64),
        )
        relation_rng = np.random.default_rng(config.seed + 7919 * (relation_offset + 1))
        edge_index_np = _build_relation_edges(
            relation_name=relation_name,
            source_type=src_type,
            destination_type=dst_type,
            source_labels=type_labels[src_type],
            destination_labels=type_labels[dst_type],
            relation_matrix=matrices["effective"],
            out_degree=rel_cfg.out_degree,
            allow_multi_edges=config.allow_multi_edges,
            rng=relation_rng,
        )
        edge_index = torch.from_numpy(edge_index_np).long()
        data[edge_type].edge_index = edge_index

        empirical_joint = empirical_label_joint(
            source_labels=type_labels[src_type][edge_index_np[0]],
            destination_labels=type_labels[dst_type][edge_index_np[1]],
            num_source_classes=config.node_types[src_type].num_classes,
            num_destination_classes=config.node_types[dst_type].num_classes,
        )
        empirical_mi = empirical_mutual_information(
            source_labels=type_labels[src_type][edge_index_np[0]],
            destination_labels=type_labels[dst_type][edge_index_np[1]],
            num_source_classes=config.node_types[src_type].num_classes,
            num_destination_classes=config.node_types[dst_type].num_classes,
        )
        theoretical_joint = theoretical_joint_from_relation(type_priors[src_type], matrices["effective"])
        theoretical_mi = mutual_information_from_relation(
            source_prior=type_priors[src_type],
            destination_prior=type_priors[dst_type],
            relation_matrix=matrices["effective"],
        )

        metadata["relations"][relation_name] = {
            "edge_type": list(edge_type),
            "source_type": src_type,
            "destination_type": dst_type,
            "out_degree": int(rel_cfg.out_degree),
            "tau": float(rel_cfg.tau),
            "source_prior": type_priors[src_type].tolist(),
            "destination_prior": type_priors[dst_type].tolist(),
            "baseline_matrix": matrices["baseline"].tolist(),
            "aligned_joint": matrices["aligned_joint"].tolist(),
            "aligned_matrix": matrices["aligned"].tolist(),
            "effective_matrix": matrices["effective"].tolist(),
            "theoretical_mutual_information_nats": float(theoretical_mi),
            "empirical_joint": empirical_joint.tolist(),
            "empirical_mutual_information_nats": float(empirical_mi),
            "num_edges": int(edge_index.size(1)),
            "shared_pair_name": relation_name,
        }

        if config.edge_set_mode == "shared_undirected":
            reverse_edge_type = make_reverse_edge_type(edge_type)
            reverse_edge_index_np = np.stack([edge_index_np[1], edge_index_np[0]], axis=0)
            data[reverse_edge_type].edge_index = torch.from_numpy(reverse_edge_index_np).long()
            reverse_empirical_joint = empirical_joint.T
            reverse_theoretical_matrix = reverse_relation_matrix_from_joint(theoretical_joint, type_priors[dst_type])
            metadata["relations"][reverse_relation_name(relation_name)] = {
                "edge_type": list(reverse_edge_type),
                "source_type": dst_type,
                "destination_type": src_type,
                "out_degree": int(rel_cfg.out_degree),
                "tau": float(rel_cfg.tau),
                "source_prior": type_priors[dst_type].tolist(),
                "destination_prior": type_priors[src_type].tolist(),
                "baseline_matrix": np.broadcast_to(
                    type_priors[src_type][None, :],
                    (config.node_types[dst_type].num_classes, config.node_types[src_type].num_classes),
                ).tolist(),
                "aligned_joint": matrices["aligned_joint"].T.tolist(),
                "aligned_matrix": reverse_relation_matrix_from_joint(matrices["aligned_joint"], type_priors[dst_type]).tolist(),
                "effective_matrix": reverse_theoretical_matrix.tolist(),
                "theoretical_mutual_information_nats": float(theoretical_mi),
                "empirical_joint": reverse_empirical_joint.tolist(),
                "empirical_mutual_information_nats": float(empirical_mi),
                "num_edges": int(edge_index.size(1)),
                "is_computational_reverse_view": True,
                "shared_pair_name": relation_name,
            }
        elif config.add_reverse_relations:
            reverse_edge_type = make_reverse_edge_type(edge_type)
            data[reverse_edge_type].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            metadata["relations"][reverse_relation_name(relation_name)] = {
                "edge_type": list(reverse_edge_type),
                "source_type": dst_type,
                "destination_type": src_type,
                "out_degree": None,
                "tau": float(rel_cfg.tau),
                "is_reverse_copy": True,
                "copied_from": relation_name,
                "num_edges": int(edge_index.size(1)),
            }

    base_relations = [
        rel_meta
        for rel_meta in metadata["relations"].values()
        if not rel_meta.get("is_reverse_copy", False) and not rel_meta.get("is_computational_reverse_view", False)
    ]
    total_edges = int(sum(rel_meta["num_edges"] for rel_meta in base_relations))
    if total_edges > 0:
        empirical_weighted = sum(
            rel_meta["empirical_mutual_information_nats"] * rel_meta["num_edges"] for rel_meta in base_relations
        ) / total_edges
        theoretical_weighted = sum(
            rel_meta["theoretical_mutual_information_nats"] * rel_meta["num_edges"] for rel_meta in base_relations
        ) / total_edges
    else:
        empirical_weighted = 0.0
        theoretical_weighted = 0.0
    metadata["global_summary"] = {
        "num_forward_relations": len(base_relations),
        "num_forward_edges": total_edges,
        "I_avg_empirical_nats": float(empirical_weighted),
        "I_avg_theoretical_nats": float(theoretical_weighted),
    }
    destination_summary = {}
    for type_name in config.node_types:
        incoming_relations = [
            rel_meta
            for rel_meta in metadata["relations"].values()
            if not rel_meta.get("is_reverse_copy", False)
            if rel_meta["destination_type"] == type_name
        ]
        incoming_edges = int(sum(rel_meta["num_edges"] for rel_meta in incoming_relations))
        if incoming_edges > 0:
            incoming_empirical_weighted = sum(
                rel_meta["empirical_mutual_information_nats"] * rel_meta["num_edges"]
                for rel_meta in incoming_relations
            ) / incoming_edges
            incoming_theoretical_weighted = sum(
                rel_meta["theoretical_mutual_information_nats"] * rel_meta["num_edges"]
                for rel_meta in incoming_relations
            ) / incoming_edges
        else:
            incoming_empirical_weighted = 0.0
            incoming_theoretical_weighted = 0.0
        destination_summary[type_name] = {
            "num_forward_relations": len(incoming_relations),
            "num_forward_edges": incoming_edges,
            "relation_names": [
                relation_name
                for relation_name, rel_meta in metadata["relations"].items()
                if not rel_meta.get("is_reverse_copy", False)
                and rel_meta["destination_type"] == type_name
            ],
            "I_avg_empirical_nats": float(incoming_empirical_weighted),
            "I_avg_theoretical_nats": float(incoming_theoretical_weighted),
        }
    metadata["destination_summary"] = destination_summary

    if config.split is not None:
        split_cfg = config.split
        target_labels = type_labels[split_cfg.target_type]
        train_mask, val_mask, test_mask = _generate_split_masks(
            num_nodes=target_labels.shape[0],
            labels=target_labels,
            train_ratio=split_cfg.train_ratio,
            val_ratio=split_cfg.val_ratio,
            seed=split_cfg.seed,
        )
        data[split_cfg.target_type].train_mask = train_mask
        data[split_cfg.target_type].val_mask = val_mask
        data[split_cfg.target_type].test_mask = test_mask
        metadata["split"] = {
            "target_type": split_cfg.target_type,
            "train_ratio": float(split_cfg.train_ratio),
            "val_ratio": float(split_cfg.val_ratio),
            "test_ratio": float(1.0 - split_cfg.train_ratio - split_cfg.val_ratio),
            "seed": int(split_cfg.seed),
            "sizes": {
                "train": int(train_mask.sum().item()),
                "val": int(val_mask.sum().item()),
                "test": int(test_mask.sum().item()),
            },
        }

    data.hcsbm_metadata = metadata
    return data, metadata
