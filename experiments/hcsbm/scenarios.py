from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .config import HCSBMConfig, NodeTypeConfig, RelationConfig, SplitConfig


def make_hcsbm_config(
    *,
    node_types: Mapping[str, NodeTypeConfig],
    relations: Mapping[str, RelationConfig],
    feature_dim: int,
    allow_multi_edges: bool = False,
    add_reverse_relations: bool = False,
    seed: int = 0,
    split: SplitConfig | None = None,
):
    return HCSBMConfig(
        node_types=dict(node_types),
        relations=dict(relations),
        feature_dim=feature_dim,
        allow_multi_edges=allow_multi_edges,
        add_reverse_relations=add_reverse_relations,
        edge_set_mode="shared_undirected",
        seed=seed,
        split=split,
    )


def simple_star_schema(
    *,
    feature_dim: int,
    seed: int = 0,
    allow_multi_edges: bool = False,
    add_reverse_relations: bool = False,
) -> HCSBMConfig:
    return make_hcsbm_config(
        feature_dim=feature_dim,
        seed=seed,
        allow_multi_edges=allow_multi_edges,
        add_reverse_relations=add_reverse_relations,
        split=SplitConfig(target_type="paper", train_ratio=0.6, val_ratio=0.2, seed=seed),
        node_types={
            "author": NodeTypeConfig(num_nodes=400, num_classes=3, class_prior=(0.5, 0.3, 0.2), gamma=2.5, sigma=0.7),
            "paper": NodeTypeConfig(num_nodes=700, num_classes=5, class_prior=(0.2, 0.2, 0.2, 0.2, 0.2), gamma=1.5, sigma=1.0),
            "venue": NodeTypeConfig(num_nodes=120, num_classes=2, class_prior=(0.65, 0.35), gamma=3.0, sigma=0.6),
            "topic": NodeTypeConfig(num_nodes=180, num_classes=4, class_prior=(0.1, 0.2, 0.3, 0.4), gamma=1.0, sigma=1.2),
        },
        relations={
            "writes": RelationConfig(source_type="author", destination_type="paper", out_degree=3, tau=0.8),
            "published_in": RelationConfig(source_type="paper", destination_type="venue", out_degree=1, tau=0.6),
            "has_topic": RelationConfig(source_type="paper", destination_type="topic", out_degree=2, tau=0.3),
        },
    )


def complete_directed_four_type_config(
    *,
    feature_dim: int,
    gamma: float,
    tau: float,
    sigma: float = 1.0,
    out_degree: int = 6,
    num_nodes: int = 600,
    class_counts: Sequence[int] = (3, 4, 5, 4),
    type_names: Sequence[str] = ("A", "B", "C", "D"),
    seed: int = 0,
    allow_multi_edges: bool = False,
    add_reverse_relations: bool = False,
    target_type: str = "A",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> HCSBMConfig:
    if len(type_names) != 4:
        raise ValueError("type_names must have length 4")
    if len(class_counts) != len(type_names):
        raise ValueError("class_counts must match type_names length")

    node_types = {}
    for type_name, num_classes in zip(type_names, class_counts):
        prior = tuple([1.0 / float(num_classes)] * int(num_classes))
        node_types[type_name] = NodeTypeConfig(
            num_nodes=int(num_nodes),
            num_classes=int(num_classes),
            class_prior=prior,
            gamma=float(gamma),
            sigma=float(sigma),
        )

    relations = {}
    for i, src in enumerate(type_names):
        for dst in type_names[i + 1:]:
            relations[f"{src}_to_{dst}"] = RelationConfig(
                source_type=src,
                destination_type=dst,
                out_degree=int(out_degree),
                tau=float(tau),
            )

    return make_hcsbm_config(
        node_types=node_types,
        relations=relations,
        feature_dim=feature_dim,
        allow_multi_edges=allow_multi_edges,
        add_reverse_relations=add_reverse_relations,
        seed=seed,
        split=SplitConfig(target_type=target_type, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed),
    )
