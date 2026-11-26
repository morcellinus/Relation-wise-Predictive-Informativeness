from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NodeTypeConfig:
    num_nodes: int
    num_classes: int
    class_prior: Sequence[float]
    gamma: float
    sigma: float
    feature_dim: Optional[int] = None
    prototype_mode: str = "orthonormal"


@dataclass(frozen=True)
class RelationConfig:
    source_type: str
    destination_type: str
    out_degree: int
    tau: float
    aligned_matrix: Optional[Sequence[Sequence[float]]] = None


@dataclass(frozen=True)
class SplitConfig:
    target_type: str
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class HCSBMConfig:
    node_types: Mapping[str, NodeTypeConfig]
    relations: Mapping[str, RelationConfig]
    feature_dim: int
    allow_multi_edges: bool = False
    add_reverse_relations: bool = False
    edge_set_mode: str = "shared_undirected"
    seed: int = 0
    split: Optional[SplitConfig] = None
    relation_aliases: Mapping[str, Tuple[str, str, str]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not self.node_types:
            raise ValueError("At least one node type must be provided")
        if not self.relations:
            raise ValueError("At least one relation must be provided")
        if self.edge_set_mode not in {"shared_undirected", "directed_independent"}:
            raise ValueError("edge_set_mode must be 'shared_undirected' or 'directed_independent'")
        for type_name, cfg in self.node_types.items():
            if cfg.num_nodes <= 0:
                raise ValueError(f"num_nodes must be positive for node type {type_name!r}")
            if cfg.num_classes <= 0:
                raise ValueError(f"num_classes must be positive for node type {type_name!r}")
            if len(cfg.class_prior) != cfg.num_classes:
                raise ValueError(
                    f"class_prior length mismatch for node type {type_name!r}: "
                    f"expected {cfg.num_classes}, got {len(cfg.class_prior)}"
                )
            if cfg.gamma <= 0:
                raise ValueError(f"gamma must be positive for node type {type_name!r}")
            if cfg.sigma <= 0:
                raise ValueError(f"sigma must be positive for node type {type_name!r}")
        seen_pairs = set()
        for relation_name, rel in self.relations.items():
            if rel.source_type not in self.node_types:
                raise ValueError(f"Unknown source type {rel.source_type!r} in relation {relation_name!r}")
            if rel.destination_type not in self.node_types:
                raise ValueError(f"Unknown destination type {rel.destination_type!r} in relation {relation_name!r}")
            if rel.source_type == rel.destination_type:
                raise ValueError(f"Self-pair relations are not supported in relation {relation_name!r}")
            if rel.out_degree <= 0:
                raise ValueError(f"out_degree must be positive in relation {relation_name!r}")
            if not 0.0 <= rel.tau <= 1.0:
                raise ValueError(f"tau must be in [0, 1] for relation {relation_name!r}")
            if rel.aligned_matrix is not None:
                if len(rel.aligned_matrix) != self.node_types[rel.source_type].num_classes:
                    raise ValueError(
                        f"aligned_matrix row count mismatch for relation {relation_name!r}: "
                        f"expected {self.node_types[rel.source_type].num_classes}, got {len(rel.aligned_matrix)}"
                    )
                for row_idx, row in enumerate(rel.aligned_matrix):
                    if len(row) != self.node_types[rel.destination_type].num_classes:
                        raise ValueError(
                            f"aligned_matrix column count mismatch for relation {relation_name!r} row {row_idx}: "
                            f"expected {self.node_types[rel.destination_type].num_classes}, got {len(row)}"
                        )
            if self.edge_set_mode == "shared_undirected":
                pair_key = tuple(sorted((rel.source_type, rel.destination_type)))
                if pair_key in seen_pairs:
                    raise ValueError(
                        "shared_undirected mode expects at most one relation entry per unordered pair; "
                        f"duplicate pair {pair_key!r} found at relation {relation_name!r}"
                    )
                seen_pairs.add(pair_key)
        if self.split is not None and self.split.target_type not in self.node_types:
            raise ValueError(f"Unknown split target type {self.split.target_type!r}")
