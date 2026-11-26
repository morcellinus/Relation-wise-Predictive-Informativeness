from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from .config import HCSBMConfig, NodeTypeConfig, RelationConfig, SplitConfig


def expand_yaml_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.yaml")))
            paths.extend(sorted(path.glob("*.yml")))
        else:
            paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ValueError("No YAML config files were found")
    return deduped


def load_yaml_payload(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML payload at {path} must be a mapping")
    return payload


def payload_to_hcsbm_config(payload: dict) -> HCSBMConfig:
    split = payload.get("split")
    return HCSBMConfig(
        node_types={name: NodeTypeConfig(**cfg) for name, cfg in payload["node_types"].items()},
        relations={name: RelationConfig(**cfg) for name, cfg in payload["relations"].items()},
        feature_dim=int(payload["feature_dim"]),
        allow_multi_edges=bool(payload.get("allow_multi_edges", False)),
        add_reverse_relations=bool(payload.get("add_reverse_relations", False)),
        edge_set_mode=str(payload.get("edge_set_mode", "shared_undirected")),
        seed=int(payload.get("seed", 0)),
        split=SplitConfig(**split) if split is not None else None,
        relation_aliases={name: tuple(edge_type) for name, edge_type in payload.get("relation_aliases", {}).items()},
    )


def build_sweep_config_from_yaml(
    payload: dict,
    *,
    gamma: float,
    tau: float,
    seed: int,
    target_type: str | None = None,
    split_seed: int | None = None,
) -> HCSBMConfig:
    sweep = payload.get("sweep", {})
    node_gamma = sweep.get("node_gamma", {})
    relation_tau = sweep.get("relation_tau", {})

    node_types = {}
    for name, cfg in payload["node_types"].items():
        cfg = dict(cfg)
        if name in node_gamma:
            gamma_value = node_gamma[name]
        elif "*" in node_gamma:
            gamma_value = node_gamma["*"]
        else:
            gamma_value = None
        if gamma_value == "sweep":
            cfg["gamma"] = float(gamma)
        elif gamma_value is not None:
            cfg["gamma"] = float(gamma_value)
        node_types[name] = NodeTypeConfig(**cfg)

    relations = {}
    for name, cfg in payload["relations"].items():
        cfg = dict(cfg)
        tau_value = relation_tau.get(name, relation_tau.get("*", "sweep"))
        cfg["tau"] = float(tau if tau_value == "sweep" else tau_value)
        relations[name] = RelationConfig(**cfg)

    split_payload = payload.get("split")
    split = None
    if split_payload is not None:
        split_cfg = dict(split_payload)
        if target_type is not None:
            split_cfg["target_type"] = target_type
        if split_seed is not None:
            split_cfg["seed"] = int(split_seed)
        split = SplitConfig(**split_cfg)
    elif target_type is not None:
        split = SplitConfig(target_type=target_type, seed=0 if split_seed is None else int(split_seed))

    cfg = HCSBMConfig(
        node_types=node_types,
        relations=relations,
        feature_dim=int(payload["feature_dim"]),
        allow_multi_edges=bool(payload.get("allow_multi_edges", False)),
        add_reverse_relations=bool(payload.get("add_reverse_relations", False)),
        edge_set_mode=str(payload.get("edge_set_mode", "shared_undirected")),
        seed=int(seed),
        split=split,
        relation_aliases={name: tuple(edge_type) for name, edge_type in payload.get("relation_aliases", {}).items()},
    )
    cfg.validate()
    return cfg
