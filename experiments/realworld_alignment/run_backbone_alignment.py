#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_zoo.models.registry import MODEL_REGISTRY, build_model
from model_zoo.trainers.node_cls_trainer import NodeClassificationTrainer, generate_split_masks, set_seed
from realworld.compute_rpi import aggregate_rows, propagate_profiles, relation_rpi


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "realworld_alignment"
DEFAULT_BEST_CONFIG_DIR = PROJECT_ROOT / "configs" / "realworld_alignment_best_configs"
DEFAULT_DATASETS = ["acm", "aminer", "dblp", "freebase", "imdb", "liar", "yelp", "aifb", "am", "bgs", "mutag"]
DEFAULT_BACKBONES = ["han", "hetgnn", "rgcn", "rshn", "simplehgn"]

DATASET_META = {
    "acm": {"target": "paper", "metapaths": [[("paper", "to", "author"), ("author", "to", "paper")], [("paper", "to", "subject"), ("subject", "to", "paper")]]},
    "aminer": {"target": "paper", "metapaths": [[("paper", "to", "author"), ("author", "to", "paper")], [("paper", "to", "reference"), ("reference", "to", "paper")]]},
    "dblp": {"target": "author", "metapaths": [[("author", "to", "paper"), ("paper", "to", "author")], [("author", "to", "paper"), ("paper", "to", "term"), ("term", "to", "paper"), ("paper", "to", "author")], [("author", "to", "paper"), ("paper", "to", "venue"), ("venue", "to", "paper"), ("paper", "to", "author")]]},
    "freebase": {"target": "movie", "metapaths": [[("movie", "to", "actor"), ("actor", "to", "movie")], [("movie", "to", "director"), ("director", "to", "movie")], [("movie", "to", "writer"), ("writer", "to", "movie")]]},
    "imdb": {"target": "movie", "metapaths": [[("movie", ">actorh", "actor"), ("actor", "to", "movie")], [("movie", "to", "director"), ("director", "to", "movie")]]},
    "liar": {"target": "news", "metapaths": [[("news", "by", "speaker"), ("speaker", "wrote", "news")], [("news", "about", "subject"), ("subject", "in", "news")], [("news", "at", "context"), ("context", "for", "news")]]},
    "yelp": {"target": "business", "metapaths": [[("business", "to", "user"), ("user", "to", "business")], [("business", "to", "service"), ("service", "to", "business")], [("business", "to", "location"), ("location", "to", "business")]]},
    "aifb": {"target": "Personen", "metapaths": []},
    "am": {"target": "proxy", "metapaths": []},
    "bgs": {"target": "Lexicon_NamedRockUnit", "metapaths": []},
    "mutag": {"target": "d", "metapaths": []},
}


class TargetMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, target_type: str, dropout: float):
        super().__init__()
        self.target_type = target_type
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):
        return self.net(data[self.target_type].x.float())


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def load_dataset(dataset: str, data_dir: Path):
    path = data_dir / dataset / f"{dataset}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    target = DATASET_META[dataset]["target"]
    if data[target].y.dim() > 1:
        data[target].y = data[target].y.argmax(dim=1)
    return data, target


def ensure_features(data, *, dim: int = 128) -> None:
    for node_type in data.node_types:
        x = getattr(data[node_type], "x", None)
        if x is None or x.numel() == 0 or float(x.abs().sum().item()) == 0.0:
            x_new = torch.empty(int(data[node_type].num_nodes), dim)
            torch.nn.init.xavier_uniform_(x_new)
            data[node_type].x = x_new
        else:
            data[node_type].x = x.float()


def infer_metapaths(data, target: str) -> list[list[tuple[str, str, str]]]:
    metapaths = []
    seen = set()
    for et1 in data.edge_types:
        et1 = tuple(et1)
        if et1[0] != target or et1[2] == target:
            continue
        for et2 in data.edge_types:
            et2 = tuple(et2)
            if et2[0] == et1[2] and et2[2] == target and (et1, et2) not in seen:
                seen.add((et1, et2))
                metapaths.append([et1, et2])
    return metapaths


def active_metapaths(dataset: str, data, target: str) -> list[list[tuple[str, str, str]]]:
    configured = DATASET_META[dataset]["metapaths"]
    out = []
    for metapath in configured:
        mp = [tuple(edge_type) for edge_type in metapath]
        if mp and all(edge_type in data.edge_types for edge_type in mp):
            out.append(mp)
    return out or infer_metapaths(data, target)


def prepare_split(data, target: str, *, seed: int, split: int) -> None:
    split_seed = int(seed) + int(split) * 1009
    train, val, test = generate_split_masks(
        int(data[target].num_nodes),
        data[target].y,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=split_seed,
    )
    data[target].train_mask = train
    data[target].val_mask = val
    data[target].test_mask = test


def edge_type_name(edge_type: tuple[str, str, str]) -> str:
    return f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}"


def matrix_for_edge(data, edge_type: tuple[str, str, str]) -> sp.csr_matrix:
    edge_index = data[edge_type].edge_index.cpu().numpy()
    values = np.ones(edge_index.shape[1], dtype=bool)
    return sp.coo_matrix(
        (values, (edge_index[0], edge_index[1])),
        shape=(int(data[edge_type[0]].num_nodes), int(data[edge_type[2]].num_nodes)),
        dtype=bool,
    ).tocsr()


def binarize(matrix: sp.spmatrix) -> sp.csr_matrix:
    out = matrix.tocsr(copy=False)
    if out.nnz:
        out.sum_duplicates()
        out.data[:] = True
        out.eliminate_zeros()
    return out


def project_metapath(matrices: dict[tuple[str, str, str], sp.csr_matrix], metapath: list[tuple[str, str, str]], start: int, end: int) -> sp.csr_matrix:
    projected = binarize(matrices[metapath[0]][start:end])
    for edge_type in metapath[1:]:
        if projected.nnz == 0:
            break
        projected = binarize(projected @ matrices[edge_type])
    return projected


def compute_mhr(data, target_type: str, target_labels: torch.Tensor, metapaths: list[list[tuple[str, str, str]]], *, dataset: str, split: int, seed: int) -> tuple[list[dict], list[dict]]:
    labels = target_labels.view(-1).long().cpu().numpy()
    train_mask = data[target_type].train_mask.view(-1).bool().cpu().numpy()
    matrices = {edge_type: matrix_for_edge(data, edge_type) for metapath in metapaths for edge_type in metapath}
    rows = []
    vals = []
    for idx, metapath in enumerate(metapaths):
        total = 0
        same = 0
        train_total = 0
        train_same = 0
        n_target = int(labels.shape[0])
        first_nnz = max(1, int(matrices[metapath[0]].nnz))
        avg_first_degree = first_nnz / max(1, matrices[metapath[0]].shape[0])
        chunk_size = 128 if n_target >= 50_000 or avg_first_degree >= 100 else 1024
        for start in range(0, n_target, chunk_size):
            end = min(start + chunk_size, n_target)
            projected = project_metapath(matrices, metapath, start, end)
            if projected.nnz == 0:
                continue
            coo = projected.tocoo(copy=False)
            src = coo.row.astype(np.int64, copy=False) + start
            dst = coo.col.astype(np.int64, copy=False)
            valid = (src != dst) & (labels[src] >= 0) & (labels[dst] >= 0)
            if not np.any(valid):
                continue
            src = src[valid]
            dst = dst[valid]
            total += int(src.size)
            same += int(np.sum(labels[src] == labels[dst]))
            train_valid = train_mask[src] & train_mask[dst]
            if np.any(train_valid):
                train_total += int(np.sum(train_valid))
                train_same += int(np.sum(labels[src[train_valid]] == labels[dst[train_valid]]))
        mhr = float(same / total) if total > 0 else float("nan")
        train_mhr = float(train_same / train_total) if train_total > 0 else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "split": int(split),
                "seed": int(seed),
                "pseudo_label_method": "hard_lp",
                "metapath_name": f"mp{idx}",
                "metapath_relations": " | ".join(edge_type_name(edge_type) for edge_type in metapath),
                "num_projected_edges": int(total),
                "MHR": mhr,
                "MHR_train_endpoint_only_aux": train_mhr,
            }
        )
        if not math.isnan(mhr):
            vals.append(mhr)
    dataset_rows = [
        {
            "dataset": dataset,
            "split": int(split),
            "seed": int(seed),
            "pseudo_label_method": "hard_lp",
            "MHR_mean": sum(vals) / len(vals) if vals else float("nan"),
            "MHR_max": max(vals) if vals else float("nan"),
            "num_metapaths": len(metapaths),
        }
    ]
    return rows, dataset_rows


def compute_chr(data, target_type: str, profiles: dict[str, torch.Tensor], relations: list[tuple[str, str, str]], *, dataset: str, split: int, seed: int, alpha: float, steps: int) -> tuple[list[dict], list[dict]]:
    rows = []
    vals = []
    for edge_type in relations:
        src, _, dst = edge_type
        if target_type not in {src, dst} or src == dst:
            continue
        edge_index = data[edge_type].edge_index.cpu()
        src_profile = profiles[src][edge_index[0]].float()
        dst_profile = profiles[dst][edge_index[1]].float()
        if src == target_type:
            src_profile = torch.nn.functional.one_hot(src_profile.argmax(dim=1), num_classes=src_profile.size(1)).float()
        if dst == target_type:
            dst_profile = torch.nn.functional.one_hot(dst_profile.argmax(dim=1), num_classes=dst_profile.size(1)).float()
        score = (src_profile * dst_profile).sum(dim=1)
        chr_value = float(score.mean().item()) if score.numel() else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "split": int(split),
                "seed": int(seed),
                "metric_scope": "full_relations",
                "relation": edge_type_name(edge_type),
                "src_type": src,
                "dst_type": dst,
                "num_edges": int(score.numel()),
                "CHR": chr_value,
                "propagation_alpha": float(alpha),
                "propagation_steps": int(steps),
            }
        )
        if not math.isnan(chr_value):
            vals.append(chr_value)
    dataset_rows = [
        {
            "dataset": dataset,
            "split": int(split),
            "seed": int(seed),
            "metric_scope": "full_relations",
            "CTH_CHR": sum(vals) / len(vals) if vals else float("nan"),
            "num_target_incident_relations": len(vals),
        }
    ]
    return rows, dataset_rows


def model_config(base: dict, *, backbone: str, metapaths: list[list[tuple[str, str, str]]]) -> dict:
    cfg = {
        "hidden_channels": int(base["hidden_dim"]),
        "num_layers": int(base["num_layers"]),
        "heads": int(base.get("heads", 4)),
        "lr": float(base["lr"]),
        "epochs": int(base["epochs"]),
        "patience": int(base["patience"]),
        "eval_every": 1,
        "dropout": float(base["dropout"]),
        "weight_decay": float(base["weight_decay"]),
        "device": base["device"],
    }
    if backbone in MODEL_REGISTRY and MODEL_REGISTRY[backbone]["needs_metapaths"]:
        cfg["metapaths"] = metapaths
    return cfg


def train_once(data, target: str, metapaths: list[list[tuple[str, str, str]]], backbone: str, config: dict, seed: int) -> dict | None:
    set_seed(seed)
    num_classes = int(data[target].y[data[target].y >= 0].max().item() + 1)
    cfg = model_config(config, backbone=backbone, metapaths=metapaths)
    if backbone == "mlp":
        model = TargetMLP(int(data[target].x.size(-1)), int(config["hidden_dim"]), num_classes, target, float(config["dropout"]))
    else:
        if MODEL_REGISTRY[backbone]["needs_metapaths"] and not metapaths:
            return None
        model = build_model(backbone, data, target, num_classes, cfg)
    trainer = NodeClassificationTrainer(model, copy.deepcopy(data), target, cfg)
    result = trainer.run()
    result.update({"backbone": backbone, "seed": seed, "config": config})
    return result


def search_space(args: argparse.Namespace, backbone: str) -> list[dict]:
    heads = args.heads_grid if backbone == "simplehgn" else [4]
    if not args.tune_hparams:
        return [
            {
                "hidden_dim": 128,
                "lr": 0.005,
                "weight_decay": 5e-4,
                "dropout": 0.5,
                "heads": 4,
                "num_layers": int(args.num_layers),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "device": args.device,
            }
        ]
    rows = []
    for hidden in args.hidden_grid:
        for lr in args.lr_grid:
            for weight_decay in args.weight_decay_grid:
                for dropout in args.dropout_grid:
                    for heads_value in heads:
                        rows.append(
                            {
                                "hidden_dim": int(hidden),
                                "lr": float(lr),
                                "weight_decay": float(weight_decay),
                                "dropout": float(dropout),
                                "heads": int(heads_value),
                                "num_layers": int(args.num_layers),
                                "epochs": int(args.epochs),
                                "patience": int(args.patience),
                                "device": args.device,
                            }
                        )
    if args.max_trials is not None:
        rows = rows[: int(args.max_trials)]
    return rows


def load_preset_best_config(args: argparse.Namespace, dataset: str, split: int, backbone: str) -> tuple[dict, dict, Path]:
    path = args.best_config_dir / dataset / str(split) / backbone / "best_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing bundled best config for dataset={dataset} split={split} backbone={backbone}: {path}"
        )
    payload = json.loads(path.read_text())
    cfg = dict(payload.get("config", {}))
    required = {"hidden_dim", "lr", "weight_decay", "dropout"}
    missing = sorted(required - set(cfg))
    if missing:
        raise KeyError(f"Best config {path} is missing required fields: {missing}")
    cfg.setdefault("heads", 4)
    cfg.setdefault("num_layers", int(args.num_layers))
    cfg["epochs"] = int(args.epochs)
    cfg["patience"] = int(args.patience)
    cfg["device"] = args.device
    return payload, cfg, path


def tune_model(data, dataset: str, split: int, target: str, metapaths: list[list[tuple[str, str, str]]], backbone: str, args: argparse.Namespace, out_dir: Path) -> dict | None:
    best_path = out_dir / "best_config.json"
    if best_path.exists() and not args.overwrite:
        cached = json.loads(best_path.read_text())
        if args.use_best_configs and cached.get("selection_mode") != "preset_best_config":
            raise RuntimeError(
                f"Existing non-preset best_config found at {best_path}. "
                "Use a fresh output directory or pass --overwrite when using --use-best-configs."
            )
        return cached
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.use_best_configs:
        preset, cfg, preset_path = load_preset_best_config(args, dataset, split, backbone)
        try:
            result = train_once(data, target, metapaths, backbone, cfg, int(args.seed) + int(split))
        except (RuntimeError, ValueError) as exc:
            message = str(exc).lower()
            if any(token in message for token in ("out of memory", "cuda error", "cublas", "cudnn")):
                raise
            pd.DataFrame(
                [
                    {
                        "trial": preset.get("trial", -1),
                        "status": "failed",
                        "error": str(exc),
                        "selection_mode": "preset_best_config",
                        "preset_path": str(preset_path),
                        **cfg,
                    }
                ]
            ).to_csv(out_dir / "train_log.csv", index=False)
            return None
        if result is None:
            pd.DataFrame(
                [
                    {
                        "trial": preset.get("trial", -1),
                        "status": "skipped",
                        "selection_mode": "preset_best_config",
                        "preset_path": str(preset_path),
                        **cfg,
                    }
                ]
            ).to_csv(out_dir / "train_log.csv", index=False)
            return None
        val_acc = float(result["best_val"].get("accuracy", float("nan")))
        test_acc = float(result["test"].get("accuracy", float("nan")))
        pd.DataFrame(
            [
                {
                    "trial": preset.get("trial", -1),
                    "status": "ok",
                    "selection_mode": "preset_best_config",
                    "preset_path": str(preset_path),
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    **cfg,
                }
            ]
        ).to_csv(out_dir / "train_log.csv", index=False)
        best = {
            "trial": preset.get("trial", -1),
            "dataset": dataset,
            "split": int(split),
            "backbone": backbone,
            "config": cfg,
            "best_val": result["best_val"],
            "test": result["test"],
            "epochs_trained": result["epochs_trained"],
            "selection_mode": "preset_best_config",
            "preset_path": str(preset_path),
            "preset_best_val": preset.get("best_val", {}),
            "preset_test": preset.get("test", {}),
        }
        save_json(best, best_path)
        return best
    best = None
    best_score = -math.inf
    train_rows = []
    for trial, cfg in enumerate(search_space(args, backbone)):
        try:
            result = train_once(data, target, metapaths, backbone, cfg, int(args.seed) + int(split))
        except (RuntimeError, ValueError) as exc:
            message = str(exc).lower()
            if any(token in message for token in ("out of memory", "cuda error", "cublas", "cudnn")):
                raise
            train_rows.append({"trial": trial, "status": "failed", "error": str(exc), **cfg})
            continue
        if result is None:
            train_rows.append({"trial": trial, "status": "skipped", **cfg})
            continue
        val_acc = float(result["best_val"].get("accuracy", float("nan")))
        test_acc = float(result["test"].get("accuracy", float("nan")))
        train_rows.append({"trial": trial, "status": "ok", "val_acc": val_acc, "test_acc": test_acc, **cfg})
        if not math.isnan(val_acc) and val_acc > best_score:
            best_score = val_acc
            best = {
                "trial": trial,
                "dataset": dataset,
                "split": int(split),
                "backbone": backbone,
                "config": cfg,
                "best_val": result["best_val"],
                "test": result["test"],
                "epochs_trained": result["epochs_trained"],
            }
    pd.DataFrame(train_rows).to_csv(out_dir / "train_log.csv", index=False)
    if best is not None:
        save_json(best, best_path)
    return best


def append_or_replace(path: Path, new_rows: list[dict], keys: list[str], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_csv(path)
    new = pd.DataFrame(new_rows)
    if old.empty:
        new.to_csv(path, index=False)
        return
    if overwrite:
        mask = pd.Series(True, index=old.index)
        for _, row in new.iterrows():
            row_mask = pd.Series(True, index=old.index)
            for key in keys:
                row_mask &= old[key].astype(str) == str(row[key])
            mask &= ~row_mask
        old = old[mask]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def run_one_dataset_split(dataset: str, split: int, args: argparse.Namespace) -> None:
    data, target = load_dataset(dataset, args.data_dir)
    set_seed(int(args.seed) + int(split))
    ensure_features(data)
    prepare_split(data, target, seed=int(args.seed), split=int(split))
    relations = sorted(tuple(edge_type) for edge_type in data.edge_types)
    metapaths = active_metapaths(dataset, data, target)

    payload = propagate_profiles(data, target, relations, alpha=args.lp_alpha, steps=args.lp_steps)
    rel_rows = relation_rpi(data, payload["hard_labels"], relations, dataset, split, int(args.seed))
    type_rows, target_rows = aggregate_rows(rel_rows, target)
    mhr_rows, mhr_dataset_rows = compute_mhr(data, target, payload["hard_labels"][target], metapaths, dataset=dataset, split=split, seed=int(args.seed))
    chr_rows, chr_dataset_rows = compute_chr(data, target, payload["profiles"], relations, dataset=dataset, split=split, seed=int(args.seed), alpha=args.lp_alpha, steps=args.lp_steps)

    metrics_dir = args.output_dir / "metrics"
    append_or_replace(metrics_dir / "rpi_relation.csv", rel_rows, ["dataset", "split", "relation"], overwrite=args.overwrite)
    append_or_replace(metrics_dir / "rpi_type.csv", type_rows, ["dataset", "split", "node_type"], overwrite=args.overwrite)
    append_or_replace(metrics_dir / "rpi_target.csv", target_rows, ["dataset", "split", "node_type"], overwrite=args.overwrite)
    append_or_replace(metrics_dir / "mhr_metapath.csv", mhr_rows, ["dataset", "split", "metapath_name"], overwrite=args.overwrite)
    append_or_replace(metrics_dir / "mhr_dataset.csv", mhr_dataset_rows, ["dataset", "split"], overwrite=args.overwrite)
    append_or_replace(metrics_dir / "chr_relation.csv", chr_rows, ["dataset", "split", "relation"], overwrite=args.overwrite)
    append_or_replace(metrics_dir / "chr_dataset.csv", chr_dataset_rows, ["dataset", "split"], overwrite=args.overwrite)

    performance_rows = []
    best_summary_rows = []
    for backbone in ["mlp", *args.backbones]:
        best = tune_model(data, dataset, split, target, metapaths, backbone, args, args.output_dir / "models" / dataset / str(split) / backbone)
        if best is None:
            continue
        cfg = best["config"]
        performance_rows.append(
            {
                "dataset": dataset,
                "split": int(split),
                "seed": int(args.seed) + int(split),
                "backbone": backbone,
                "num_layers": cfg["num_layers"],
                "best_val_acc": best["best_val"].get("accuracy", float("nan")),
                "test_acc": best["test"].get("accuracy", float("nan")),
                "test_macro_f1": best["test"].get("macro_f1", float("nan")),
                "selected_hidden_dim": cfg["hidden_dim"],
                "selected_lr": cfg["lr"],
                "selected_weight_decay": cfg["weight_decay"],
                "selected_dropout": cfg["dropout"],
                "selected_heads": cfg.get("heads", 4),
                "hyperparameter_config_id": best["trial"],
            }
        )
        best_summary_rows.append({"dataset": dataset, "split": int(split), "backbone": backbone, **cfg, "trial": best["trial"], "val_accuracy": best["best_val"].get("accuracy", float("nan")), "test_accuracy": best["test"].get("accuracy", float("nan"))})
    append_or_replace(args.output_dir / "performance" / "performance.csv", performance_rows, ["dataset", "split", "backbone"], overwrite=args.overwrite)
    append_or_replace(args.output_dir / "results" / "best_configs_summary.csv", best_summary_rows, ["dataset", "split", "backbone"], overwrite=args.overwrite)
    print(f"[done] dataset={dataset} split={split} backbones={','.join([r['backbone'] for r in performance_rows])}", flush=True)


def build_alignment_tables(output_dir: Path) -> None:
    perf = read_csv(output_dir / "performance" / "performance.csv")
    if perf.empty or "mlp" not in set(perf["backbone"].astype(str)):
        raise RuntimeError("No MLP baseline performance found; cannot compute HGNN deltas.")
    mlp = perf[perf["backbone"] == "mlp"][["dataset", "split", "test_acc"]].rename(columns={"test_acc": "mlp_test_acc"})
    hgnn = perf[perf["backbone"] != "mlp"].merge(mlp, on=["dataset", "split"], how="inner")
    hgnn["delta_hgnn"] = hgnn["test_acc"].astype(float) - hgnn["mlp_test_acc"].astype(float)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    hgnn.to_csv(results_dir / "alignment_backbone_level.csv", index=False)
    perf.to_csv(results_dir / "performance_selected_backbones.csv", index=False)

    rpi_target = read_csv(output_dir / "metrics" / "rpi_target.csv")
    mhr = read_csv(output_dir / "metrics" / "mhr_dataset.csv")
    chr_df = read_csv(output_dir / "metrics" / "chr_dataset.csv")
    delta = hgnn.groupby(["dataset", "split"], as_index=False).agg(
        avg_delta_hgnn=("delta_hgnn", "mean"),
        median_delta_hgnn=("delta_hgnn", "median"),
    )
    align = delta.merge(rpi_target[["dataset", "split", "RPI_type"]].rename(columns={"RPI_type": "RPI_target"}), on=["dataset", "split"], how="left")
    align = align.merge(mhr[["dataset", "split", "MHR_mean"]], on=["dataset", "split"], how="left")
    align = align.merge(chr_df[["dataset", "split", "CTH_CHR"]], on=["dataset", "split"], how="left")
    align.to_csv(results_dir / "alignment_dataset_level.csv", index=False)

    family_rows = []
    for backbone in sorted(set(perf["backbone"].astype(str))):
        if backbone == "mlp":
            family = "baseline"
        else:
            family = MODEL_REGISTRY.get(backbone, {}).get("family", "unknown")
        family_rows.append({"backbone": backbone, "family": family})
    pd.DataFrame(family_rows).to_csv(results_dir / "model_families.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-world HGNN alignment experiments and draw the alignment figure.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--tune-hparams", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-best-configs", action="store_true", help="Use bundled tuned best configs instead of running a grid search.")
    parser.add_argument("--best-config-dir", type=Path, default=DEFAULT_BEST_CONFIG_DIR)
    parser.add_argument("--hidden-grid", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--lr-grid", nargs="+", type=float, default=[0.001, 0.003, 0.005, 0.01])
    parser.add_argument("--weight-decay-grid", nargs="+", type=float, default=[0.0, 1e-5, 5e-5, 1e-4])
    parser.add_argument("--dropout-grid", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    parser.add_argument("--heads-grid", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--lp-alpha", type=float, default=0.9)
    parser.add_argument("--lp-steps", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.best_config_dir = args.best_config_dir.resolve()
    args.backbones = [b.lower() for b in args.backbones]
    unknown = [b for b in args.backbones if b not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown backbones: {unknown}. Available: {sorted(MODEL_REGISTRY)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.output_dir / "run_config.json")
    for split in args.splits:
        for dataset in args.datasets:
            run_one_dataset_split(dataset, int(split), args)
    build_alignment_tables(args.output_dir)
    print(f"[saved] {args.output_dir / 'results' / 'alignment_backbone_level.csv'}", flush=True)


if __name__ == "__main__":
    main()
