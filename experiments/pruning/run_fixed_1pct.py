from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_zoo.trainers.node_cls_trainer import set_seed
from pruning.core import (
    apply_pair_budget,
    build_pair_order,
    eligible_scored_groups,
    stable_int_seed,
)
from realworld_alignment.run_backbone_alignment import (
    active_metapaths,
    ensure_features,
    load_dataset,
    prepare_split,
    train_once,
)


DATASETS = [
    "acm",
    "aifb",
    "am",
    "aminer",
    "bgs",
    "dblp",
    "freebase",
    "imdb",
    "liar",
    "mutag",
    "yelp",
]
ARMS = ["full_graph", "rpi_low_first", "global_random_edges"]
RATIO = 0.01
BACKBONE = "rgcn"
KEYS = ["dataset", "split", "replicate", "arm"]


def read_config(
    config_dir: Path, dataset: str, split: int, args: argparse.Namespace
) -> dict:
    path = config_dir / dataset / str(split) / BACKBONE / "best_config.json"
    payload = json.loads(path.read_text())
    config = dict(payload.get("config", payload))
    if args.max_epochs is not None:
        config["epochs"] = min(int(config["epochs"]), int(args.max_epochs))
    if args.max_patience is not None:
        config["patience"] = min(
            int(config["patience"]), int(args.max_patience)
        )
    if args.device is not None:
        config["device"] = args.device
    return config


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    frame = frame[frame["status"] == "ok"]
    return {
        (str(row.dataset), int(row.split), int(row.replicate), str(row.arm))
        for row in frame.itertuples(index=False)
    }


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def relation_scores(
    frame: pd.DataFrame, dataset: str, split: int
) -> dict[str, tuple[float, int]]:
    rows = frame[
        (frame["dataset"] == dataset) & (frame["split"].astype(int) == split)
    ]
    return {
        str(row.relation): (float(row.RPI), int(row.num_edges))
        for row in rows.itertuples(index=False)
        if math.isfinite(float(row.RPI))
    }


def validate_rpi_artifacts(args: argparse.Namespace) -> Path:
    rpi_dir = args.rpi_dir.resolve()
    config_path = rpi_dir / "run_config.json"
    metrics_path = rpi_dir / "metrics" / "rpi_relation.csv"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing operational RPI under {rpi_dir}. Run scripts/run_pruning_experiment.sh."
        )
    config = json.loads(config_path.read_text())
    expected = {"seed": int(args.data_seed), "alpha": 0.9, "steps": 50}
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Operational RPI protocol mismatch: {mismatches}")
    if not set(args.datasets).issubset(set(config.get("datasets", []))):
        raise RuntimeError("Operational RPI does not cover every requested dataset")
    if not set(args.splits).issubset(set(config.get("splits", []))):
        raise RuntimeError("Operational RPI does not cover every requested split")
    return metrics_path


def ensure_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    payload = {
        "protocol": "table_3_fixed_1pct_reverse_paired",
        "datasets": list(args.datasets),
        "splits": [int(value) for value in args.splits],
        "backbone": BACKBONE,
        "arms": ARMS,
        "ratio": RATIO,
        "replicates": int(args.replicates),
        "seed_base": int(args.seed_base),
        "seed_stride": int(args.seed_stride),
        "data_seed": int(args.data_seed),
        "max_epochs": args.max_epochs,
        "max_patience": args.max_patience,
        "device": args.device,
        "max_cases": args.max_cases,
        "plan_only": bool(args.plan_only),
        "operational_rpi": {
            "pseudo_label_method": "hard_lp",
            "metric_scope": "full_relations",
            "alpha": 0.9,
            "steps": 50,
            "supervision": "train_target_labels_only",
        },
    }
    path = output_dir / "run_config.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(
                "Existing output uses a different protocol; choose a new --output-dir"
            )
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "fixed_1pct_runs.csv"
    metrics_path = validate_rpi_artifacts(args)
    rpi = pd.read_csv(metrics_path)
    required_columns = {
        "dataset",
        "split",
        "pseudo_label_method",
        "metric_scope",
        "relation",
        "num_edges",
        "RPI",
    }
    if not required_columns.issubset(rpi.columns):
        raise RuntimeError(
            f"Operational RPI is missing columns: {sorted(required_columns - set(rpi.columns))}"
        )
    rpi = rpi[
        (rpi["pseudo_label_method"] == "hard_lp")
        & (rpi["metric_scope"] == "full_relations")
    ]
    ensure_run_config(args, output_dir)
    done = completed_keys(raw_path)
    cases = [
        (dataset, split) for dataset in args.datasets for split in args.splits
    ]
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    total = len(cases) * int(args.replicates) * len(ARMS)
    progress = tqdm(total=total, desc="Table 3 fixed-1% runs", dynamic_ncols=True)
    for dataset, split in cases:
        set_seed(int(args.data_seed) + int(split))
        data, target = load_dataset(dataset, args.data_dir.resolve())
        ensure_features(data)
        prepare_split(data, target, seed=args.data_seed, split=split)
        metapaths = active_metapaths(dataset, data, target)
        config = read_config(args.config_dir.resolve(), dataset, split, args)
        groups = eligible_scored_groups(
            data,
            target,
            relation_scores(rpi, dataset, split),
            int(config["num_layers"]),
        )
        if not groups:
            raise RuntimeError(f"No eligible groups for {dataset} split={split}")
        eligible_units = sum(int(group["capacity_units"]) for group in groups)
        budget_units = max(1, int(round(eligible_units * RATIO)))
        seed_dataset = "aminer_legacy" if dataset == "aminer" else dataset
        for replicate in range(int(args.replicates)):
            train_seed = (
                int(args.seed_base)
                + replicate * int(args.seed_stride)
                + int(split)
            )
            for arm in ARMS:
                key = (dataset, int(split), int(replicate), arm)
                progress.set_postfix_str(
                    f"{dataset} s={split} {arm} seed={replicate}", refresh=False
                )
                if key in done:
                    progress.update()
                    continue
                graph_seed = int(args.data_seed) + stable_int_seed(
                    "exp4_v8",
                    seed_dataset,
                    split,
                    BACKBONE,
                    arm,
                    replicate,
                )
                status = "planned" if args.plan_only else "ok"
                error = ""
                val_acc = float("nan")
                test_acc = float("nan")
                removed = {}
                order_names = []
                removed_units = 0
                try:
                    if arm == "full_graph":
                        graph = data
                    else:
                        ordered_groups, ordered_units, order_names = build_pair_order(
                            groups, arm, graph_seed
                        )
                        graph, removed = apply_pair_budget(
                            data,
                            groups,
                            ordered_groups,
                            ordered_units,
                            budget_units,
                        )
                        removed_units = sum(removed.values())
                    if not args.plan_only:
                        result = train_once(
                            graph,
                            target,
                            metapaths,
                            BACKBONE,
                            config,
                            train_seed,
                        )
                        if result is None:
                            status = "train_failed"
                        else:
                            val_acc = float(result["best_val"]["accuracy"])
                            test_acc = float(result["test"]["accuracy"])
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                arm_budget = 0 if arm == "full_graph" else budget_units
                row = {
                    "protocol": "table_3_fixed_1pct_reverse_paired",
                    "dataset": dataset,
                    "split": int(split),
                    "backbone": BACKBONE,
                    "arm": arm,
                    "ratio": 0.0 if arm == "full_graph" else RATIO,
                    "replicate": int(replicate),
                    "train_seed": int(train_seed),
                    "graph_seed": int(graph_seed),
                    "relation_group_count": len(groups),
                    "eligible_pair_units": int(eligible_units),
                    "eligible_directed_edges": int(2 * eligible_units),
                    "budget_pair_units": int(arm_budget),
                    "budget_directed_edges": int(2 * arm_budget),
                    "removed_pair_units": int(removed_units),
                    "removed_directed_edges": int(2 * removed_units),
                    "budget_exact": bool(removed_units == arm_budget),
                    "reverse_pair_integrity": True,
                    "relation_order_json": json.dumps(order_names),
                    "removed_pair_units_by_group_json": json.dumps(
                        removed, sort_keys=True
                    ),
                    "best_val_acc": val_acc,
                    "test_acc": test_acc,
                    "status": status,
                    "error": error,
                }
                append_row(raw_path, row)
                if status == "ok":
                    done.add(key)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                progress.update()
    progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-1% R-GCN experiment for Table 3."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "realworld_alignment_best_configs",
    )
    parser.add_argument(
        "--rpi-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "table_3_fixed_1pct" / "operational_rpi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "table_3_fixed_1pct",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=9042)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-patience", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
