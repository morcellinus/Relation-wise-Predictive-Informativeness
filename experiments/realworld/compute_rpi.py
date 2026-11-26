#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT.parent, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rpi import _core
from simpleheterosage.trainers.node_cls_trainer import generate_split_masks


DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "realworld_rpi"
DEFAULT_DATASETS = [
    "acm",
    "aminer",
    "dblp",
    "freebase",
    "imdb",
    "liar",
    "yelp",
    "aifb",
    "am",
    "bgs",
    "mutag",
]

TARGET_TYPES = {
    "acm": "paper",
    "aminer": "paper",
    "dblp": "author",
    "freebase": "movie",
    "imdb": "movie",
    "liar": "news",
    "yelp": "business",
    "aifb": "Personen",
    "am": "proxy",
    "bgs": "Lexicon_NamedRockUnit",
    "mutag": "d",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(dataset: str, data_dir: Path):
    path = data_dir / dataset / f"{dataset}.pt"
    if not path.exists():
        path = data_dir / f"{dataset}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    target = TARGET_TYPES[dataset]
    if data[target].y.dim() > 1:
        data[target].y = data[target].y.argmax(dim=1)
    return data, target


def stratified_masks(y: torch.Tensor, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    return generate_split_masks(y.numel(), y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)


def propagate_profiles(data, target_type: str, relations: list[tuple[str, str, str]], alpha: float, steps: int):
    return _core.propagate_profiles(
        data,
        target_type,
        relations,
        alpha,
        steps,
        y=data[target_type].y.view(-1).long().cpu(),
        train_mask=data[target_type].train_mask.view(-1).bool().cpu(),
    )


def relation_rpi(data, labels: dict[str, torch.Tensor], relations: list[tuple[str, str, str]], dataset: str, split: int, seed: int):
    metadata = {
        "dataset": dataset,
        "split": int(split),
        "seed": int(seed),
        "pseudo_label_method": "hard_lp",
        "metric_scope": "full_relations",
    }
    return [
        {**metadata, **row, "RPI_weighted_aux": row["RPI"]}
        for row in _core.score_relations(data, labels, relations)
    ]


def aggregate_rows(rows: list[dict], target_type: str):
    first = rows[0] if rows else {}
    fields = ("dataset", "split", "seed", "pseudo_label_method", "metric_scope")
    metadata = {field: first.get(field) for field in fields}
    type_rows = [{**metadata, **row} for row in _core.aggregate_by_destination(rows)]
    target_rows = [row for row in type_rows if row["node_type"] == target_type]
    return type_rows, target_rows


def save_pseudo_payload(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload["profiles"], out_dir / "pseudo_profiles.pt")
    torch.save(payload["hard_labels"], out_dir / "pseudo_hard_labels.pt")
    torch.save(payload["confidence"], out_dir / "pseudo_confidence.pt")
    with open(out_dir / "pseudo_metadata.json", "w") as f:
        json.dump(payload["metadata"], f, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> None:
    out = args.output_dir
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    all_relation_rows = []
    all_type_rows = []
    all_target_rows = []
    for dataset in args.datasets:
        data, target = load_dataset(dataset, args.data_dir)
        relations = sorted(tuple(et) for et in data.edge_types)
        for split in args.splits:
            seed = int(args.seed)
            set_seed(seed + int(split))
            train, val, test = stratified_masks(data[target].y, seed=args.seed + split * 1009)
            data[target].train_mask = train
            data[target].val_mask = val
            data[target].test_mask = test
            payload = propagate_profiles(data, target, relations, alpha=args.alpha, steps=args.steps)
            if args.save_pseudo:
                save_pseudo_payload(payload, out / "pseudo_labels" / dataset / str(split) / "hard_lp")
            rel_rows = relation_rpi(data, payload["hard_labels"], relations, dataset, split, seed)
            type_rows, target_rows = aggregate_rows(rel_rows, target)
            all_relation_rows.extend(rel_rows)
            all_type_rows.extend(type_rows)
            all_target_rows.extend(target_rows)
            print(f"[done] dataset={dataset} split={split} target={target} relations={len(relations)}")
    pd.DataFrame(all_relation_rows).to_csv(out / "metrics" / "rpi_relation.csv", index=False)
    pd.DataFrame(all_type_rows).to_csv(out / "metrics" / "rpi_type.csv", index=False)
    pd.DataFrame(all_target_rows).to_csv(out / "metrics" / "rpi_target.csv", index=False)
    with open(out / "run_config.json", "w") as f:
        json.dump(vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)}, f, indent=2, default=str)
    print(f"saved metrics under {out / 'metrics'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute train-label-induced relation-level RPI for 11 real-world datasets.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save-pseudo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    run(args)


if __name__ == "__main__":
    main()
