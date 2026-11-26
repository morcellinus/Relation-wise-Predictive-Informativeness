#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["acm", "aifb", "am", "aminer", "bgs", "dblp", "freebase", "imdb", "liar", "mutag", "yelp"]
BACKBONES = ["mlp", "han", "hetgnn", "rgcn", "rshn", "simplehgn"]


def main() -> None:
    rows = []
    for dataset in DATASETS:
        path = ROOT / "data" / "processed" / dataset / f"{dataset}.pt"
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Missing physical dataset file: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "dataset": dataset,
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "node_types": len(data.node_types),
                "edge_types": len(data.edge_types),
            }
        )
    config_count = 0
    table_3_config_count = 0
    for dataset in DATASETS:
        for split in range(5):
            for backbone in BACKBONES:
                path = ROOT / "configs" / "realworld_alignment_best_configs" / dataset / str(split) / backbone / "best_config.json"
                if not path.is_file():
                    raise RuntimeError(f"Missing best config: {path}")
                config_count += 1
                if backbone == "rgcn":
                    table_3_config_count += 1
    table_3_files = [
        ROOT / "pruning" / "core.py",
        ROOT / "pruning" / "run_fixed_1pct.py",
        ROOT / "pruning" / "analyze_fixed_1pct.py",
        ROOT / "scripts" / "run_pruning_experiment.sh",
        ROOT / "scripts" / "build_pruning_table.sh",
    ]
    missing = [str(path.relative_to(ROOT)) for path in table_3_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing Table 3 files: {missing}")
    result = {
        "datasets": rows,
        "best_configs": config_count,
        "table_3_rgcn_configs": table_3_config_count,
        "table_3_protocol": "fixed_1pct_reverse_paired",
        "status": "ok",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
