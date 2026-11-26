from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ORDER = [
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
DISPLAY = {
    "acm": "ACM",
    "aifb": "AIFB",
    "am": "AM",
    "aminer": "AMiner",
    "bgs": "BGS",
    "dblp": "DBLP",
    "freebase": "Freebase",
    "imdb": "IMDB",
    "liar": "LIAR",
    "mutag": "MUTAG",
    "yelp": "Yelp",
}
ARMS = ["full_graph", "rpi_low_first", "global_random_edges"]
ARM_COLUMNS = {
    "full_graph": "full_graph",
    "rpi_low_first": "operational_rpi",
    "global_random_edges": "global_random",
}


def build_tables(
    raw_path: Path, allow_incomplete: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(raw_path)
    frame = frame[frame["status"] == "ok"].copy()
    if set(frame["protocol"]) != {"table_3_fixed_1pct_reverse_paired"}:
        raise RuntimeError("Unexpected experiment protocol")
    keys = ["dataset", "split", "backbone", "replicate", "arm"]
    if frame.duplicated(keys, keep=False).any():
        raise RuntimeError("Duplicate successful runs detected")
    if set(frame["backbone"]) != {"rgcn"}:
        raise RuntimeError(f"Unexpected backbones: {sorted(frame['backbone'].unique())}")
    if set(frame["arm"]) != set(ARMS):
        raise RuntimeError(f"Unexpected arms: {sorted(frame['arm'].unique())}")
    pruned = frame[frame["arm"] != "full_graph"]
    full = frame[frame["arm"] == "full_graph"]
    if not bool(np.isclose(pruned["ratio"], 0.01).all()):
        raise RuntimeError("Every pruned run must use the fixed 1% ratio")
    if not bool(np.isclose(full["ratio"], 0.0).all()):
        raise RuntimeError("Full-graph rows must use ratio zero")
    if not bool(frame["budget_exact"].all()):
        raise RuntimeError("At least one run violates its exact edge budget")
    if not bool(frame["reverse_pair_integrity"].all()):
        raise RuntimeError("At least one run violates reverse-pair integrity")
    case_index = ["dataset", "split", "backbone", "replicate"]
    cases = frame.pivot(index=case_index, columns="arm", values="test_acc").reset_index()
    missing = [arm for arm in ARMS if arm not in cases.columns]
    if missing:
        raise RuntimeError(f"Missing arms: {missing}")
    cases = cases.rename(columns=ARM_COLUMNS)
    method_columns = list(ARM_COLUMNS.values())
    if bool(cases[method_columns].isna().any().any()):
        raise RuntimeError("At least one paired case is missing a successful arm")
    if not allow_incomplete:
        if set(cases["dataset"]) != set(DATASET_ORDER):
            raise RuntimeError(
                f"Expected eleven datasets, found: {sorted(cases['dataset'].unique())}"
            )
        if set(cases["replicate"].astype(int)) != set(range(5)):
            raise RuntimeError("Expected training-seed replicates 0 through 4")
        split_counts = cases.groupby(["dataset", "replicate"])["split"].nunique()
        seed_counts = cases.groupby("dataset")["replicate"].nunique()
        if not bool(split_counts.eq(5).all()):
            raise RuntimeError(
                f"Each dataset-seed requires five splits: {split_counts.value_counts().to_dict()}"
            )
        if not bool(seed_counts.eq(5).all()):
            raise RuntimeError(
                f"Each dataset requires five training seeds: {seed_counts.to_dict()}"
            )
    seed_rows = cases.groupby(["dataset", "replicate"], as_index=False).agg(
        full_graph=("full_graph", "mean"),
        operational_rpi=("operational_rpi", "mean"),
        global_random=("global_random", "mean"),
        splits=("split", "nunique"),
    )
    summary = seed_rows.groupby("dataset", as_index=False).agg(
        full_mean=("full_graph", "mean"),
        full_std=("full_graph", "std"),
        rpi_mean=("operational_rpi", "mean"),
        rpi_std=("operational_rpi", "std"),
        global_random_mean=("global_random", "mean"),
        global_random_std=("global_random", "std"),
        seeds=("replicate", "nunique"),
    )
    summary["rpi_minus_full_pp"] = 100.0 * (
        summary["rpi_mean"] - summary["full_mean"]
    )
    summary["rpi_minus_global_random_pp"] = 100.0 * (
        summary["rpi_mean"] - summary["global_random_mean"]
    )
    summary["rpi_strict_best"] = summary["rpi_mean"] > summary[
        ["full_mean", "global_random_mean"]
    ].max(axis=1)
    summary["rpi_co_best"] = summary["rpi_mean"] >= summary[
        ["full_mean", "global_random_mean"]
    ].max(axis=1)
    return cases, seed_rows, summary


def latex_table(summary: pd.DataFrame) -> str:
    indexed = summary.set_index("dataset")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset-level test accuracy (\%) of R-GCN on the full graph, after fixed 1\% operational-RPI-guided reverse-paired edge pruning, and after exact-budget global-random reverse-paired edge pruning. Values are mean $\pm$ sample standard deviation over five training seeds after averaging each seed across five splits. The highest unrounded mean is bold.}",
        r"\label{tab:fixed_rpi_pruning}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Dataset & Full graph & Operational-RPI & Global random \\",
        r"\midrule",
    ]
    columns = [
        ("full_mean", "full_std"),
        ("rpi_mean", "rpi_std"),
        ("global_random_mean", "global_random_std"),
    ]
    for dataset in DATASET_ORDER:
        if dataset not in indexed.index:
            continue
        row = indexed.loc[dataset]
        maximum = max(float(row[mean]) for mean, _ in columns)
        cells = []
        for mean, std in columns:
            value = f"{100 * row[mean]:.3f} \\pm {100 * row[std]:.3f}"
            if np.isclose(float(row[mean]), maximum, atol=1e-12, rtol=0.0):
                value = rf"\mathbf{{{value}}}"
            cells.append(f"${value}$")
        lines.append(f"{DISPLAY[dataset]} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def markdown_table(summary: pd.DataFrame) -> str:
    indexed = summary.set_index("dataset")
    lines = [
        "# Table 3: Fixed 1% R-GCN Pruning",
        "",
        "Test accuracy (%), mean +/- sample standard deviation over five training seeds after averaging each seed across five splits.",
        "",
        "| Dataset | Full graph | Operational-RPI | Global random |",
        "|---|---:|---:|---:|",
    ]
    columns = [
        ("full_mean", "full_std"),
        ("rpi_mean", "rpi_std"),
        ("global_random_mean", "global_random_std"),
    ]
    for dataset in DATASET_ORDER:
        if dataset not in indexed.index:
            continue
        row = indexed.loc[dataset]
        maximum = max(float(row[mean]) for mean, _ in columns)
        cells = []
        for mean, std in columns:
            value = f"{100 * row[mean]:.3f} +/- {100 * row[std]:.3f}"
            if np.isclose(float(row[mean]), maximum, atol=1e-12, rtol=0.0):
                value = f"**{value}**"
            cells.append(value)
        lines.append(f"| {DISPLAY[dataset]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Table 3 from fixed-1% R-GCN runs."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, seeds, summary = build_tables(
        args.input.resolve(), args.allow_incomplete
    )
    cases.to_csv(output_dir / "paired_case_results.csv", index=False)
    seeds.to_csv(output_dir / "seed_level_results.csv", index=False)
    summary.to_csv(output_dir / "table_3_summary.csv", index=False)
    (output_dir / "table_3.tex").write_text(latex_table(summary))
    (output_dir / "table_3.md").write_text(markdown_table(summary))
    report = {
        "protocol": "table_3_fixed_1pct_reverse_paired",
        "datasets": int(len(summary)),
        "rpi_strict_best": int(summary["rpi_strict_best"].sum()),
        "rpi_co_best": int(summary["rpi_co_best"].sum()),
        "rpi_wins_vs_full": int((summary["rpi_minus_full_pp"] > 0).sum()),
        "rpi_ties_vs_full": int(
            np.isclose(summary["rpi_minus_full_pp"], 0.0, atol=1e-12).sum()
        ),
        "rpi_wins_vs_global_random": int(
            (summary["rpi_minus_global_random_pp"] > 0).sum()
        ),
        "mean_rpi_minus_full_pp": float(summary["rpi_minus_full_pp"].mean()),
        "mean_rpi_minus_global_random_pp": float(
            summary["rpi_minus_global_random_pp"].mean()
        ),
    }
    (output_dir / "table_3_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
