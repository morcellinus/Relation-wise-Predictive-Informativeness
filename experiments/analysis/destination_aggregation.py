from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realworld_alignment.run_backbone_alignment import load_dataset


RULES = [
    ("uniform_macro", "Uniform"),
    ("relation_edge_count", "Edge count"),
    ("target_node_coverage", "Target coverage"),
    ("mean_incoming_degree", "Mean degree"),
]
BACKBONES = ["han", "hetgnn", "rgcn", "rshn", "simplehgn"]
DISPLAY = {"han": "HAN", "hetgnn": "HetGNN", "rgcn": "R-GCN", "rshn": "RSHN", "simplehgn": "Simple-HGN"}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def structural_weights(dataset: str, data, target: str) -> dict[str, dict[str, float]]:
    rows = {}
    for edge_type in data.edge_types:
        edge_type = tuple(edge_type)
        if edge_type[2] != target:
            continue
        edge_index = data[edge_type].edge_index.cpu().numpy()
        edge_count = float(edge_index.shape[1])
        coverage = float(np.unique(edge_index[1]).size)
        rows[f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}"] = {
            "edge_count_weight": edge_count,
            "target_coverage_weight": coverage,
            "mean_incoming_degree_weight": edge_count / coverage if coverage > 0 else float("nan"),
        }
    return rows


def build_scores(input_dir: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation = pd.read_csv(input_dir / "metrics" / "rpi_relation.csv")
    targets = pd.read_csv(input_dir / "metrics" / "rpi_target.csv")[["dataset", "split", "node_type"]]
    relation = relation[
        (relation["pseudo_label_method"] == "hard_lp")
        & (relation["metric_scope"] == "full_relations")
        & relation["RPI"].notna()
    ].merge(targets, on=["dataset", "split"], how="inner")
    relation = relation[relation["dst_type"] == relation["node_type"]].copy()
    audit_rows = []
    for dataset in sorted(relation["dataset"].unique()):
        data, target = load_dataset(dataset, data_dir)
        weights = structural_weights(dataset, data, target)
        for name, values in weights.items():
            audit_rows.append({"dataset": dataset, "relation": name, **values})
    audit = pd.DataFrame(audit_rows)
    relation = relation.merge(audit, on=["dataset", "relation"], how="left")
    score_rows = []
    for (dataset, split), group in relation.groupby(["dataset", "split"]):
        values = group["RPI"].to_numpy(float)
        rule_values = {
            "uniform_macro": float(values.mean()),
            "relation_edge_count": weighted_mean(values, group["edge_count_weight"].to_numpy(float)),
            "target_node_coverage": weighted_mean(values, group["target_coverage_weight"].to_numpy(float)),
            "mean_incoming_degree": weighted_mean(values, group["mean_incoming_degree_weight"].to_numpy(float)),
        }
        for rule, value in rule_values.items():
            score_rows.append(
                {
                    "dataset": dataset,
                    "split": int(split),
                    "weight_rule": rule,
                    "RPI_target_aggregated": value,
                    "num_incoming_relations": len(group),
                }
            )
    return pd.DataFrame(score_rows), audit


def correlations(input_dir: Path, scores: pd.DataFrame) -> pd.DataFrame:
    alignment = pd.read_csv(input_dir / "results" / "alignment_backbone_level.csv")
    alignment = alignment[alignment["backbone"].isin(BACKBONES)]
    split_rows = alignment.merge(scores, on=["dataset", "split"], how="inner")
    dataset_rows = split_rows.groupby(["dataset", "backbone", "weight_rule"], as_index=False).agg(
        RPI_target_aggregated=("RPI_target_aggregated", "mean"),
        delta_hgnn=("delta_hgnn", "mean"),
        splits=("split", "nunique"),
    )
    rows = []
    for (rule, backbone), group in dataset_rows.groupby(["weight_rule", "backbone"]):
        x = group["RPI_target_aggregated"].to_numpy(float)
        y = group["delta_hgnn"].to_numpy(float)
        if len(group) >= 2 and np.std(x) > 1e-12:
            prediction = np.polyval(np.polyfit(x, y, 1), x)
            ss_res = float(np.sum((y - prediction) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rho = float(stats.spearmanr(x, y).statistic)
        else:
            r2 = rho = float("nan")
        rows.append(
            {
                "weight_rule": rule,
                "backbone": backbone,
                "n_datasets": len(group),
                "spearman_rho": rho,
                "linear_r2": r2,
            }
        )
    return dataset_rows, pd.DataFrame(rows)


def short(value: float) -> str:
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0") else "-" + text[2:] if text.startswith("-0") else text


def latex_table(frame: pd.DataFrame) -> str:
    backbones = [backbone for backbone in BACKBONES if backbone in set(frame["backbone"])]
    best_rho = frame.groupby("backbone")["spearman_rho"].max().to_dict()
    best_r2 = frame.groupby("backbone")["linear_r2"].max().to_dict()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robustness to destination-type aggregation. Each cell reports Spearman correlation $\rho$ and linear $R^2$ between destination-type operational RPI and HGNN gain across the 11 datasets. Bold indicates the best result.}",
        r"\label{tab:destination_aggregation}",
        r"\scriptsize",
        r"\begin{tabular}{l" + "c" * len(backbones) + "}",
        r"\toprule",
        "Aggregation & " + " & ".join(DISPLAY[b] for b in backbones) + r" \\",
        r"\midrule",
    ]
    for rule, label in RULES:
        cells = []
        for backbone in backbones:
            row = frame[(frame["weight_rule"] == rule) & (frame["backbone"] == backbone)].iloc[0]
            rho = short(float(row["spearman_rho"]))
            r2 = short(float(row["linear_r2"]))
            if np.isclose(row["spearman_rho"], best_rho[backbone]):
                rho = rf"\mathbf{{{rho}}}"
            if np.isclose(row["linear_r2"], best_r2[backbone]):
                r2 = rf"\mathbf{{{r2}}}"
            cells.append(rf"${rho}/{r2}$")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, audit = build_scores(args.input_dir.resolve(), args.data_dir.resolve())
    dataset_rows, correlation_rows = correlations(args.input_dir.resolve(), scores)
    scores.to_csv(output_dir / "destination_aggregation_split_scores.csv", index=False)
    audit.to_csv(output_dir / "destination_aggregation_structural_weights.csv", index=False)
    dataset_rows.to_csv(output_dir / "destination_aggregation_dataset_scores.csv", index=False)
    correlation_rows.to_csv(output_dir / "destination_aggregation_correlations.csv", index=False)
    (output_dir / "table_2.tex").write_text(latex_table(correlation_rows))


if __name__ == "__main__":
    main()
