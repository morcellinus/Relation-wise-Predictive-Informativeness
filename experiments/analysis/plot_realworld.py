from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"rpi": "#8c6bb1", "mhr": "#f28e6b", "chr": "#63c29a"}
GROUPS = [
    ("Academic", ["acm", "aminer", "dblp"], "#4c78a8"),
    ("Movie", ["freebase", "imdb"], "#f58518"),
    ("Knowledge graphs", ["aifb", "am", "bgs", "mutag"], "#54a24b"),
    ("Others", ["liar", "yelp"], "#b279a2"),
]


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def relation_frame(input_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_dir / "metrics" / "rpi_relation.csv")
    frame = frame[
        (frame["pseudo_label_method"] == "hard_lp")
        & (frame["metric_scope"] == "full_relations")
    ]
    return (
        frame.groupby(["dataset", "relation"], as_index=False)["RPI"]
        .mean()
        .dropna(subset=["RPI"])
    )


def metric_frame(input_dir: Path) -> pd.DataFrame:
    rpi = pd.read_csv(input_dir / "metrics" / "rpi_target.csv")
    mhr = pd.read_csv(input_dir / "metrics" / "mhr_dataset.csv")
    chr_frame = pd.read_csv(input_dir / "metrics" / "chr_dataset.csv")
    rpi = rpi.groupby("dataset")["RPI_type"].agg(["mean", "std"]).rename(columns={"mean": "RPI", "std": "RPI_std"})
    mhr = mhr.groupby("dataset")["MHR_mean"].agg(["mean", "std"]).rename(columns={"mean": "MHR", "std": "MHR_std"})
    chr_frame = chr_frame.groupby("dataset")["CTH_CHR"].agg(["mean", "std"]).rename(columns={"mean": "CHR", "std": "CHR_std"})
    return pd.concat([rpi, mhr, chr_frame], axis=1).dropna()


def plot_relation_distribution(input_dir: Path, output_dir: Path, dpi: int) -> None:
    frame = relation_frame(input_dir)
    available = set(frame["dataset"])
    order: list[str] = []
    colors: dict[str, str] = {}
    spans: list[tuple[str, int, int, str]] = []
    for group, datasets, color in GROUPS:
        start = len(order)
        for dataset in datasets:
            if dataset in available:
                order.append(dataset)
                colors[dataset] = color
        if len(order) > start:
            spans.append((group, start, len(order) - 1, color))
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(8.2, 3.45))
    for x, dataset in enumerate(order):
        values = frame.loc[frame["dataset"] == dataset, "RPI"].to_numpy(float)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.vlines(x, values.min(), values.max(), color="#9c9c9c", linewidth=2, alpha=0.75, zorder=1)
        ax.vlines(x, q1, q3, color=colors[dataset], linewidth=11, alpha=0.78, zorder=2)
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=40,
            color=colors[dataset],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(x, median, marker="D", s=120, facecolor="white", edgecolor="black", linewidth=1.5, zorder=4)
        ax.text(x, 1.035, f"n={len(values)}", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#555555")
    for group, start, end, color in spans:
        if start:
            ax.axvline(start - 0.5, color="#bdbdbd", linewidth=1.1, alpha=0.72)
        ax.text((start + end) / 2, 1.13, group, ha="center", va="bottom", fontsize=10.6, fontweight="bold", color=color)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90, ha="center", va="top", fontsize=15, fontweight="bold")
    ax.set_ylabel("Relation-level RPI", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.03, 1.09)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#cfcfcf", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, output_dir, "figure_6", dpi)


def plot_dataset_metrics(input_dir: Path, output_dir: Path, dpi: int) -> None:
    frame = metric_frame(input_dir).sort_values("RPI", ascending=False)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(6.25, 3.0))
    handles = []
    for metric, offset, color, zorder in [
        ("RPI", -0.14, COLORS["rpi"], 6),
        ("MHR", 0.0, COLORS["mhr"], 4),
        ("CHR", 0.14, COLORS["chr"], 4),
    ]:
        handles.append(
            ax.errorbar(
                x + offset,
                frame[metric],
                yerr=frame[f"{metric}_std"],
                fmt="o",
                markersize=9.5,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                capsize=2.3,
                capthick=1.0,
                elinewidth=1.05,
                zorder=zorder,
            )
        )
    ax.set_xticks(x)
    ax.set_xticklabels(frame.index, rotation=90, ha="center", va="top", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.19)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.42, len(frame) - 0.58)
    ax.grid(alpha=0.16)
    ax.legend(
        handles,
        ["RPI(t)", "MHR", "CHR"],
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.07),
        frameon=True,
        prop={"weight": "bold", "size": 9.8},
    )
    save_figure(fig, output_dir, "figure_7", dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    plot_relation_distribution(args.input_dir.resolve(), args.output_dir.resolve(), args.dpi)
    plot_dataset_metrics(args.input_dir.resolve(), args.output_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
