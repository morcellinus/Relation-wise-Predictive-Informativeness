from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BACKBONES = ["han", "hetgnn", "rgcn", "rshn", "simplehgn"]
METRICS = [
    ("MHR_mean", "MHR", "#fc8d62"),
    ("CTH_CHR", "CHR", "#66c2a5"),
    ("RPI_target", "RPI(t)", "#8e6bb7"),
]
LABELS = {
    ("han", "MHR_mean"): {"aifb", "bgs", "liar"},
    ("han", "CTH_CHR"): {"aifb", "aminer", "liar"},
    ("han", "RPI_target"): {"aifb", "aminer", "yelp"},
    ("hetgnn", "MHR_mean"): {"bgs", "imdb", "liar"},
    ("hetgnn", "CTH_CHR"): {"aminer", "freebase", "liar"},
    ("hetgnn", "RPI_target"): {"aminer", "freebase", "yelp"},
    ("rgcn", "MHR_mean"): {"aminer", "liar", "yelp"},
    ("rgcn", "CTH_CHR"): {"aminer", "liar", "yelp"},
    ("rgcn", "RPI_target"): {"aminer", "freebase", "liar"},
    ("rshn", "MHR_mean"): {"aminer", "liar", "yelp"},
    ("rshn", "CTH_CHR"): {"aminer", "liar", "yelp"},
    ("rshn", "RPI_target"): {"aminer", "freebase", "yelp"},
    ("simplehgn", "MHR_mean"): {"aifb", "aminer", "liar"},
    ("simplehgn", "CTH_CHR"): {"aminer", "freebase", "liar"},
    ("simplehgn", "RPI_target"): {"aminer", "liar", "yelp"},
}


def build_frame(input_dir: Path) -> pd.DataFrame:
    performance = pd.read_csv(input_dir / "results" / "alignment_backbone_level.csv")
    alignment = pd.read_csv(input_dir / "results" / "alignment_dataset_level.csv")
    merged = performance.merge(
        alignment[["dataset", "split", "RPI_target", "MHR_mean", "CTH_CHR"]],
        on=["dataset", "split"],
        how="left",
    )
    merged = merged[merged["backbone"].isin(BACKBONES)]
    return merged.groupby(["dataset", "backbone"], as_index=False).agg(
        delta_hgnn=("delta_hgnn", "mean"),
        delta_hgnn_std=("delta_hgnn", "std"),
        RPI_target=("RPI_target", "mean"),
        RPI_target_std=("RPI_target", "std"),
        MHR_mean=("MHR_mean", "mean"),
        MHR_mean_std=("MHR_mean", "std"),
        CTH_CHR=("CTH_CHR", "mean"),
        CTH_CHR_std=("CTH_CHR", "std"),
        split_count=("split", "nunique"),
    )


def confidence_band(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x) < 3:
        return None
    x_mean = float(x.mean())
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 1e-12:
        return None
    fit = np.polyfit(x, y, 1)
    residual = y - np.polyval(fit, x)
    residual_se = math.sqrt(float(np.sum(residual**2)) / (len(x) - 2))
    margin = stats.t.ppf(0.975, len(x) - 2) * residual_se * np.sqrt(1 / len(x) + (grid - x_mean) ** 2 / sxx)
    pred = np.polyval(fit, grid)
    return pred - margin, pred + margin


def label_position(dataset: str, x: float, y: float, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float, str, str]:
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    dx, dy, ha, va = 0.035 * x_span, 0.025 * y_span, "left", "bottom"
    if x > xlim[0] + 0.72 * x_span:
        dx, ha = -0.035 * x_span, "right"
    if y > ylim[0] + 0.76 * y_span:
        dy, va = -0.035 * y_span, "top"
    if dataset in {"liar", "yelp"} and y < ylim[0] + 0.32 * y_span:
        dy = 0.045 * y_span
    return x + dx, y + dy, ha, va


def plot(frame: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if not (frame["split_count"] == 5).all():
        raise RuntimeError("Figure 8 requires five completed splits per dataset and backbone")
    fig, axes = plt.subplots(5, 3, figsize=(9.62, 9.10), dpi=dpi, squeeze=False)
    x_limits = {}
    for key, _, _ in METRICS:
        values = frame[key].to_numpy(float)
        std = frame[f"{key}_std"].to_numpy(float)
        bounds = np.concatenate([values - std, values + std])
        pad = max((bounds.max() - bounds.min()) * 0.055, 0.01)
        x_limits[key] = (float(bounds.min() - pad), float(bounds.max() + pad))
    y_limits = {}
    for backbone in BACKBONES:
        sub = frame[frame["backbone"] == backbone]
        values = sub["delta_hgnn"].to_numpy(float)
        std = sub["delta_hgnn_std"].to_numpy(float)
        bounds = np.concatenate([values - std, values + std])
        pad = max((bounds.max() - bounds.min()) * 0.09, 0.01)
        y_limits[backbone] = (float(bounds.min() - pad), float(bounds.max() + pad))
    for row, backbone in enumerate(BACKBONES):
        sub = frame[frame["backbone"] == backbone]
        for column, (key, title, color) in enumerate(METRICS):
            ax = axes[row, column]
            x = sub[key].to_numpy(float)
            y = sub["delta_hgnn"].to_numpy(float)
            grid = np.linspace(x.min(), x.max(), 240)
            fit = np.polyfit(x, y, 1)
            band = confidence_band(x, y, grid)
            if band is not None:
                ax.fill_between(grid, band[0], band[1], color="#8c8c8c", alpha=0.22, linewidth=0)
            ax.plot(grid, np.polyval(fit, grid), color="#111111", linewidth=2.6)
            ax.errorbar(
                x,
                y,
                xerr=sub[f"{key}_std"].to_numpy(float),
                yerr=sub["delta_hgnn_std"].to_numpy(float),
                fmt="none",
                ecolor=color,
                elinewidth=1.15,
                capsize=2.2,
                alpha=0.58,
            )
            ax.scatter(x, y, s=70, color=color, edgecolor="none", zorder=3)
            for item in sub.itertuples(index=False):
                dataset = str(item.dataset)
                if dataset not in LABELS[(backbone, key)]:
                    continue
                lx, ly, ha, va = label_position(dataset, float(getattr(item, key)), float(item.delta_hgnn), x_limits[key], y_limits[backbone])
                ax.text(lx, ly, dataset, fontsize=10.5, fontweight="bold", ha=ha, va=va)
            rho = stats.spearmanr(x, y).statistic
            prediction = np.polyval(fit, x)
            ss_res = float(np.sum((y - prediction) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot
            ax.text(
                0.045,
                0.945,
                f"rho={rho:.2f}\nR²={r2:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13.9,
                fontweight="bold",
                color="#8b0000" if key == "RPI_target" else "#111111",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": 0.92},
            )
            ax.axhline(0, color="#cfcfcf", linewidth=1.1)
            ax.grid(True, color="#d9d9d9", linewidth=0.65, alpha=0.55)
            ax.set_xlim(x_limits[key])
            ax.set_ylim(y_limits[backbone])
            if row == 0:
                ax.set_title(title, fontsize=18, fontweight="bold")
            if column == 0:
                ax.set_ylabel(backbone, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.06, wspace=0.175, hspace=0.205)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "figure_8_points_mean_std.csv", index=False)
    fig.savefig(output_dir / "figure_8.png", dpi=dpi)
    fig.savefig(output_dir / "figure_8.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    plot(build_frame(args.input_dir.resolve()), args.output_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
