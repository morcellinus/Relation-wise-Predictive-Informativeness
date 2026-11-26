from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def compose_figure_3(input_dir: Path, output_dir: Path, dpi: int) -> None:
    accuracy = mpimg.imread(require(input_dir / "RQ1" / "rq1_accuracy_rpi_overlay.png"))
    pca = mpimg.imread(require(input_dir / "RQ1" / "rq1_p_pca_overview.png"))
    top = int(pca.shape[0] * 0.115)
    bottom = int(pca.shape[0] * 0.315)
    first = pca[top:bottom, int(pca.shape[1] * 0.04) : int(pca.shape[1] * 0.515)]
    second = pca[top:bottom, int(pca.shape[1] * 0.525) : int(pca.shape[1] * 0.995)]
    fig = plt.figure(figsize=(8.2, 5.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 0.72], wspace=0.08)
    left = fig.add_subplot(grid[0, 0])
    right_grid = grid[0, 1].subgridspec(2, 1, hspace=0.08)
    upper = fig.add_subplot(right_grid[0, 0])
    lower = fig.add_subplot(right_grid[1, 0])
    left.imshow(accuracy)
    upper.imshow(first)
    lower.imshow(second)
    left.axis("off")
    upper.axis("off")
    lower.axis("off")
    upper.set_title(r"$K_s=K_t$", fontsize=13, fontweight="bold", pad=2)
    upper.text(-0.08, 0.5, r"$\tau=0.0$", transform=upper.transAxes, rotation=90, ha="center", va="center", fontsize=11)
    lower.text(-0.08, 0.5, r"$\tau=1.0$", transform=lower.transAxes, rotation=90, ha="center", va="center", fontsize=11)
    fig.text(0.33, 0.02, "(a) Accuracy and RPI", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.79, 0.02, "(b) Representation", ha="center", fontsize=15, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure_3.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "figure_3.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_figure(input_dir: Path, output_dir: Path, source: str, target: str) -> None:
    for suffix in [".png", ".pdf"]:
        shutil.copy2(require(input_dir / f"{source}{suffix}"), output_dir / f"{target}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compose_figure_3(input_dir, output_dir, args.dpi)
    copy_figure(input_dir, output_dir, "RQ2/rq2_p_accuracy_overlay_v6", "figure_4")
    copy_figure(input_dir, output_dir, "RQ3/rq3_p_accuracy_overlay_v6", "figure_5")


if __name__ == "__main__":
    main()
