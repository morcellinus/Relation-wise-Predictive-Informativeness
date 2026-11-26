from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hcsbm.evaluation import extract_simpleheterosage_embeddings
from hcsbm.generator import generate_hcsbm
from hcsbm.yaml_utils import build_sweep_config_from_yaml, load_yaml_payload
from simpleheterosage.models.relation.simpleheterosage import SimpleHeteroSAGE
from simpleheterosage.trainers.node_cls_trainer import NodeClassificationTrainer, set_seed


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "results" / "hcsbm_final_experiments"
DEFAULT_DEGREE = 6
TAUS = [i / 10.0 for i in range(11)]
PCA_TAUS = [0.0, 1.0]

AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14
LEGEND_SIZE = 14
MARKER_SIZE = 9
BASELINE_MARKER_SIZE = 6
TITLE_SIZE = 18
PCA_POINT_SIZE = 13
PCA_LABEL_SIZE = 12
PCA_TAU_LABEL_SIZE = 12
PCA_GROUP_LABEL_SIZE = 12
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

LABELSPACE_CASES = [
    {
        "suffix": "ks_eq_kt_4_4_m6",
        "source_classes": 4,
        "target_classes": 4,
        "label": r"$\mathbf{K}_{\mathbf{s}} = \mathbf{K}_{\mathbf{t}}$",
        "plain": "K_s = K_t",
        "color": "#2b8cbe",
        "marker": "o",
        "rq1_name": "rq1_ks_eq_kt_4_4_m6",
    },
    {
        "suffix": "ks_lt_kt_2_4_m6",
        "source_classes": 2,
        "target_classes": 4,
        "label": r"$\mathbf{K}_{\mathbf{s}} < \mathbf{K}_{\mathbf{t}}$",
        "plain": "K_s < K_t",
        "color": "#d95f0e",
        "marker": "s",
        "rq1_name": "rq1_ks_lt_kt_2_4_m6",
    },
    {
        "suffix": "ks_gt_kt_8_4_m6",
        "source_classes": 8,
        "target_classes": 4,
        "label": r"$\mathbf{K}_{\mathbf{s}} > \mathbf{K}_{\mathbf{t}}$",
        "plain": "K_s > K_t",
        "color": "#1b9e77",
        "marker": "^",
        "rq1_name": "rq1_ks_gt_kt_8_4_m6",
    },
]

RQ2_VARIANTS = [
    {
        "key": "reinforcement",
        "title": "Reinforcement",
        "description": "all incident relations jointly swept",
        "relation_tau": {"A_to_P": "sweep", "C_to_P": "sweep", "T_to_P": "sweep"},
    },
    {
        "key": "dilution",
        "title": "Dilution",
        "description": "only A-P swept, C-P and T-P noisy",
        "relation_tau": {"A_to_P": "sweep", "C_to_P": 0.0, "T_to_P": 0.0},
    },
]


def uniform_prior(num_classes: int) -> list[float]:
    return [1.0 / float(num_classes)] * int(num_classes)


def node(num_classes: int) -> dict:
    return {
        "num_nodes": 1000,
        "num_classes": int(num_classes),
        "class_prior": uniform_prior(num_classes),
        "gamma": 1.0,
        "sigma": 1.0,
    }


def case_name(variant: dict, case: dict) -> str:
    return f"rq2_{variant['key']}_{case['suffix']}"


def make_payload(variant: dict, case: dict, degree: int) -> dict:
    src_classes = int(case["source_classes"])
    dst_classes = int(case["target_classes"])
    return {
        "name": case_name(variant, case),
        "description": (
            f"Final RQ2 {variant['key']}: {variant['description']}, "
            f"sources {src_classes}->{dst_classes} classes, m_r={degree}."
        ),
        "feature_dim": 64,
        "allow_multi_edges": False,
        "add_reverse_relations": False,
        "edge_set_mode": "shared_undirected",
        "target_types": ["P"],
        "supervision_types": ["P"],
        "split": {
            "target_type": "P",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "seed": 0,
        },
        "node_types": {
            "A": node(src_classes),
            "C": node(src_classes),
            "T": node(src_classes),
            "P": node(dst_classes),
        },
        "relations": {
            "A_to_P": {"source_type": "A", "destination_type": "P", "out_degree": int(degree)},
            "C_to_P": {"source_type": "C", "destination_type": "P", "out_degree": int(degree)},
            "T_to_P": {"source_type": "T", "destination_type": "P", "out_degree": int(degree)},
        },
        "sweep": {"relation_tau": variant["relation_tau"]},
    }


def ensure_layout(output_root: Path) -> None:
    for subdir in ["RQ1", "RQ2", "RQ3"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def write_configs(output_root: Path, degree: int) -> dict[tuple[str, str], Path]:
    config_dir = output_root / "RQ2" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in RQ2_VARIANTS:
        for case in LABELSPACE_CASES:
            path = config_dir / f"{case_name(variant, case)}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(make_payload(variant, case, degree), f, sort_keys=False)
            paths[(variant["key"], case["suffix"])] = path
    return paths


def summary_path(output_root: Path, variant: dict, case: dict) -> Path:
    name = case_name(variant, case)
    return output_root / "RQ2" / name / "summary.csv"


def rq1_summary_path(output_root: Path, case: dict) -> Path:
    return output_root / "RQ1" / case["rq1_name"] / "summary.csv"


def run_tau_sweeps(args: argparse.Namespace, config_paths: dict[tuple[str, str], Path]) -> None:
    runner = ROOT / "run_simpleheterosage_tau_sweep.py"
    for variant in RQ2_VARIANTS:
        for case in LABELSPACE_CASES:
            path = summary_path(args.output_root, variant, case)
            if args.resume and path.exists():
                print(f"[skip] existing summary: {path}", flush=True)
                continue
            config_path = config_paths[(variant["key"], case["suffix"])]
            cmd = [
                args.python_bin,
                str(runner),
                "--config-yaml",
                str(config_path),
                "--output-dir",
                str(args.output_root / "RQ2"),
                "--gammas",
                "1.0",
                "--taus",
                *[f"{tau:.1f}" for tau in TAUS],
                "--num-seeds",
                str(args.num_seeds),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--hidden",
                str(args.hidden),
                "--layers",
                "1",
                "--dropout",
                str(args.dropout),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--supervision-mode",
                "per_target",
                "--linear-probe",
                "--fisher-ratio",
                "--i-avg-scope",
                "destination",
                "--title",
                f"HCSBM Final RQ2 {variant['key']} {case['plain']}",
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            print(f"[run] RQ2 {variant['key']} {case['plain']} ({case['source_classes']}->{case['target_classes']})", flush=True)
            subprocess.run(cmd, check=True)


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return pd.read_csv(path)


def p_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["target_type"] == "P"].sort_values("tau")


def plot_curve(
    ax,
    df: pd.DataFrame,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
    markersize: int,
    shade: bool = True,
) -> None:
    sub = p_rows(df)
    xs = sub["tau"].to_numpy(dtype=float)
    ys = sub["accuracy_mean"].to_numpy(dtype=float)
    std = sub["accuracy_std"].to_numpy(dtype=float) if "accuracy_std" in sub else None
    ax.plot(
        xs,
        ys,
        marker=marker,
        markersize=markersize,
        color=color,
        linewidth=2.4 if linestyle == "-" else 2.0,
        linestyle=linestyle,
        label=label,
    )
    if shade and std is not None:
        ax.fill_between(xs, ys - std, ys + std, color=color, alpha=0.10)


def style_ax(ax, *, ylabel: str | None, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
    ax.set_xlabel(r"$\tau$", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel("Accuracy", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="best",
        frameon=False,
        prop={"weight": "bold", "size": LEGEND_SIZE},
        handlelength=2.2,
    )


def plot_overlay(output_root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0), squeeze=False)
    for j, variant in enumerate(RQ2_VARIANTS):
        ax = axes[0, j]
        for case in LABELSPACE_CASES:
            plot_curve(
                ax,
                load_summary(summary_path(output_root, variant, case)),
                label=rf"Acc, {case['label']}",
                color=case["color"],
                marker=case["marker"],
                linestyle="-",
                markersize=MARKER_SIZE,
            )
        for case in LABELSPACE_CASES:
            plot_curve(
                ax,
                load_summary(rq1_summary_path(output_root, case)),
                label=rf"RQ1, {case['label']}",
                color=case["color"],
                marker=case["marker"],
                linestyle=":",
                markersize=BASELINE_MARKER_SIZE,
                shade=False,
            )
        style_ax(ax, ylabel="Accuracy" if j == 0 else None, title=variant["title"])

    fig.subplots_adjust(left=0.0867, right=0.9609, bottom=0.1466, top=0.9398, wspace=0.160)
    out_path = output_root / "RQ2" / "rq2_p_accuracy_overlay_v6"
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def trainer_config(*, args: argparse.Namespace) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "hidden_channels": args.hidden,
        "num_layers": 1,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.pca_epochs,
        "patience": args.pca_patience,
        "eval_every": 1,
        "device": device,
    }


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.size(1) < 2:
        return torch.cat([x, torch.zeros(x.size(0), 2 - x.size(1))], dim=1)
    q = min(8, x.size(0), x.size(1))
    try:
        _, _, v = torch.pca_lowrank(x, q=q, center=False)
        return x @ v[:, :2]
    except Exception:
        _, _, v = torch.linalg.svd(x, full_matrices=False)
        return x @ v[:2, :].T


def train_and_project(config_path: Path, *, tau: float, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    payload = load_yaml_payload(config_path)
    cfg = build_sweep_config_from_yaml(payload, gamma=1.0, tau=float(tau), seed=0, target_type="P", split_seed=0)
    data, _ = generate_hcsbm(cfg)
    set_seed(0)
    num_classes = int(data["P"].y.max().item() + 1)
    model = SimpleHeteroSAGE.from_data(
        data,
        target_type="P",
        num_classes=num_classes,
        hidden_channels=args.hidden,
        num_layers=1,
        dropout=args.dropout,
    )
    trainer = NodeClassificationTrainer(model, data, "P", trainer_config(args=args))
    _ = trainer.run()
    embeddings = extract_simpleheterosage_embeddings(trainer.model, trainer.data)
    return pca_2d(embeddings["P"]), trainer.data["P"].y.detach().cpu()


def scatter_panel(ax, coords: torch.Tensor, labels: torch.Tensor) -> None:
    labels = labels.to(torch.long)
    for cls in torch.unique(labels).tolist():
        mask = labels == cls
        pts = coords[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=PCA_POINT_SIZE,
            alpha=0.70,
            c=PALETTE[int(cls) % len(PALETTE)],
            edgecolors="none",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.12)


def plot_pca(output_root: Path, config_paths: dict[tuple[str, str], Path], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(
        len(LABELSPACE_CASES),
        len(RQ2_VARIANTS) * len(PCA_TAUS),
        figsize=(20.0 / 3.0, 5.0),
        squeeze=False,
    )
    for i, case in enumerate(LABELSPACE_CASES):
        col = 0
        for variant in RQ2_VARIANTS:
            config_path = config_paths[(variant["key"], case["suffix"])]
            for tau in PCA_TAUS:
                coords, labels = train_and_project(config_path, tau=tau, args=args)
                scatter_panel(axes[i, col], coords, labels)
                col += 1

    fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.14, hspace=0.52, wspace=0.06)

    for i, case in enumerate(LABELSPACE_CASES):
        left = axes[i, 0].get_position()
        right = axes[i, -1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            max(left.y1, right.y1) + 0.010,
            case["label"],
            ha="center",
            va="bottom",
            fontsize=PCA_LABEL_SIZE,
            fontweight="bold",
        )

    top_axes = axes[0]
    for start_col, variant in zip([0, 2], RQ2_VARIANTS):
        left = top_axes[start_col].get_position()
        right = top_axes[start_col + 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            0.920,
            variant["title"],
            ha="center",
            va="bottom",
            fontsize=PCA_GROUP_LABEL_SIZE,
            fontweight="bold",
        )

    bottom_axes = axes[-1]
    for col, tau in enumerate(PCA_TAUS * len(RQ2_VARIANTS)):
        pos = bottom_axes[col].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y0 - 0.035,
            rf"$\tau = {tau:.1f}$",
            ha="center",
            va="top",
            fontsize=PCA_TAU_LABEL_SIZE,
            fontweight="bold",
        )

    out_path = output_root / "RQ2" / "rq2_p_pca_overview"
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

    for variant in RQ2_VARIANTS:
        plot_pca_variant(output_root, config_paths, args, variant)


def plot_pca_variant(
    output_root: Path,
    config_paths: dict[tuple[str, str], Path],
    args: argparse.Namespace,
    variant: dict,
) -> None:
    fig, axes = plt.subplots(len(LABELSPACE_CASES), len(PCA_TAUS), figsize=(10.0 / 3.0, 5.0), squeeze=False)
    for i, case in enumerate(LABELSPACE_CASES):
        config_path = config_paths[(variant["key"], case["suffix"])]
        for j, tau in enumerate(PCA_TAUS):
            coords, labels = train_and_project(config_path, tau=tau, args=args)
            scatter_panel(axes[i, j], coords, labels)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.14, hspace=0.52, wspace=0.03)

    fig.text(
        0.5,
        0.920,
        variant["title"],
        ha="center",
        va="bottom",
        fontsize=PCA_GROUP_LABEL_SIZE,
        fontweight="bold",
    )

    for i, case in enumerate(LABELSPACE_CASES):
        left = axes[i, 0].get_position()
        right = axes[i, -1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            max(left.y1, right.y1) + 0.010,
            case["label"],
            ha="center",
            va="bottom",
            fontsize=PCA_LABEL_SIZE,
            fontweight="bold",
        )

    bottom_axes = axes[-1]
    for j, tau in enumerate(PCA_TAUS):
        pos = bottom_axes[j].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y0 - 0.035,
            rf"$\tau = {tau:.1f}$",
            ha="center",
            va="top",
            fontsize=PCA_TAU_LABEL_SIZE,
            fontweight="bold",
        )

    out_path = output_root / "RQ2" / f"rq2_p_pca_{variant['key']}"
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_metadata(output_root: Path, config_paths: dict[tuple[str, str], Path], args: argparse.Namespace) -> None:
    payload = {
        "degree_m_r": args.degree,
        "labelspace_cases": [
            {
                "label": case["plain"],
                "source_classes": case["source_classes"],
                "target_classes": case["target_classes"],
            }
            for case in LABELSPACE_CASES
        ],
        "variants": [variant["key"] for variant in RQ2_VARIANTS],
        "configs": {f"{key[0]}::{key[1]}": str(path) for key, path in config_paths.items()},
        "taus": TAUS,
        "pca_taus": PCA_TAUS,
        "accuracy_overlay": str(output_root / "RQ2" / "rq2_p_accuracy_overlay_v6.png"),
        "pca_plot": str(output_root / "RQ2" / "rq2_p_pca_overview.png"),
        "split_pca_plots": {
            variant["key"]: str(output_root / "RQ2" / f"rq2_p_pca_{variant['key']}.png")
            for variant in RQ2_VARIANTS
        },
    }
    (output_root / "RQ2" / "run_metadata.json").write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--pca-epochs", type=int, default=200)
    ap.add_argument("--pca-patience", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    ensure_layout(args.output_root)
    config_paths = write_configs(args.output_root, args.degree)
    run_tau_sweeps(args, config_paths)
    plot_overlay(args.output_root)
    plot_pca(args.output_root, config_paths, args)
    write_metadata(args.output_root, config_paths, args)
    print(f"[done] {args.output_root / 'RQ2'}", flush=True)


if __name__ == "__main__":
    main()
