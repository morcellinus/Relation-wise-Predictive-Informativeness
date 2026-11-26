from __future__ import annotations

import argparse
import json
import math
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
TARGET_CLASSES = 4

AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14
LEGEND_SIZE = 14
MARKER_SIZE = 9
RPI_MARKER_SIZE = 6
PCA_POINT_SIZE = 13
PCA_LABEL_SIZE = 12
PCA_TAU_LABEL_SIZE = 12
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

CASES = [
    {
        "name": "rq1_ks_eq_kt_4_4_m6",
        "source_classes": 4,
        "target_classes": 4,
        "label": r"$\mathbf{K}_{\mathbf{s}} = \mathbf{K}_{\mathbf{t}}$",
        "plain": "K_s = K_t",
        "color": "#2b8cbe",
        "marker": "o",
    },
    {
        "name": "rq1_ks_lt_kt_2_4_m6",
        "source_classes": 2,
        "target_classes": 4,
        "label": r"$\mathbf{K}_{\mathbf{s}} < \mathbf{K}_{\mathbf{t}}$",
        "plain": "K_s < K_t",
        "color": "#d95f0e",
        "marker": "s",
    },
    {
        "name": "rq1_ks_gt_kt_8_4_m6",
        "source_classes": 8,
        "target_classes": 4,
        "label": r"$\mathbf{K}_{\mathbf{s}} > \mathbf{K}_{\mathbf{t}}$",
        "plain": "K_s > K_t",
        "color": "#1b9e77",
        "marker": "^",
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


def make_payload(case: dict, degree: int) -> dict:
    return {
        "name": case["name"],
        "description": (
            f"Final RQ1: A-P relation, {case['plain']}, "
            f"{case['source_classes']}->{case['target_classes']} classes, m_r={degree}."
        ),
        "feature_dim": 64,
        "allow_multi_edges": False,
        "add_reverse_relations": False,
        "edge_set_mode": "shared_undirected",
        "target_types": ["A", "P"],
        "supervision_types": ["A", "P"],
        "split": {
            "target_type": "A",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "seed": 0,
        },
        "node_types": {
            "A": node(case["source_classes"]),
            "P": node(case["target_classes"]),
        },
        "relations": {
            "A_to_P": {
                "source_type": "A",
                "destination_type": "P",
                "out_degree": int(degree),
            }
        },
        "sweep": {"relation_tau": {"A_to_P": "sweep"}},
    }


def ensure_layout(output_root: Path) -> None:
    for subdir in ["RQ1", "RQ2", "RQ3"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def write_configs(output_root: Path, degree: int) -> list[Path]:
    config_dir = output_root / "RQ1" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in CASES:
        path = config_dir / f"{case['name']}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(make_payload(case, degree), f, sort_keys=False)
        paths.append(path)
    return paths


def summary_path(output_root: Path, case: dict) -> Path:
    return output_root / "RQ1" / case["name"] / "summary.csv"


def run_tau_sweeps(args: argparse.Namespace, config_paths: Sequence[Path]) -> None:
    runner = ROOT / "run_simpleheterosage_tau_sweep.py"
    for case, config_path in zip(CASES, config_paths):
        if args.resume and summary_path(args.output_root, case).exists():
            print(f"[skip] existing summary: {summary_path(args.output_root, case)}", flush=True)
            continue
        cmd = [
            args.python_bin,
            str(runner),
            "--config-yaml",
            str(config_path),
            "--output-dir",
            str(args.output_root / "RQ1"),
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
            "multi_type",
            "--linear-probe",
            "--fisher-ratio",
            "--i-avg-scope",
            "destination",
            "--title",
            f"HCSBM Final RQ1 {case['plain']}",
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        print(f"[run] RQ1 {case['plain']} ({case['source_classes']}->{case['target_classes']})", flush=True)
        subprocess.run(cmd, check=True)


def load_summary(output_root: Path, case: dict) -> pd.DataFrame:
    path = summary_path(output_root, case)
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return pd.read_csv(path)


def plot_overlay(output_root: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.0))
    ax2 = ax.twinx()
    for case in CASES:
        df = load_summary(output_root, case)
        sub = df[df["target_type"] == "P"].sort_values("tau")
        xs = sub["tau"].to_numpy(dtype=float)
        acc = sub["accuracy_mean"].to_numpy(dtype=float)
        acc_std = sub["accuracy_std"].to_numpy(dtype=float)
        rpi = sub["I_avg_destination_theoretical_nats"].to_numpy(dtype=float) / math.log(TARGET_CLASSES)

        ax.plot(
            xs,
            acc,
            marker=case["marker"],
            markersize=MARKER_SIZE,
            color=case["color"],
            linewidth=2.4,
            linestyle="-",
            label=rf"Acc, {case['label']}",
        )
        ax.fill_between(xs, acc - acc_std, acc + acc_std, color=case["color"], alpha=0.10)
        ax2.plot(
            xs,
            rpi,
            marker=case["marker"],
            markersize=RPI_MARKER_SIZE,
            color=case["color"],
            linewidth=2.0,
            linestyle=":",
            label=f"RPI, {case['label']}",
        )

    ax.set_xlabel(r"$\tau$", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax2.set_ylabel("RPI", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="best",
        ncol=1,
        frameon=False,
        prop={"weight": "bold", "size": LEGEND_SIZE},
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.1580, right=0.8417, bottom=0.1466, top=0.9398)
    out_path = output_root / "RQ1" / "rq1_accuracy_rpi_overlay"
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


def plot_pca(output_root: Path, config_paths: Sequence[Path], args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(len(CASES), len(PCA_TAUS), figsize=(10.0 / 3.0, 5.0), squeeze=False)
    for i, (case, config_path) in enumerate(zip(CASES, config_paths)):
        for j, tau in enumerate(PCA_TAUS):
            coords, labels = train_and_project(config_path, tau=tau, args=args)
            scatter_panel(axes[i, j], coords, labels)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.14, hspace=0.52, wspace=0.03)

    for i, case in enumerate(CASES):
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

    out_path = output_root / "RQ1" / "rq1_p_pca_overview"
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_metadata(output_root: Path, config_paths: Sequence[Path], args: argparse.Namespace) -> None:
    payload = {
        "degree_m_r": args.degree,
        "cases": [
            {
                "name": case["name"],
                "label": case["plain"],
                "source_classes": case["source_classes"],
                "target_classes": case["target_classes"],
                "summary": str(summary_path(output_root, case)),
            }
            for case in CASES
        ],
        "configs": [str(path) for path in config_paths],
        "taus": TAUS,
        "pca_taus": PCA_TAUS,
        "accuracy_rpi_overlay": str(output_root / "RQ1" / "rq1_accuracy_rpi_overlay.png"),
        "pca_plot": str(output_root / "RQ1" / "rq1_p_pca_overview.png"),
    }
    (output_root / "RQ1" / "run_metadata.json").write_text(json.dumps(payload, indent=2))


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
    print(f"[done] {args.output_root / 'RQ1'}", flush=True)


if __name__ == "__main__":
    main()
