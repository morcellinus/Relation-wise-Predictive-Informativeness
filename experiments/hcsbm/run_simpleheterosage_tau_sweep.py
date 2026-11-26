from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hcsbm.generator import generate_hcsbm
from hcsbm.scenarios import complete_directed_four_type_config
from hcsbm.yaml_utils import build_sweep_config_from_yaml, expand_yaml_paths, load_yaml_payload
from hcsbm.evaluation import (
    MultiTypeEvalConfig,
    attach_all_type_splits,
    evaluate_embeddings_by_type,
    extract_multitype_embeddings,
    train_multitype_supervised,
)
from simpleheterosage.models.relation.simpleheterosage import SimpleHeteroSAGE
from simpleheterosage.trainers.node_cls_trainer import NodeClassificationTrainer, set_seed


DEFAULT_GAMMAS = [0.5, 1.0, 2.0, 4.0]
DEFAULT_TAUS = [i / 10.0 for i in range(11)]
DEFAULT_TYPES = ["A", "B", "C", "D"]
DEFAULT_CLASS_COUNTS = [3, 4, 5, 4]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-yaml", nargs="+", help="YAML file(s) or directories of YAML HCSBM base configs")
    parser.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    parser.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAUS)
    parser.add_argument("--target-types", nargs="+")
    parser.add_argument("--type-names", nargs="+")
    parser.add_argument("--class-counts", nargs="+", type=int)
    parser.add_argument("--num-nodes", type=int, default=600)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--out-degree", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of averaged runs using consecutive seed offsets")
    parser.add_argument("--device", default=None)
    parser.add_argument("--title", default="SimpleHeteroSAGE Tau Sweep")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1", "micro_f1", "auc"])
    parser.add_argument(
        "--supervision-mode",
        default="per_target",
        choices=["per_target", "multi_type"],
        help="per_target keeps the old one-head-per-run behavior; multi_type trains one encoder with one head per node type",
    )
    parser.add_argument("--linear-probe", action="store_true", help="Train frozen-embedding linear probes per node type")
    parser.add_argument("--fisher-ratio", action="store_true", help="Compute class Fisher ratio on final embeddings per node type")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=30)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--i-avg-scope",
        default="destination",
        choices=["destination", "global"],
        help="Report graph-level I_avg or incoming-relation I_avg for each target destination type",
    )
    return parser.parse_args()


def build_trainer_config(args):
    config = {
        "hidden_channels": args.hidden,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "eval_every": args.eval_every,
    }
    if args.device is not None:
        config["device"] = args.device
    return config


def build_multitype_eval_config(args):
    return MultiTypeEvalConfig(
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
        device=args.device,
        probe_epochs=args.probe_epochs,
        probe_patience=args.probe_patience,
        probe_lr=args.probe_lr,
        probe_weight_decay=args.probe_weight_decay,
    )


def resolve_target_types(args, yaml_payload: dict | None):
    if args.target_types:
        return list(args.target_types)
    if yaml_payload is not None:
        if "target_types" in yaml_payload:
            return list(yaml_payload["target_types"])
        return list(yaml_payload["node_types"].keys())
    return list(DEFAULT_TYPES)


def resolve_supervision_types(args, yaml_payload: dict | None):
    if yaml_payload is not None and "supervision_types" in yaml_payload:
        return list(yaml_payload["supervision_types"])
    return resolve_target_types(args, yaml_payload)


def attach_target_split(data, target_type: str, split_seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    labels = data[target_type].y.detach().cpu().numpy()
    rng = np.random.default_rng(split_seed)
    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = cls_idx.size
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_mask[cls_idx[:n_train]] = True
        val_mask[cls_idx[n_train:n_train + n_val]] = True
        test_mask[cls_idx[n_train + n_val:]] = True
    data[target_type].train_mask = torch_from_bool(train_mask)
    data[target_type].val_mask = torch_from_bool(val_mask)
    data[target_type].test_mask = torch_from_bool(test_mask)


def torch_from_bool(values):
    import torch

    return torch.from_numpy(values).bool()


def train_one_target(data, target_type: str, args):
    import torch

    work_data = copy.deepcopy(data)
    attach_target_split(work_data, target_type=target_type, split_seed=args.split_seed)
    set_seed(args.train_seed)
    num_classes = int(work_data[target_type].y.max().item() + 1)
    model = SimpleHeteroSAGE.from_data(
        work_data,
        target_type=target_type,
        num_classes=num_classes,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    trainer = NodeClassificationTrainer(model, work_data, target_type, build_trainer_config(args))
    results = trainer.run()
    results["device"] = str(trainer.device)
    return results


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def select_i_avg(meta: dict, target_type: str, scope: str):
    global_summary = meta["global_summary"]
    destination_summary = meta.get("destination_summary", {}).get(target_type, {})
    if scope == "global":
        selected = global_summary
    else:
        selected = destination_summary
    return {
        "I_avg_scope": scope,
        "I_avg_empirical_nats": float(selected.get("I_avg_empirical_nats", 0.0)),
        "I_avg_theoretical_nats": float(selected.get("I_avg_theoretical_nats", 0.0)),
        "I_avg_global_empirical_nats": float(global_summary.get("I_avg_empirical_nats", 0.0)),
        "I_avg_global_theoretical_nats": float(global_summary.get("I_avg_theoretical_nats", 0.0)),
        "I_avg_destination_empirical_nats": float(destination_summary.get("I_avg_empirical_nats", 0.0)),
        "I_avg_destination_theoretical_nats": float(destination_summary.get("I_avg_theoretical_nats", 0.0)),
        "I_avg_destination_num_edges": int(destination_summary.get("num_forward_edges", 0)),
        "I_avg_destination_num_relations": int(destination_summary.get("num_forward_relations", 0)),
    }


def plot_grid(rows, gammas, target_types, output_path: Path, title: str, metric: str = "accuracy", ylabel: str | None = None):
    fig, axes = plt.subplots(len(gammas), len(target_types), figsize=(4.8 * len(target_types), 3.6 * len(gammas)), squeeze=False)
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    label = ylabel or metric.replace("_", " ")
    for i, gamma in enumerate(gammas):
        gamma_rows = [row for row in rows if abs(row["gamma"] - gamma) < 1e-12]
        for j, target in enumerate(target_types):
            ax = axes[i, j]
            target_rows = sorted(
                [row for row in gamma_rows if row["target_type"] == target],
                key=lambda item: item["tau"],
            )
            xs = [row["tau"] for row in target_rows]
            ys = [row[mean_key] for row in target_rows]
            y_std = [row[std_key] for row in target_rows]
            ax.plot(xs, ys, marker="o", color="#2b8cbe", linewidth=2, label=label)
            lower = np.array(ys) - np.array(y_std)
            upper = np.array(ys) + np.array(y_std)
            if metric in {"accuracy", "macro_f1", "micro_f1", "auc", "linear_probe_accuracy"}:
                lower = np.maximum(0.0, lower)
                upper = np.minimum(1.0, upper)
            ax.fill_between(xs, lower, upper, color="#2b8cbe", alpha=0.15)
            if metric in {"accuracy", "macro_f1", "micro_f1", "auc", "linear_probe_accuracy"}:
                y_min = max(0.0, float(np.min(ys) - 0.05))
                y_max = min(1.0, float(np.max(ys) + 0.05))
            else:
                span = float(np.max(ys) - np.min(ys))
                pad = 0.05 if span == 0 else 0.1 * span
                y_min = float(np.min(ys) - pad)
                y_max = float(np.max(ys) + pad)
            if y_max <= y_min:
                y_min = y_min - 0.05
                y_max = y_max + 0.05
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.25)
            if i == len(gammas) - 1:
                ax.set_xlabel(r"$\tau$")
            if j == 0:
                ax.set_ylabel(label)
            ax.set_title(f"gamma={gamma}, target={target}", fontsize=11)
            if i == 0 and j == len(target_types) - 1:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def save_rows(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(seed_rows):
    grouped = {}
    for row in seed_rows:
        key = (row["gamma"], row["tau"], row["target_type"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (gamma, tau, target_type), rows in sorted(grouped.items()):
        out = {
            "gamma": float(gamma),
            "tau": float(tau),
            "target_type": target_type,
            "num_runs": len(rows),
            "I_avg_scope": rows[0].get("I_avg_scope", "global"),
            "I_avg_empirical_nats": float(rows[0]["I_avg_empirical_nats"]),
            "I_avg_theoretical_nats": float(rows[0]["I_avg_theoretical_nats"]),
            "I_avg_global_empirical_nats": float(rows[0].get("I_avg_global_empirical_nats", rows[0]["I_avg_empirical_nats"])),
            "I_avg_global_theoretical_nats": float(rows[0].get("I_avg_global_theoretical_nats", rows[0]["I_avg_theoretical_nats"])),
            "I_avg_destination_empirical_nats": float(rows[0].get("I_avg_destination_empirical_nats", rows[0]["I_avg_empirical_nats"])),
            "I_avg_destination_theoretical_nats": float(rows[0].get("I_avg_destination_theoretical_nats", rows[0]["I_avg_theoretical_nats"])),
            "I_avg_destination_num_edges": int(rows[0].get("I_avg_destination_num_edges", 0)),
            "I_avg_destination_num_relations": int(rows[0].get("I_avg_destination_num_relations", 0)),
        }
        for metric in ["accuracy", "macro_f1", "micro_f1", "auc"]:
            mean, std = _mean_std([row[metric] for row in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        for metric in ["linear_probe_accuracy", "linear_probe_macro_f1", "linear_probe_micro_f1", "linear_probe_auc", "fisher_ratio"]:
            if metric not in rows[0]:
                continue
            mean, std = _mean_std([row[metric] for row in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        epochs_mean, epochs_std = _mean_std([row["epochs_trained"] for row in rows])
        params_mean, params_std = _mean_std([row["num_parameters"] for row in rows])
        out["epochs_trained_mean"] = epochs_mean
        out["epochs_trained_std"] = epochs_std
        out["num_parameters_mean"] = params_mean
        out["num_parameters_std"] = params_std
        summary.append(out)
    return summary


def build_sweep_config(args, *, gamma: float, tau: float, graph_seed: int, split_seed: int, target_type: str, yaml_payload: dict | None):
    if yaml_payload is not None:
        return build_sweep_config_from_yaml(
            yaml_payload,
            gamma=gamma,
            tau=tau,
            seed=graph_seed,
            target_type=target_type,
            split_seed=split_seed,
        )
    return complete_directed_four_type_config(
        feature_dim=args.feature_dim,
        gamma=gamma,
        tau=tau,
        sigma=args.sigma,
        out_degree=args.out_degree,
        num_nodes=args.num_nodes,
        class_counts=args.class_counts if args.class_counts is not None else DEFAULT_CLASS_COUNTS,
        type_names=args.type_names if args.type_names is not None else DEFAULT_TYPES,
        seed=graph_seed,
        target_type=target_type,
    )


def run_one_multitype_eval(args, *, yaml_payload: dict | None, output_dir: Path, title: str):
    target_types = resolve_target_types(args, yaml_payload)
    supervision_types = resolve_supervision_types(args, yaml_payload)
    seed_rows = []
    raw_payload = []
    eval_cfg = build_multitype_eval_config(args)
    for gamma in args.gammas:
        for tau in args.taus:
            for seed_offset in range(args.num_seeds):
                graph_seed = args.graph_seed + seed_offset
                train_seed = args.train_seed + seed_offset
                split_seed = args.split_seed + seed_offset
                cfg = build_sweep_config(
                    args,
                    gamma=gamma,
                    tau=tau,
                    graph_seed=graph_seed,
                    split_seed=split_seed,
                    target_type=target_types[0],
                    yaml_payload=yaml_payload,
                )
                data, meta = generate_hcsbm(cfg)
                split_data = attach_all_type_splits(data, target_types=supervision_types, split_seed=split_seed)
                print(
                    f"[start] mode=multi_type gamma={gamma} tau={tau} seed={seed_offset} "
                    f"targets={','.join(target_types)} supervision={','.join(supervision_types)} layers={args.layers}",
                    flush=True,
                )
                results = train_multitype_supervised(
                    split_data,
                    target_types=supervision_types,
                    cfg=eval_cfg,
                    seed=train_seed,
                )
                print(
                    f"[trained] mode=multi_type gamma={gamma} tau={tau} seed={seed_offset} "
                    f"device={results['device']} epochs={results['epochs_trained']}",
                    flush=True,
                )
                embedding_metrics = {}
                if args.linear_probe or args.fisher_ratio:
                    print(
                        f"[eval-emb] mode=multi_type gamma={gamma} tau={tau} seed={seed_offset} "
                        f"linear_probe={args.linear_probe} fisher_ratio={args.fisher_ratio}",
                        flush=True,
                    )
                    embeddings = extract_multitype_embeddings(results["model"], results["data"])
                    embedding_metrics = evaluate_embeddings_by_type(
                        embeddings=embeddings,
                        data=results["data"],
                        target_types=supervision_types,
                        cfg=eval_cfg,
                        seed=train_seed,
                        run_linear_probe=args.linear_probe,
                        run_fisher=args.fisher_ratio,
                    )
                raw_payload.append(
                    {
                        "gamma": gamma,
                        "tau": tau,
                        "seed_offset": seed_offset,
                        "graph_seed": graph_seed,
                        "train_seed": train_seed,
                        "split_seed": split_seed,
                        "metadata": meta,
                    }
                )
                for target_type in target_types:
                    test_metrics = results["test"][target_type]
                    i_avg_values = select_i_avg(meta, target_type=target_type, scope=args.i_avg_scope)
                    row = {
                        "gamma": float(gamma),
                        "tau": float(tau),
                        "target_type": target_type,
                        "supervision_mode": "multi_type",
                        "seed_offset": int(seed_offset),
                        "graph_seed": int(graph_seed),
                        "train_seed": int(train_seed),
                        "split_seed": int(split_seed),
                        "accuracy": float(test_metrics["accuracy"]),
                        "macro_f1": float(test_metrics["macro_f1"]),
                        "micro_f1": float(test_metrics["micro_f1"]),
                        "auc": float(test_metrics["auc"]),
                        "epochs_trained": int(results["epochs_trained"]),
                        "num_parameters": int(results["num_parameters"]),
                    }
                    for key, value in embedding_metrics.get(target_type, {}).items():
                        row[key] = float(value)
                    row.update(i_avg_values)
                    seed_rows.append(row)
                    print(
                        f"[done] mode=multi_type gamma={gamma} tau={tau} seed={seed_offset} "
                        f"target={target_type} device={results['device']} acc={row['accuracy']:.4f} "
                        f"macro_f1={row['macro_f1']:.4f} I_avg_{row['I_avg_scope']}={row['I_avg_empirical_nats']:.4f}",
                        flush=True,
                    )

    summary_rows = aggregate_rows(seed_rows)
    save_rows(seed_rows, output_dir / "per_seed.csv")
    save_rows(summary_rows, output_dir / "summary.csv")
    with open(output_dir / "raw_metadata.json", "w") as f:
        json.dump(raw_payload, f, indent=2)
    plot_grid(
        rows=summary_rows,
        gammas=list(args.gammas),
        target_types=target_types,
        output_path=output_dir / "simpleheterosage_tau_sweep",
        title=title,
        metric="accuracy",
        ylabel="accuracy",
    )
    if summary_rows and "linear_probe_accuracy_mean" in summary_rows[0]:
        plot_grid(
            rows=summary_rows,
            gammas=list(args.gammas),
            target_types=target_types,
            output_path=output_dir / "simpleheterosage_linear_probe_tau_sweep",
            title=f"{title} Linear Probe",
            metric="linear_probe_accuracy",
            ylabel="linear probe accuracy",
        )
    if summary_rows and "fisher_ratio_mean" in summary_rows[0]:
        plot_grid(
            rows=summary_rows,
            gammas=list(args.gammas),
            target_types=target_types,
            output_path=output_dir / "simpleheterosage_fisher_ratio_tau_sweep",
            title=f"{title} Fisher Ratio",
            metric="fisher_ratio",
            ylabel="Fisher ratio",
        )


def run_one_sweep(args, *, yaml_payload: dict | None, output_dir: Path, title: str):
    if args.supervision_mode == "multi_type":
        run_one_multitype_eval(args, yaml_payload=yaml_payload, output_dir=output_dir, title=title)
        return
    target_types = resolve_target_types(args, yaml_payload)
    seed_rows = []
    raw_payload = []
    for gamma in args.gammas:
        for tau in args.taus:
            for seed_offset in range(args.num_seeds):
                graph_seed = args.graph_seed + seed_offset
                train_seed = args.train_seed + seed_offset
                split_seed = args.split_seed + seed_offset
                cfg = build_sweep_config(
                    args,
                    gamma=gamma,
                    tau=tau,
                    graph_seed=graph_seed,
                    split_seed=split_seed,
                    target_type=target_types[0],
                    yaml_payload=yaml_payload,
                )
                data, meta = generate_hcsbm(cfg)
                raw_payload.append(
                    {
                        "gamma": gamma,
                        "tau": tau,
                        "seed_offset": seed_offset,
                        "graph_seed": graph_seed,
                        "train_seed": train_seed,
                        "split_seed": split_seed,
                        "metadata": meta,
                    }
                )
                for target_type in target_types:
                    prev_train_seed = args.train_seed
                    prev_split_seed = args.split_seed
                    args.train_seed = train_seed
                    args.split_seed = split_seed
                    print(
                        f"[start] mode=per_target gamma={gamma} tau={tau} seed={seed_offset} "
                        f"target={target_type} layers={args.layers}",
                        flush=True,
                    )
                    results = train_one_target(data, target_type=target_type, args=args)
                    args.train_seed = prev_train_seed
                    args.split_seed = prev_split_seed
                    i_avg_values = select_i_avg(meta, target_type=target_type, scope=args.i_avg_scope)
                    row = {
                        "gamma": float(gamma),
                        "tau": float(tau),
                        "target_type": target_type,
                        "supervision_mode": "per_target",
                        "seed_offset": int(seed_offset),
                        "graph_seed": int(graph_seed),
                        "train_seed": int(train_seed),
                        "split_seed": int(split_seed),
                        "accuracy": float(results["test"]["accuracy"]),
                        "macro_f1": float(results["test"]["macro_f1"]),
                        "micro_f1": float(results["test"]["micro_f1"]),
                        "auc": float(results["test"]["auc"]),
                        "epochs_trained": int(results["epochs_trained"]),
                        "num_parameters": int(results["num_parameters"]),
                    }
                    row.update(i_avg_values)
                    seed_rows.append(row)
                    print(
                        f"[done] gamma={gamma} tau={tau} seed={seed_offset} target={target_type} "
                        f"device={results['device']} acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
                        f"I_avg_{row['I_avg_scope']}={row['I_avg_empirical_nats']:.4f}",
                        flush=True,
                    )

    summary_rows = aggregate_rows(seed_rows)
    save_rows(seed_rows, output_dir / "per_seed.csv")
    save_rows(summary_rows, output_dir / "summary.csv")
    with open(output_dir / "raw_metadata.json", "w") as f:
        json.dump(raw_payload, f, indent=2)
    plot_grid(
        rows=summary_rows,
        gammas=list(args.gammas),
        target_types=target_types,
        output_path=output_dir / "simpleheterosage_tau_sweep",
        title=title,
        metric="accuracy",
        ylabel="accuracy",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.config_yaml:
        yaml_paths = expand_yaml_paths(args.config_yaml)
        for yaml_path in yaml_paths:
            payload = load_yaml_payload(yaml_path)
            name = payload.get("name", yaml_path.stem)
            run_dir = output_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            title = f"{args.title} [{name}]"
            print(f"[config] {yaml_path} -> {run_dir}", flush=True)
            run_one_sweep(args, yaml_payload=payload, output_dir=run_dir, title=title)
    else:
        run_one_sweep(args, yaml_payload=None, output_dir=output_dir, title=args.title)


if __name__ == "__main__":
    main()
