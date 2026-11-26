from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from simpleheterosage.models.common.layers import get_in_channels_dict
from simpleheterosage.trainers.node_cls_trainer import compute_metrics, set_seed


@dataclass
class MultiTypeEvalConfig:
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 30
    eval_every: int = 1
    device: str | None = None
    probe_epochs: int = 200
    probe_patience: int = 30
    probe_lr: float = 0.01
    probe_weight_decay: float = 0.0


class MultiTypeSimpleHeteroSAGE(nn.Module):
    def __init__(
        self,
        in_channels_dict: Mapping[str, int],
        hidden_channels: int,
        num_classes_dict: Mapping[str, int],
        target_types: Iterable[str],
        edge_types,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.node_types = list(in_channels_dict)
        self.target_types = list(target_types)
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_ch, hidden_channels) for nt, in_ch in in_channels_dict.items()
        })
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs = {et: SAGEConv((hidden_channels, hidden_channels), hidden_channels) for et in edge_types}
            self.convs.append(HeteroConv(convs, aggr="sum"))
        self.classifier_heads = nn.ModuleDict({
            nt: nn.Linear(hidden_channels, int(num_classes_dict[nt])) for nt in self.target_types
        })

    def encode(self, data):
        x_dict = {nt: F.relu(self.input_proj[nt](data[nt].x)) for nt in self.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                nt: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                for nt, x in x_dict.items()
            }
        return x_dict

    def forward(self, data):
        h_dict = self.encode(data)
        missing = [nt for nt in self.target_types if nt not in h_dict]
        if missing:
            raise KeyError(
                "No representation was produced for target types "
                f"{missing}. Each evaluated type needs at least one incoming relation."
            )
        return {nt: self.classifier_heads[nt](h_dict[nt]) for nt in self.target_types}


def attach_all_type_splits(
    data,
    target_types: Iterable[str],
    split_seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    work_data = copy.deepcopy(data)
    for offset, node_type in enumerate(target_types):
        labels = work_data[node_type].y.detach().cpu().numpy()
        rng = np.random.default_rng(int(split_seed) + 1009 * offset)
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
        work_data[node_type].train_mask = torch.from_numpy(train_mask).bool()
        work_data[node_type].val_mask = torch.from_numpy(val_mask).bool()
        work_data[node_type].test_mask = torch.from_numpy(test_mask).bool()
    return work_data


def _move_data_to_device(data, device: str):
    for nt in data.node_types:
        if getattr(data[nt], "x", None) is not None:
            data[nt].x = data[nt].x.to(device)
        if getattr(data[nt], "y", None) is not None:
            data[nt].y = data[nt].y.to(device)
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            if hasattr(data[nt], mask_name):
                setattr(data[nt], mask_name, getattr(data[nt], mask_name).to(device))
    for et in data.edge_types:
        data[et].edge_index = data[et].edge_index.to(device)
    return data


def _num_classes_dict(data, target_types: Iterable[str]) -> Dict[str, int]:
    return {nt: int(data[nt].y.max().item() + 1) for nt in target_types}


def _masked_loss_and_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    masked_logits = logits[mask]
    masked_y = y[mask]
    loss = F.cross_entropy(masked_logits, masked_y)
    preds = masked_logits.argmax(dim=-1)
    return loss, compute_metrics(masked_y, preds, masked_logits)


@torch.no_grad()
def evaluate_multitype_model(model, data, target_types: Iterable[str], split: str):
    model.eval()
    logits_dict = model(data)
    out = {}
    for nt in target_types:
        y = data[nt].y
        mask = getattr(data[nt], f"{split}_mask")
        loss, metrics = _masked_loss_and_metrics(logits_dict[nt], y, mask)
        out[nt] = {"loss": float(loss.item()), **metrics}
    return out


def train_multitype_supervised(
    data,
    target_types: List[str],
    cfg: MultiTypeEvalConfig,
    seed: int,
):
    set_seed(seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = _move_data_to_device(data, device)
    model = MultiTypeSimpleHeteroSAGE(
        in_channels_dict=get_in_channels_dict(data),
        hidden_channels=cfg.hidden_channels,
        num_classes_dict=_num_classes_dict(data, target_types),
        target_types=target_types,
        edge_types=list(data.edge_types),
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    best_val_metrics = {}
    no_improve = 0
    t0 = time.time()
    last_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad()
        logits_dict = model(data)
        losses = []
        for nt in target_types:
            y = data[nt].y
            losses.append(F.cross_entropy(logits_dict[nt][data[nt].train_mask], y[data[nt].train_mask]))
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()

        if epoch % max(1, cfg.eval_every) != 0 and epoch != cfg.epochs:
            continue
        val_results = evaluate_multitype_model(model, data, target_types, "val")
        val_loss = sum(item["loss"] for item in val_results.values())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_results
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_results = evaluate_multitype_model(model, data, target_types, "test")
    return {
        "model": model,
        "data": data,
        "device": device,
        "best_val": best_val_metrics,
        "test": test_results,
        "epochs_trained": last_epoch,
        "total_time_seconds": time.time() - t0,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }


@torch.no_grad()
def extract_simpleheterosage_embeddings(model, data):
    model.eval()
    x_dict = {nt: F.relu(model.input_proj[nt](data[nt].x)) for nt in model.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    for conv in model.convs:
        x_dict = conv(x_dict, edge_index_dict)
        x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}
    return x_dict


@torch.no_grad()
def extract_multitype_embeddings(model: MultiTypeSimpleHeteroSAGE, data):
    model.eval()
    return model.encode(data)


def fisher_ratio(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    x = embeddings.detach()
    y = labels.detach()
    classes = torch.unique(y)
    if classes.numel() <= 1:
        return 0.0
    global_mean = x.mean(dim=0)
    inter = torch.zeros((), device=x.device)
    intra = torch.zeros((), device=x.device)
    total = max(1, int(x.size(0)))
    for cls in classes:
        cls_x = x[y == cls]
        if cls_x.numel() == 0:
            continue
        cls_mean = cls_x.mean(dim=0)
        inter = inter + cls_x.size(0) * torch.sum((cls_mean - global_mean) ** 2)
        intra = intra + torch.sum((cls_x - cls_mean) ** 2)
    return float((inter / total / (intra / total + 1e-12)).detach().cpu().item())


def train_linear_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    cfg: MultiTypeEvalConfig,
    seed: int,
):
    set_seed(seed)
    device = embeddings.device
    probe = nn.Linear(embeddings.size(-1), num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay)
    best_val = float("inf")
    best_state = None
    no_improve = 0
    last_epoch = 0
    frozen = embeddings.detach()
    for epoch in range(1, cfg.probe_epochs + 1):
        last_epoch = epoch
        probe.train()
        optimizer.zero_grad()
        logits = probe(frozen)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(frozen)[val_mask]
            val_loss = float(F.cross_entropy(val_logits, labels[val_mask]).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.probe_patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_logits = probe(frozen)[test_mask]
        test_y = labels[test_mask]
        test_pred = test_logits.argmax(dim=-1)
        metrics = compute_metrics(test_y, test_pred, test_logits)
    metrics["epochs_trained"] = int(last_epoch)
    return metrics


def evaluate_embeddings_by_type(
    embeddings: Mapping[str, torch.Tensor],
    data,
    target_types: Iterable[str],
    cfg: MultiTypeEvalConfig,
    seed: int,
    run_linear_probe: bool = False,
    run_fisher: bool = False,
):
    out = {}
    for offset, nt in enumerate(target_types):
        type_out = {}
        if nt not in embeddings:
            out[nt] = type_out
            continue
        h = embeddings[nt]
        y = data[nt].y
        if run_fisher:
            type_out["fisher_ratio"] = fisher_ratio(h, y)
        if run_linear_probe:
            probe_metrics = train_linear_probe(
                embeddings=h,
                labels=y,
                train_mask=data[nt].train_mask,
                val_mask=data[nt].val_mask,
                test_mask=data[nt].test_mask,
                num_classes=int(y.max().item() + 1),
                cfg=cfg,
                seed=int(seed) + 7919 * offset,
            )
            for key, value in probe_metrics.items():
                type_out[f"linear_probe_{key}"] = value
        out[nt] = type_out
    return out
