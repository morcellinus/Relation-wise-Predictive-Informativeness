from __future__ import annotations

import json
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_split_masks(num_nodes: int, y: torch.Tensor, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    labels = y.cpu().numpy()
    labeled = np.where(labels >= 0)[0]
    for cls in np.unique(labels[labeled]):
        idx = labeled[labels[labeled] == cls]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train[idx[:n_train]] = True
        val[idx[n_train:n_train + n_val]] = True
        test[idx[n_train + n_val:]] = True
    return train, val, test


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_logits: Optional[torch.Tensor] = None) -> Dict[str, float]:
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "auc": 0.0,
    }
    if y_logits is not None:
        try:
            probs = torch.softmax(y_logits, dim=-1).detach().cpu().numpy()
            if probs.shape[1] == 2:
                metrics["auc"] = float(roc_auc_score(yt, probs[:, 1]))
            elif len(np.unique(yt)) >= 2:
                metrics["auc"] = float(roc_auc_score(yt, probs, multi_class="ovr", average="macro"))
        except Exception:
            pass
    return metrics


class NodeClassificationTrainer:
    def __init__(self, model, data, target_type: str, config: Dict):
        self.model = model
        self.data = data
        self.target_type = target_type
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config.get("lr", 0.005)
        self.weight_decay = config.get("weight_decay", 5e-4)
        self.epochs = config.get("epochs", 200)
        self.patience = config.get("patience", 30)
        self.eval_every = max(1, int(config.get("eval_every", 1)))
        self.model = self.model.to(self.device)
        for nt in data.node_types:
            if getattr(data[nt], "x", None) is not None:
                data[nt].x = data[nt].x.to(self.device)
        for et in data.edge_types:
            data[et].edge_index = data[et].edge_index.to(self.device)
        self.y = data[target_type].y.to(self.device)
        self.train_mask = data[target_type].train_mask.to(self.device)
        self.val_mask = data[target_type].val_mask.to(self.device)
        self.test_mask = data[target_type].test_mask.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.data)
        loss = F.cross_entropy(logits[self.train_mask], self.y[self.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def _evaluate_outputs(self, mask: torch.Tensor):
        self.model.eval()
        logits = self.model(self.data)
        masked_logits = logits[mask]
        masked_y = self.y[mask]
        loss = float(F.cross_entropy(masked_logits, masked_y).item())
        preds = masked_logits.argmax(dim=-1)
        return loss, masked_y, preds, masked_logits

    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor, return_metrics: bool = True):
        loss, masked_y, preds, masked_logits = self._evaluate_outputs(mask)
        if not return_metrics:
            return loss, None
        return loss, compute_metrics(masked_y, preds, masked_logits)

    def run(self) -> Dict:
        best_val = float("inf")
        best_state = None
        best_val_metrics = {}
        no_improve = 0
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            self.train_epoch()
            if epoch % self.eval_every != 0 and epoch != self.epochs:
                continue
            val_loss, masked_y, preds, masked_logits = self._evaluate_outputs(self.val_mask)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                best_val_metrics = compute_metrics(masked_y, preds, masked_logits)
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        _, test_metrics = self.evaluate(self.test_mask)
        return {
            "best_val": best_val_metrics,
            "test": test_metrics,
            "epochs_trained": epoch,
            "total_time_seconds": time.time() - t0,
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
        }

    def save_results(self, results: Dict, out_path: str):
        import os

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
