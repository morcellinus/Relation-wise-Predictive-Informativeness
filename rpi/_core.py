from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch


def edge_type_name(edge_type: tuple[str, str, str]) -> str:
    return f"{edge_type[0]}-{edge_type[1]}-{edge_type[2]}"


def global_offsets(data) -> tuple[dict[str, int], int]:
    offsets = {}
    cursor = 0
    for nt in data.node_types:
        offsets[nt] = cursor
        cursor += int(data[nt].num_nodes)
    return offsets, cursor


def build_transition(data, relations: list[tuple[str, str, str]]) -> sp.csr_matrix:
    offsets, total = global_offsets(data)
    rows = []
    cols = []
    for src, rel, dst in relations:
        if (src, rel, dst) not in data.edge_types:
            continue
        ei = data[(src, rel, dst)].edge_index.cpu().numpy()
        s = ei[0] + offsets[src]
        d = ei[1] + offsets[dst]
        rows.extend([s, d])
        cols.extend([d, s])
    if not rows:
        return sp.csr_matrix((total, total), dtype=np.float32)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((val, (row, col)), shape=(total, total), dtype=np.float32).tocsr()
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    inv_degree = np.reciprocal(np.maximum(degree, 1.0)).astype(np.float32)
    return sp.diags(inv_degree) @ adj


def propagate_profiles(
    data, target_type: str, relations: list[tuple[str, str, str]], alpha: float, steps: int,
    *, y: torch.Tensor, train_mask: torch.Tensor,
):
    seed_mask = train_mask & (y >= 0)
    num_classes = int(y[y >= 0].max().item() + 1)
    offsets, total = global_offsets(data)
    q0 = np.zeros((total, num_classes), dtype=np.float32)
    target_offset = offsets[target_type]
    seed_local = seed_mask.nonzero(as_tuple=False).view(-1).numpy()
    if seed_local.size:
        q0[target_offset + seed_local, y[seed_mask].numpy()] = 1.0
    transition = build_transition(data, relations)
    q = q0.copy()
    seed_global = target_offset + seed_local
    for _ in range(int(steps)):
        q = float(alpha) * (transition @ q) + (1.0 - float(alpha)) * q0
        if seed_global.size:
            q[seed_global] = q0[seed_global]
    row_sum = q.sum(axis=1, keepdims=True)
    nonzero = row_sum[:, 0] > 0
    q[nonzero] /= row_sum[nonzero]
    q[~nonzero] = 1.0 / float(num_classes)
    profiles = {}
    hard = {}
    confidence = {}
    for nt in data.node_types:
        start = offsets[nt]
        end = start + int(data[nt].num_nodes)
        tensor = torch.from_numpy(q[start:end]).float()
        profiles[nt] = tensor
        hard[nt] = tensor.argmax(dim=1).long()
        confidence[nt] = tensor.max(dim=1).values
    metadata = {
        "target_type": target_type,
        "num_classes": num_classes,
        "alpha": float(alpha),
        "steps": int(steps),
        "seed_train_nodes": int(seed_mask.sum().item()),
        "seeded_train_nodes_per_class": torch.bincount(y[seed_mask], minlength=num_classes).cpu().tolist(),
        "supervision": "train_target_labels_only",
    }
    return {"profiles": profiles, "hard_labels": hard, "confidence": confidence, "metadata": metadata}


def _localize(labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    labels = labels.view(-1).long().cpu()
    valid = labels >= 0
    vals = sorted(int(v) for v in labels[valid].unique().tolist())
    mapping = {v: i for i, v in enumerate(vals)}
    out = torch.full_like(labels, -1)
    for old, new in mapping.items():
        out[labels == old] = int(new)
    return out, len(vals)


def score_relations(data, labels: dict[str, torch.Tensor], relations: list[tuple[str, str, str]]):
    rows = []
    for et in relations:
        src, rel, dst = et
        if et not in data.edge_types:
            continue
        ei = data[et].edge_index.cpu()
        src_lbl, src_k = _localize(labels[src])
        dst_lbl, dst_k = _localize(labels[dst])
        s = src_lbl[ei[0]]
        d = dst_lbl[ei[1]]
        valid = (s >= 0) & (d >= 0)
        edge_count = int(valid.sum().item())
        if edge_count == 0 or src_k == 0 or dst_k == 0:
            h_dst = mi_val = rpi = float("nan")
        else:
            joint = torch.zeros(src_k, dst_k, dtype=torch.float64)
            for si, di in zip(s[valid].tolist(), d[valid].tolist()):
                joint[int(si), int(di)] += 1.0
            joint /= float(edge_count)
            p_s = joint.sum(dim=1)
            p_d = joint.sum(dim=0)
            denom = torch.outer(p_s, p_d)
            mask = (joint > 0) & (denom > 0)
            mi = torch.zeros_like(joint)
            mi[mask] = joint[mask] * torch.log(joint[mask] / denom[mask])
            mi_val = float(mi.sum().item())
            h_dst = float(-(p_d[p_d > 0] * torch.log(p_d[p_d > 0])).sum().item())
            rpi = float(mi_val / h_dst) if h_dst > 0 else float("nan")
        rows.append(
            {
                "relation": edge_type_name(et),
                "src_type": src,
                "dst_type": dst,
                "relation_name": rel,
                "num_edges": edge_count,
                "src_num_labels": int(src_k),
                "dst_num_labels": int(dst_k),
                "H_dst": h_dst,
                "MI": mi_val,
                "RPI": rpi,
            }
        )
    return rows


def aggregate_by_destination(rows: list[dict]):
    by_dst = defaultdict(list)
    for row in rows:
        val = float(row["RPI"])
        if math.isnan(val):
            continue
        by_dst[row["dst_type"]].append(row)
    type_rows = []
    for dst, vals in sorted(by_dst.items()):
        simple = sum(float(r["RPI"]) for r in vals) / len(vals)
        den = sum(float(r["num_edges"]) for r in vals)
        weighted = sum(float(r["RPI"]) * float(r["num_edges"]) for r in vals) / den if den > 0 else float("nan")
        type_rows.append(
            {
                "node_type": dst,
                "num_incoming_relations": len(vals),
                "RPI_type": simple,
                "RPI_type_weighted_aux": weighted,
            }
        )
    return type_rows
