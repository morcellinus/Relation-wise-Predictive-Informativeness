from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def normalize_probability_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(vec < 0):
        raise ValueError(f"{name} must be non-negative")
    total = vec.sum()
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return vec / total


def cumulative_intervals(prior: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(prior, dtype=np.float64)
    starts = np.concatenate(([0.0], ends[:-1]))
    return starts, ends


def interval_overlap_coupling(source_prior: np.ndarray, destination_prior: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = normalize_probability_vector(source_prior, name="source_prior")
    dst = normalize_probability_vector(destination_prior, name="destination_prior")
    src_starts, src_ends = cumulative_intervals(src)
    dst_starts, dst_ends = cumulative_intervals(dst)

    joint = np.maximum(
        0.0,
        np.minimum(src_ends[:, None], dst_ends[None, :]) - np.maximum(src_starts[:, None], dst_starts[None, :]),
    )
    coupling = np.divide(joint, src[:, None], out=np.zeros_like(joint), where=src[:, None] > 0)
    coupling /= coupling.sum(axis=1, keepdims=True)
    return joint, coupling


def baseline_relation_matrix(destination_prior: np.ndarray, num_source_classes: int) -> np.ndarray:
    dst = normalize_probability_vector(destination_prior, name="destination_prior")
    return np.broadcast_to(dst[None, :], (num_source_classes, dst.size)).copy()


def _normalize_relation_matrix_rows(matrix: np.ndarray, *, name: str) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if np.any(mat < 0):
        raise ValueError(f"{name} must be non-negative")
    row_sums = mat.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"{name} rows must have positive mass")
    return mat / row_sums


def effective_relation_matrix(
    source_prior: np.ndarray,
    destination_prior: np.ndarray,
    tau: float,
    *,
    aligned_matrix: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    src = normalize_probability_vector(source_prior, name="source_prior")
    dst = normalize_probability_vector(destination_prior, name="destination_prior")
    if aligned_matrix is None:
        joint_aligned, aligned = interval_overlap_coupling(src, dst)
    else:
        aligned = _normalize_relation_matrix_rows(aligned_matrix, name="aligned_matrix")
        if aligned.shape != (src.size, dst.size):
            raise ValueError(
                f"aligned_matrix shape mismatch: expected {(src.size, dst.size)}, got {aligned.shape}"
            )
        joint_aligned = src[:, None] * aligned
    baseline = baseline_relation_matrix(dst, src.size)
    effective = (1.0 - tau) * baseline + tau * aligned
    return {
        "baseline": baseline,
        "aligned_joint": joint_aligned,
        "aligned": aligned,
        "effective": effective,
    }


def theoretical_joint_from_relation(source_prior: np.ndarray, relation_matrix: np.ndarray) -> np.ndarray:
    src = normalize_probability_vector(source_prior, name="source_prior")
    mat = np.asarray(relation_matrix, dtype=np.float64)
    return src[:, None] * mat


def reverse_relation_matrix_from_joint(joint: np.ndarray, reverse_source_prior: np.ndarray) -> np.ndarray:
    rev_src = normalize_probability_vector(reverse_source_prior, name="reverse_source_prior")
    joint = np.asarray(joint, dtype=np.float64)
    reverse = np.divide(joint.T, rev_src[:, None], out=np.zeros((joint.shape[1], joint.shape[0]), dtype=np.float64), where=rev_src[:, None] > 0)
    row_sums = reverse.sum(axis=1, keepdims=True)
    reverse = np.divide(reverse, row_sums, out=np.zeros_like(reverse), where=row_sums > 0)
    return reverse


def mutual_information_from_joint(joint: np.ndarray) -> float:
    joint = np.asarray(joint, dtype=np.float64)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    p_row = joint.sum(axis=1, keepdims=True)
    p_col = joint.sum(axis=0, keepdims=True)
    denom = p_row @ p_col
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def mutual_information_from_relation(source_prior: np.ndarray, destination_prior: np.ndarray, relation_matrix: np.ndarray) -> float:
    src = normalize_probability_vector(source_prior, name="source_prior")
    dst = normalize_probability_vector(destination_prior, name="destination_prior")
    mat = np.asarray(relation_matrix, dtype=np.float64)
    joint = src[:, None] * mat
    ratio = np.divide(mat, dst[None, :], out=np.ones_like(mat), where=dst[None, :] > 0)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(ratio[mask])))


def empirical_label_joint(
    source_labels: np.ndarray,
    destination_labels: np.ndarray,
    num_source_classes: int,
    num_destination_classes: int,
) -> np.ndarray:
    if source_labels.shape != destination_labels.shape:
        raise ValueError("source_labels and destination_labels must have matching shapes")
    if source_labels.size == 0:
        return np.zeros((num_source_classes, num_destination_classes), dtype=np.float64)
    flat = source_labels.astype(np.int64) * num_destination_classes + destination_labels.astype(np.int64)
    counts = np.bincount(flat, minlength=num_source_classes * num_destination_classes).reshape(
        num_source_classes, num_destination_classes
    )
    return counts.astype(np.float64) / float(source_labels.size)


def empirical_mutual_information(
    source_labels: np.ndarray,
    destination_labels: np.ndarray,
    num_source_classes: int,
    num_destination_classes: int,
) -> float:
    joint = empirical_label_joint(source_labels, destination_labels, num_source_classes, num_destination_classes)
    return mutual_information_from_joint(joint)


def relation_name_to_edge_type(
    relation_name: str,
    source_type: str,
    destination_type: str,
    relation_aliases: Dict[str, Tuple[str, str, str]] | None = None,
) -> Tuple[str, str, str]:
    if relation_aliases and relation_name in relation_aliases:
        return relation_aliases[relation_name]
    return source_type, relation_name, destination_type


def reverse_relation_name(relation_name: str) -> str:
    return f"{relation_name}_rev"


def make_reverse_edge_type(edge_type: Tuple[str, str, str]) -> Tuple[str, str, str]:
    src, rel, dst = edge_type
    return dst, reverse_relation_name(rel), src


def choose_feature_dim(common_dim: int, type_specific_dim: int | None) -> int:
    dim = type_specific_dim if type_specific_dim is not None else common_dim
    if dim <= 0:
        raise ValueError("feature dimension must be positive")
    return dim
