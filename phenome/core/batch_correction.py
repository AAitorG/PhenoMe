"""
Batch correction (plate / batch effects) for embedding or property matrices.

Default method is **sphering**: control-based mean centering and whitening per batch
using an eigendecomposition of the control covariance with ridge regularization.

Designed for use with streaming fetches: statistics are accumulated in passes over
chunks of rows without requiring the full matrix in memory.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .results_metadata import _is_missing_store_value, filter_indices, get_metadata_value_from_dict

MethodName = Literal["sphering", "zscore"]


def stable_batch_id(value: Any) -> str:
    """String id for grouping rows by batch metadata value."""
    if _is_missing_store_value(value):
        return "__missing__"
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))
    return str(value)


def validate_batch_metadata_present(
    metadata: Sequence[Mapping[str, Any] | dict[str, Any]],
    batch_metadata_key: str,
) -> None:
    """Raise when any row lacks a usable batch metadata value."""
    missing_rows = 0
    for meta in metadata:
        if not isinstance(meta, dict):
            meta = {}
        val = get_metadata_value_from_dict(meta, batch_metadata_key)
        if stable_batch_id(val) == "__missing__":
            missing_rows += 1
    if missing_rows > 0:
        raise ValueError(
            f"{missing_rows} of {len(metadata)} row(s) lack batch metadata key "
            f"{batch_metadata_key!r} (mapped to {stable_batch_id(None)!r}). "
            "Fix the metadata column name or fill missing values before batch correction."
        )


@dataclass
class BatchCorrectionStats:
    """Per-batch statistics for correction."""

    batch_id: str
    mean: np.ndarray  # (D,) float32
    n_controls: int
    # Sphering: (X - mean) @ whiten_mat.T  -> (N, D)
    whiten_mat: np.ndarray | None = None
    # Z-score: (X - mean) / std  elementwise; std shape (D,)
    std: np.ndarray | None = None


def _normalize_control_filters(control_filters: Mapping[str, Any] | None) -> dict[str, Any]:
    if not control_filters:
        raise ValueError(
            "control_filters is required for batch correction (e.g. negative controls). "
            "Pass a dict like {'Treatment': 'DMSO'} with the same semantics as filter_indices()."
        )
    return dict(control_filters)


def _group_indices_by_batch(
    metadata: Sequence[Mapping[str, Any] | dict[str, Any]],
    batch_metadata_key: str,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            meta = {}
        val = get_metadata_value_from_dict(meta, batch_metadata_key)
        groups[stable_batch_id(val)].append(i)
    return dict(groups)


def _control_indices_for_batch(
    batch_indices: list[int],
    control_set: set[int],
) -> list[int]:
    return sorted(i for i in batch_indices if i in control_set)


def _accumulate_mean(
    indices: np.ndarray,
    fetch_rows: Callable[[np.ndarray], np.ndarray],
    chunk_size: int,
) -> tuple[np.ndarray, int, int]:
    """Return (sum_vector, count, dim) from streaming row fetches."""
    if indices.size == 0:
        raise ValueError("No control indices to accumulate mean.")
    dim = -1
    total = np.zeros(0, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        block = np.asarray(fetch_rows(chunk_idx), dtype=np.float64)
        if block.ndim != 2:
            raise ValueError(f"fetch_rows must return 2-D array, got shape {block.shape}")
        if dim < 0:
            dim = block.shape[1]
            total = np.zeros(dim, dtype=np.float64)
        elif block.shape[1] != dim:
            raise ValueError(
                f"Inconsistent feature dimension: expected {dim}, got {block.shape[1]}"
            )
        total += block.sum(axis=0)
        count += block.shape[0]
    return total, count, dim


def _accumulate_covariance(
    indices: np.ndarray,
    mean: np.ndarray,
    fetch_rows: Callable[[np.ndarray], np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    """Scatter matrix sum (X-mu)^T (X-mu) over rows; divide by (n-1) outside for cov."""
    dim = mean.shape[0]
    ss = np.zeros((dim, dim), dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        block = np.asarray(fetch_rows(chunk_idx), dtype=np.float64)
        y = block - mean.reshape(1, -1)
        ss += y.T @ y
    return ss


def _whiten_matrix_from_cov(
    cov: np.ndarray,
    *,
    ridge_multiplier: float,
) -> np.ndarray:
    """Return (D,D) float32 matrix W such that (X - mu) @ W.T is whitened (row vectors)."""
    d = cov.shape[0]
    tr = float(np.trace(cov))
    ridge = max(1e-12, ridge_multiplier * (tr / max(d, 1)))
    cov_reg = cov + ridge * np.eye(d, dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov_reg)
    # Clamp small / negative eigenvalues from numerical noise
    evals_clamped = np.maximum(evals, ridge)
    inv_sqrt = 1.0 / np.sqrt(evals_clamped)
    # W = Q * diag(inv_sqrt)  -> (D,D); x_new = x @ W  with x row vector
    whiten = (evecs * inv_sqrt.reshape(1, -1)) @ evecs.T
    return whiten.astype(np.float32)


def compute_batch_stats(
    metadata: Sequence[Mapping[str, Any] | dict[str, Any]],
    batch_metadata_key: str,
    control_filters: Mapping[str, Any],
    fetch_rows: Callable[[np.ndarray], np.ndarray],
    method: MethodName = "sphering",
    *,
    chunk_size: int = 4096,
    ridge_multiplier: float = 1e-3,
    min_controls: int = 2,
    results_for_filter: Any | None = None,
) -> tuple[dict[str, BatchCorrectionStats], dict[str, Any]]:
    """Compute per-batch correction statistics from control wells only.

    Args:
        metadata: Per-row metadata dicts (length N), aligned with fetch_rows indices.
        batch_metadata_key: Metadata key for batch / plate id (case-insensitive).
        control_filters: Include rows matching these filters (same as ``filter_indices``).
        fetch_rows: ``(idx_arr,) -> (len(idx_arr), D)`` feature matrix for those indices.
        method: ``\"sphering\"`` or ``\"zscore\"``.
        chunk_size: Max rows per call to ``fetch_rows``.
        ridge_multiplier: Scale for covariance ridge (relative to trace/dim).
        min_controls: Minimum control rows required per batch.
        results_for_filter: Object with ``metadata`` / ``img_path`` for ``filter_indices``;
            if None, a lightweight wrapper is built from *metadata*.

    Returns:
        (stats_by_batch_id, info) where *info* has keys ``batches``, ``controls_per_batch``.
    """
    _normalize_control_filters(control_filters)
    n_meta = len(metadata)
    if n_meta == 0:
        raise ValueError("metadata is empty.")

    if results_for_filter is None:

        class _MetaOnly:
            def __init__(self, meta: Sequence[Any]) -> None:
                self.metadata = list(meta)
                self.img_path = [""] * len(meta)

        results_for_filter = _MetaOnly(metadata)

    n_images = len(results_for_filter.img_path)
    if n_meta != n_images:
        raise ValueError(
            f"Batch correction requires metadata length ({n_meta}) to match "
            f"number of images ({n_images})."
        )

    validate_batch_metadata_present(metadata, batch_metadata_key)

    missing_indices = [
        i
        for i in range(n_meta)
        if stable_batch_id(
            get_metadata_value_from_dict(
                metadata[i] if isinstance(metadata[i], dict) else {},
                batch_metadata_key,
            )
        )
        == "__missing__"
    ]
    if missing_indices:
        raise ValueError(
            f"{len(missing_indices)} row(s) lack batch metadata key "
            f"{batch_metadata_key!r} (indices include {missing_indices[:5]})."
        )

    control_set = set(filter_indices(results_for_filter, filters=dict(control_filters)))
    if not control_set:
        raise ValueError(
            "No rows matched control_filters; cannot estimate batch correction. "
            "Check metadata keys and values."
        )

    groups = _group_indices_by_batch(metadata, batch_metadata_key)
    stats: dict[str, BatchCorrectionStats] = {}
    controls_per_batch: dict[str, int] = {}

    for batch_id, batch_indices in groups.items():
        ctrl = _control_indices_for_batch(batch_indices, control_set)
        controls_per_batch[batch_id] = len(ctrl)
        if len(ctrl) < min_controls:
            raise ValueError(
                f"Batch {batch_id!r} has only {len(ctrl)} control row(s); "
                f"need at least {min_controls}. "
                "Add controls per plate or relax min_controls (not recommended)."
            )
        idx_arr = np.asarray(ctrl, dtype=np.int64)

        total, count, _dim = _accumulate_mean(idx_arr, fetch_rows, chunk_size)
        mean64 = total / max(count, 1)
        mean = mean64.astype(np.float32)

        if method == "zscore":
            ss = _accumulate_covariance(idx_arr, mean64, fetch_rows, chunk_size)
            var = np.diag(ss) / max(count - 1, 1)
            std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
            stats[batch_id] = BatchCorrectionStats(
                batch_id=batch_id,
                mean=mean,
                n_controls=len(ctrl),
                whiten_mat=None,
                std=std,
            )
            continue

        # sphering
        ss = _accumulate_covariance(idx_arr, mean64, fetch_rows, chunk_size)
        cov = ss / max(len(ctrl) - 1, 1)
        whiten = _whiten_matrix_from_cov(cov, ridge_multiplier=ridge_multiplier)
        stats[batch_id] = BatchCorrectionStats(
            batch_id=batch_id,
            mean=mean,
            n_controls=len(ctrl),
            whiten_mat=whiten,
            std=None,
        )

    info = {
        "batches": sorted(stats.keys()),
        "controls_per_batch": controls_per_batch,
        "method": method,
        "batch_metadata_key": batch_metadata_key,
    }
    return stats, info


def apply_sphering_correction(
    x: np.ndarray,
    mean: np.ndarray,
    whiten_mat: np.ndarray,
) -> np.ndarray:
    """Apply sphering to rows of *x* (shape (N, D))."""
    y = x.astype(np.float32, copy=False) - mean.astype(np.float32, copy=False).reshape(1, -1)
    return (y @ whiten_mat.T).astype(np.float32, copy=False)


def apply_zscore_correction(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Per-feature z-score using control mean/std for this batch."""
    y = x.astype(np.float32, copy=False) - mean.astype(np.float32, copy=False).reshape(1, -1)
    s = std.astype(np.float32, copy=False).reshape(1, -1)
    s = np.maximum(s, 1e-12)
    return (y / s).astype(np.float32, copy=False)


def apply_batch_correction_rows(
    x: np.ndarray,
    row_batch_ids: Sequence[str],
    stats_by_batch: Mapping[str, BatchCorrectionStats],
    method: MethodName,
) -> np.ndarray:
    """Correct each row using the batch id for that row (same order as *x*)."""
    if len(row_batch_ids) != x.shape[0]:
        raise ValueError("row_batch_ids length must match number of rows in x.")
    out = np.empty_like(x, dtype=np.float32)
    by_batch: dict[str, list[int]] = defaultdict(list)
    for i, bid in enumerate(row_batch_ids):
        by_batch[bid].append(i)
    for bid, row_idxs in by_batch.items():
        st = stats_by_batch.get(bid)
        if st is None:
            raise KeyError(f"No batch stats for batch_id={bid!r}. Known: {sorted(stats_by_batch)}")
        sub = x[row_idxs]
        if method == "sphering":
            if st.whiten_mat is None:
                raise ValueError(f"Missing whiten_mat for batch {bid!r}")
            out[row_idxs] = apply_sphering_correction(sub, st.mean, st.whiten_mat)
        else:
            if st.std is None:
                raise ValueError(f"Missing std for batch {bid!r}")
            out[row_idxs] = apply_zscore_correction(sub, st.mean, st.std)
    return out


def row_batch_ids_from_metadata(
    metadata: Sequence[Mapping[str, Any] | dict[str, Any]],
    batch_metadata_key: str,
    indices: np.ndarray | Sequence[int] | None = None,
) -> list[str]:
    """Batch id string per row for the given logical indices (default: all rows)."""
    idx_list = range(len(metadata)) if indices is None else list(indices)
    out: list[str] = []
    for i in idx_list:
        meta = metadata[i] if i < len(metadata) else {}
        if not isinstance(meta, dict):
            meta = {}
        val = get_metadata_value_from_dict(meta, batch_metadata_key)
        out.append(stable_batch_id(val))
    return out


def build_property_fetch_fn(
    properties: Sequence[Mapping[str, Any] | dict[str, Any]],
    keys: Sequence[str],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return fetch_rows(indices) -> (len(indices), len(keys)) from property dicts."""

    def fetch_rows(indices_arr: np.ndarray) -> np.ndarray:
        rows = []
        for i in np.asarray(indices_arr, dtype=np.int64).tolist():
            prop = properties[i] if i < len(properties) else {}
            if not isinstance(prop, dict):
                prop = {}
            row_vals = [float(prop.get(k, np.nan)) for k in keys]
            if not all(np.isfinite(v) for v in row_vals):
                raise ValueError(
                    f"Non-finite property values at row {i} for keys {list(keys)}; "
                    "fix missing values before batch correction."
                )
            rows.append(row_vals)
        return np.asarray(rows, dtype=np.float32)

    return fetch_rows
