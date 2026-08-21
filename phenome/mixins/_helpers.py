"""Internal Implementation: Helper functions for pipeline mixins.

Consolidates utility functions shared across analysis, properties,
and visualization mixins to avoid fragmentation and code duplication.
These helpers focus on data conversion and common mathematical operations.
"""

import numpy as np

from .._logging import get_logger

logger = get_logger(__name__)


def numpy_for_torch(x: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, writable array for :func:`torch.from_numpy`.

    Embeddings and other pipeline arrays may be read-only views (e.g. HDF5, slices).
    Row subsets ``data[indices]`` can also be read-only views. PyTorch requires
    a writable buffer when sharing with NumPy.
    """
    x = np.ascontiguousarray(x)
    if not x.flags.writeable:
        x = x.copy()
    return x


def index_array_for_torch(a: object) -> np.ndarray:
    """Writable integer array for ``tensor[indices]`` and related NumPy index ops.

    PyTorch may call :func:`torch.from_numpy` on advanced-index arrays. Pandas
    :meth:`~pandas.Index.to_numpy` often returns a read-only view, which triggers
    the same non-writable warning.
    """
    return np.array(a, dtype=np.intp, copy=True, order="C")


def map_to_full(values: np.ndarray, valid_indices: list[int], n_total: int) -> np.ndarray:
    """Map values for valid_indices into a full-length NaN array.

    Args:
        values: np.ndarray shape (len(valid_indices),).
        valid_indices: List[int]. Global indices where values apply.
        n_total: Total number of images.

    Returns:
        np.ndarray shape (n_total,), dtype float32. NaN where not in valid_indices.
    """
    full = np.full(n_total, np.nan, dtype=np.float32)
    full[np.asarray(valid_indices, dtype=np.int64)] = values.astype(np.float32)
    return full


# Supported metrics for ordering rows in the embedding-property summary.
ORDER_METRICS: tuple[str, ...] = ("mean_abs", "max_abs", "mean", "std")


def outlier_mask_from_distances(
    dists: np.ndarray,
    method: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (distances, outlier_mask, threshold_value) for z-score or IQR methods."""
    if method == "z-score":
        mu, sigma = np.mean(dists), np.std(dists, ddof=1)
        if sigma == 0:
            return dists, np.zeros_like(dists, dtype=bool), float(mu)
        tv = mu + threshold * sigma
        return dists, dists > tv, float(tv)
    if method == "iqr":
        q1, q3 = np.percentile(dists, 25), np.percentile(dists, 75)
        iqr = q3 - q1
        if iqr == 0:
            return dists, np.zeros_like(dists, dtype=bool), float(q3)
        tv = q3 + threshold * iqr
        return dists, dists > tv, float(tv)
    raise ValueError(f"Unknown method: {method}")


def compute_silhouette_if_valid(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    clustering_method: str = "kmeans",
) -> float | None:
    """Compute silhouette score when cluster count and sample sizes are valid."""
    n_actual_clusters = int(np.nanmax(labels)) + 1 if np.any(~np.isnan(labels)) else 0
    if n_actual_clusters < 2 or len(matrix) < n_actual_clusters:
        return None

    mask = ~np.isnan(labels)
    if clustering_method == "dbscan" and not np.all(mask):
        sil_labels = labels[mask]
        sil_matrix = matrix[mask]
    else:
        sil_labels = labels
        sil_matrix = matrix

    unique_labels = np.unique(sil_labels[~np.isnan(sil_labels)])
    if len(unique_labels) < 2 or len(unique_labels) > len(sil_matrix) - 1:
        logger.warning(
            "Silhouette score skipped: need 2 <= n_clusters <= n_samples-1 (got %d clusters, %d samples).",
            len(unique_labels),
            len(sil_matrix),
        )
        return None
    if not all(np.sum(sil_labels == u) >= 2 for u in unique_labels):
        logger.warning("Silhouette score skipped: need at least 2 samples per cluster.")
        return None

    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(sil_matrix, sil_labels.astype(int)))
    except ValueError as exc:
        logger.warning("Silhouette score failed: %s", exc)
        return None


def _require_embeddings_for_temporal(db: object, embedding_dim: int, results: object) -> None:
    """Raise unless checkpoint or in-memory embeddings cover every current row."""
    if db is not None:
        if embedding_dim <= 0:
            raise ValueError(
                "Checkpoint has no embeddings; cannot add temporal images. "
                "Load an embeddings checkpoint first."
            )
        return
    emb = getattr(results, "embeddings", None)
    n = int(getattr(results, "n_images", 0))
    if emb is None or not isinstance(emb, np.ndarray) or emb.ndim != 2 or emb.shape[0] != n:
        raise ValueError(
            "Embeddings are not loaded in memory (results.embeddings is missing "
            "or does not match n_images). Call load_results(..., lazy_checkpoint=True) "
            "or enter checkpoint_context() before process_temporal_images()."
        )
