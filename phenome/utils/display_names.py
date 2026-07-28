"""Human-readable labels for internal API keys (plots, logs, reports)."""

from __future__ import annotations

_DR_METHOD: dict[str, str] = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
}

_CORRELATION_METHOD: dict[str, str] = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "distance_correlation": "Distance correlation",
    "mutual_info": "Mutual information",
}

_CLUSTERING_METHOD: dict[str, str] = {
    "kmeans": "k-means",
    "dbscan": "DBSCAN",
    "gmm": "GMM",
}


def capitalize_preserve(name: str) -> str:
    """Uppercase the first character only, leaving the rest unchanged.

    Unlike ``str.capitalize``, this preserves channel labels such as
    ``intensity_mean_DAPI`` → ``Intensity_mean_DAPI`` (not ``Intensity_mean_dapi``).
    """
    if not name:
        return name
    return name[0].upper() + name[1:]


def _snake_fallback(key: str) -> str:
    """Title-case words from snake_case for unknown keys."""
    parts = [p for p in key.split("_") if p]
    if not parts:
        return key
    return " ".join(capitalize_preserve(p) for p in parts)


def format_dr_method(method: str) -> str:
    """Display name for dimensionality reduction: ``pca`` → PCA, ``tsne`` → t-SNE, etc."""
    if not method:
        return method
    key = method.strip().lower()
    return _DR_METHOD.get(key, _snake_fallback(key))


def format_correlation_method(method: str) -> str:
    """Display name for correlation metrics (plot titles, progress bars)."""
    if not method:
        return method
    key = method.strip().lower()
    return _CORRELATION_METHOD.get(key, _snake_fallback(key))


def format_clustering_method(method: str) -> str:
    """Display name for clustering algorithms in logs."""
    if not method:
        return method
    key = method.strip().lower()
    return _CLUSTERING_METHOD.get(key, _snake_fallback(key))
