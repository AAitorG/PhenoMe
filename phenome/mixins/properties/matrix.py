"""
Property matrix construction and NaN handling.

Provides _get_property_matrix, _normalize_property_matrix, _handle_nan_matrix,
and _detect_nan_properties. Used by PhenoMeProperties and PhenoMeAnalysis.
"""

from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..._logging import get_logger
from ...core.pipeline_results import PhenoMeResults

logger = get_logger(__name__)


def detect_nan_properties(
    results: PhenoMeResults,
    property_keys: list[str] | None = None,
    indices: list[int] | None = None,
    count_missing_key_as_nan: bool = True,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Detect which properties contain NaNs and count occurrences.

    Args:
        results: Pipeline results with 'properties' and 'img_path'.
        property_keys: Property keys to check. If None, uses all from first props dict.
        indices: Image indices to check. If None, checks all images.
        count_missing_key_as_nan: If True (default), a missing key on a row counts as NaN.
            If False, only explicit NaN / None numeric values count (absent keys are ignored).

    Returns:
        Tuple of (nan_counts, nan_indices).
    """
    properties_list = results.properties
    if property_keys is None:
        keys = sorted(properties_list[0].keys()) if properties_list else []
    else:
        keys = property_keys

    if not keys:
        return {}, {}

    if indices is None:
        indices = list(range(results.n_images))

    nan_counts: dict[str, int] = {}
    nan_indices: dict[str, list[int]] = {}

    for key in keys:
        nan_idx_for_key: list[int] = []
        for i in indices:
            if i >= len(properties_list):
                if count_missing_key_as_nan:
                    nan_idx_for_key.append(i)
                continue
            props = properties_list[i]
            if not isinstance(props, dict):
                if count_missing_key_as_nan:
                    nan_idx_for_key.append(i)
                continue
            if key not in props:
                if count_missing_key_as_nan:
                    nan_idx_for_key.append(i)
                continue
            raw = props[key]
            if raw is None:
                nan_idx_for_key.append(i)
                continue
            try:
                val_f = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
            except (TypeError, ValueError):
                continue
            if np.isnan(val_f):
                nan_idx_for_key.append(i)

        nan_counts[key] = len(nan_idx_for_key)
        nan_indices[key] = nan_idx_for_key

    return nan_counts, nan_indices


def warn_if_nan_properties(
    results: PhenoMeResults,
    property_keys: list[str] | None = None,
    indices: list[int] | None = None,
    count_missing_key_as_nan: bool = True,
) -> None:
    """Log a warning if any properties contain NaNs (explicit NaNs, not missing keys by default when aligned with expected keys)."""
    nan_counts, nan_indices = detect_nan_properties(
        results,
        property_keys=property_keys,
        indices=indices,
        count_missing_key_as_nan=count_missing_key_as_nan,
    )
    affected = {k: v for k, v in nan_counts.items() if v > 0}
    if not affected:
        return

    n_total = results.n_images
    samples_affected = (
        len({idx for indices_list in nan_indices.values() for idx in indices_list})
        if nan_indices
        else 0
    )

    logger.warning(
        "NaNs detected in property columns (NaN count per column): %s. "
        "%d/%d images have at least one such NaN. "
        "Use detect_nan_properties() for per-column row indices.",
        affected,
        samples_affected,
        n_total,
    )


def get_property_matrix(
    results: PhenoMeResults,
    cache: dict[str, Any] | None = None,
    indices: list[int] | None = None,
    property_keys: list[str] | None = None,
    normalize: bool = True,
    handle_nans: Literal["filter", "impute", "warn"] = "filter",
) -> tuple[np.ndarray, list[int], list[str], dict[str, Any] | None]:
    """Build property matrix from numeric properties.

    Args:
        results: Pipeline results with 'properties' and 'img_path'.
        cache: Mutable cache dict for StandardScaler (key '_property_norm_cache').
        indices: Image indices to include. If None, uses all images.
        property_keys: Property keys to use. If None, auto-detects.
        normalize: Whether to normalize using StandardScaler.
        handle_nans: How to handle NaN values.

    Returns:
        Tuple of (matrix, valid_indices, used_keys, updated_cache).
    """
    properties_list = results.properties

    if property_keys is None:
        if not properties_list:
            raise ValueError("No properties available. Run compute_properties() first.")
        if not isinstance(properties_list[0], dict):
            raise ValueError(f"Properties must be dicts, got: {type(properties_list[0])}")
        seen: dict[str, None] = {}
        for p in properties_list:
            if isinstance(p, dict):
                for k in p:
                    if k not in seen:
                        seen[k] = None
        keys = list(seen)
    else:
        keys = property_keys

    if not keys:
        raise ValueError("No numeric properties available for 'properties' source.")

    if indices is None:
        indices = list(range(results.n_images))

    def extract_property_array(key: str) -> np.ndarray:
        return np.array(
            [props.get(key, np.nan) for props in properties_list],
            dtype=np.float32,
        )

    arrays = [extract_property_array(k) for k in keys]
    matrix = np.stack([arr[indices] for arr in arrays], axis=1).astype(np.float32)

    if normalize:
        matrix, cache = normalize_property_matrix(matrix, keys, results, cache)

    matrix, valid_indices = handle_nan_matrix(matrix, indices, keys, handle_nans)

    return matrix, valid_indices, keys, cache


def normalize_property_matrix(
    matrix: np.ndarray,
    keys: list[str],
    results: PhenoMeResults,
    cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Normalize property matrix using cached StandardScaler."""
    n_samples = results.n_images
    cache_valid = (
        cache is not None
        and cache.get("keys") == tuple(keys)
        and cache.get("n_samples") == n_samples
        and cache.get("scaler") is not None
    )

    if not cache_valid:
        properties_list = results.properties
        all_indices = list(range(n_samples))

        def extract_array(key: str) -> np.ndarray:
            return np.array(
                [props.get(key, np.nan) for props in properties_list],
                dtype=np.float32,
            )

        all_arrays = [extract_array(k) for k in keys]
        all_matrix = np.stack([arr[all_indices] for arr in all_arrays], axis=1).astype(np.float32)

        scaler = StandardScaler()
        non_nan_mask = ~np.isnan(all_matrix).any(axis=1)
        if non_nan_mask.sum() > 0:
            scaler.fit(all_matrix[non_nan_mask])
            zero_var = scaler.var_ < 1e-12
            if zero_var.any():
                n_zero = int(zero_var.sum())
                zero_names = [keys[i] for i in range(len(keys)) if zero_var[i]]
                logger.warning(
                    "Replacing %d zero-variance column(s) with scale=1 to avoid inf/NaN: %s",
                    n_zero,
                    zero_names[:10],
                )
                scaler.scale_[zero_var] = 1.0
        else:
            scaler.mean_ = np.zeros(all_matrix.shape[1], dtype=np.float32)
            scaler.scale_ = np.ones(all_matrix.shape[1], dtype=np.float32)
            scaler.var_ = np.ones(all_matrix.shape[1], dtype=np.float32)
            scaler.n_features_in_ = all_matrix.shape[1]
            scaler.feature_names_in_ = None

        cache = {
            "keys": tuple(keys),
            "n_samples": n_samples,
            "scaler": scaler,
        }

    assert cache is not None, "cache must be set in cache_valid or above"
    scaler = cache["scaler"]
    non_nan_mask = ~np.isnan(matrix).any(axis=1)
    if non_nan_mask.sum() > 0:
        matrix_normalized = matrix.copy()
        matrix_normalized[non_nan_mask] = scaler.transform(matrix[non_nan_mask])
        matrix = matrix_normalized

    return matrix, cache


def handle_nan_matrix(
    matrix: np.ndarray,
    indices: list[int],
    keys: list[str],
    handle_nans: Literal["filter", "impute", "warn"],
) -> tuple[np.ndarray, list[int]]:
    """Handle NaN values in property matrix."""
    has_nans = np.isnan(matrix).any()
    if not has_nans:
        return matrix, indices

    nan_counts = np.isnan(matrix).sum(axis=0)
    nan_props = {keys[i]: int(nan_counts[i]) for i in range(len(keys)) if nan_counts[i] > 0}

    if handle_nans == "filter":
        valid_mask = ~np.isnan(matrix).any(axis=1)
        matrix = matrix[valid_mask]
        valid_indices = [indices[i] for i in range(len(indices)) if valid_mask[i]]
        if len(indices) - len(valid_indices) > 0:
            logger.warning(
                "Filtered out %d samples with NaN values. NaN counts: %s",
                len(indices) - len(valid_indices),
                nan_props,
            )
        return matrix, valid_indices

    if handle_nans == "impute":
        matrix = np.nan_to_num(matrix, nan=0.0)
        logger.warning(
            "Imputed NaN values with 0.0 (this biases distances and correlations "
            "for missing-at-random data; consider 'filter' instead). NaN counts: %s",
            nan_props,
        )
        return matrix, indices

    logger.warning(
        "Matrix contains NaN values. NaN counts: %s. May cause errors in ML algorithms.",
        nan_props,
    )
    return matrix, indices
