"""
Property matrix construction and NaN handling.

Provides _get_property_matrix, _normalize_property_matrix, _handle_nan_matrix,
and _detect_nan_properties. Used by PhenoMeProperties and PhenoMeAnalysis.

``handle_nans`` modes: ``filter`` (drop rows), ``drop_columns`` (drop NaN
properties), ``impute``, ``warn``, ``keep``.
"""

from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..._logging import get_logger
from ...core.pipeline_results import PhenoMeResults

logger = get_logger(__name__)


def _coerce_property_value(value: Any) -> float:
    """Return a scalar float property value, or NaN when absent/non-numeric."""
    if value is None:
        return np.nan
    try:
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except (IndexError, TypeError, ValueError):
        return np.nan


def _is_nan_property_value(value: Any) -> bool:
    """Return whether a property value is explicitly missing/NaN."""
    if value is None:
        return True
    try:
        val_f = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except IndexError:
        return True
    except (TypeError, ValueError):
        return False
    return bool(np.isnan(val_f))


def _property_value_at(
    properties_list: list[dict[str, Any]],
    index: int,
    key: str,
) -> float:
    """Fetch one property value while preserving image/property row alignment."""
    if index >= len(properties_list):
        return np.nan
    props = properties_list[index]
    if not isinstance(props, dict):
        return np.nan
    return _coerce_property_value(props.get(key, np.nan))


def _discover_property_keys(properties_list: list[dict[str, Any]]) -> list[str]:
    """Discover property keys across all rows, including after leading empty rows."""
    seen: dict[str, None] = {}
    for props in properties_list:
        if isinstance(props, dict):
            for key in props:
                if key not in seen:
                    seen[key] = None
    return list(seen)


def detect_nan_properties(
    results: PhenoMeResults,
    property_keys: list[str] | None = None,
    indices: list[int] | None = None,
    count_missing_key_as_nan: bool = True,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Detect which properties contain NaNs and count occurrences.

    Args:
        results: Pipeline results with 'properties' and 'img_path'.
        property_keys: Property keys to check. If None, uses all keys found across rows.
        indices: Image indices to check. If None, checks all images.
        count_missing_key_as_nan: If True (default), a missing key on a row counts as NaN.
            If False, only explicit NaN / None numeric values count (absent keys are ignored).

    Returns:
        Tuple of (nan_counts, nan_indices).
    """
    properties_list = results.properties
    if property_keys is None:
        keys = sorted(_discover_property_keys(properties_list)) if properties_list else []
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
            if _is_nan_property_value(props[key]):
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
    handle_nans: Literal["filter", "impute", "warn", "keep", "drop_columns"] = "filter",
) -> tuple[np.ndarray, list[int], list[str], dict[str, Any] | None]:
    """Build property matrix from numeric properties.

    Args:
        results: Pipeline results with 'properties' and 'img_path'.
        cache: Mutable cache dict for StandardScaler (key '_property_norm_cache').
        indices: Image indices to include. If None, uses all images.
        property_keys: Property keys to use. If None, auto-detects.
        normalize: Whether to normalize using StandardScaler.
        handle_nans: How to handle NaN values.
            ``filter`` drops samples (rows) with any NaN.
            ``drop_columns`` drops properties (columns) with any NaN and keeps samples.
            ``impute`` / ``warn`` / ``keep`` leave sample count unchanged.

    Returns:
        Tuple of (matrix, valid_indices, used_keys, updated_cache).
    """
    properties_list = results.properties

    if property_keys is None:
        if not properties_list:
            raise ValueError("No properties available. Run compute_properties() first.")
        if not isinstance(properties_list[0], dict):
            raise ValueError(f"Properties must be dicts, got: {type(properties_list[0])}")
        keys = _discover_property_keys(properties_list)
    else:
        keys = property_keys

    if not keys:
        raise ValueError("No numeric properties available for 'properties' source.")

    if indices is None:
        indices = list(range(results.n_images))

    n_images = results.n_images

    def extract_property_array(key: str) -> np.ndarray:
        return np.array(
            [_property_value_at(properties_list, i, key) for i in range(n_images)],
            dtype=np.float32,
        )

    arrays = [extract_property_array(k) for k in keys]
    matrix = np.stack([arr[indices] for arr in arrays], axis=1).astype(np.float32)

    # Drop NaN columns before normalize so the scaler matches the returned keys.
    if handle_nans == "drop_columns":
        matrix, indices, keys = handle_nan_matrix(matrix, indices, keys, handle_nans)
        if matrix.size == 0 or matrix.shape[1] == 0:
            return matrix.reshape(len(indices), 0), indices, keys, cache
        if normalize:
            matrix, cache = normalize_property_matrix(matrix, keys, results, cache)
        return matrix, indices, keys, cache

    if normalize:
        matrix, cache = normalize_property_matrix(matrix, keys, results, cache)

    matrix, valid_indices, keys = handle_nan_matrix(matrix, indices, keys, handle_nans)

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
                [_property_value_at(properties_list, i, key) for i in range(n_samples)],
                dtype=np.float32,
            )

        all_arrays = [extract_array(k) for k in keys]
        all_matrix = np.stack([arr[all_indices] for arr in all_arrays], axis=1).astype(np.float32)

        # Fit per-column on finite values so incomplete rows still contribute
        # to each property's mean/scale (needed for handle_nans='keep'/pairwise).
        n_features = all_matrix.shape[1]
        means = np.zeros(n_features, dtype=np.float64)
        scales = np.ones(n_features, dtype=np.float64)
        vars_ = np.ones(n_features, dtype=np.float64)
        zero_names: list[str] = []
        for j in range(n_features):
            col = all_matrix[:, j]
            ok = np.isfinite(col)
            if ok.sum() == 0:
                continue
            # Match StandardScaler (population std, ddof=0).
            m = float(np.mean(col[ok]))
            v = float(np.var(col[ok]))
            s = float(np.sqrt(v)) if v > 0 else 0.0
            means[j] = m
            if s < 1e-12:
                scales[j] = 1.0
                vars_[j] = 1.0
                zero_names.append(keys[j])
            else:
                scales[j] = s
                vars_[j] = v

        if zero_names:
            logger.warning(
                "Replacing %d zero-variance column(s) with scale=1 to avoid inf/NaN: %s",
                len(zero_names),
                zero_names[:10],
            )

        scaler = StandardScaler()
        scaler.mean_ = means
        scaler.scale_ = scales
        scaler.var_ = vars_
        scaler.n_features_in_ = n_features
        scaler.feature_names_in_ = None
        scaler.n_samples_seen_ = int(np.isfinite(all_matrix).any(axis=1).sum())

        cache = {
            "keys": tuple(keys),
            "n_samples": n_samples,
            "scaler": scaler,
        }

    assert cache is not None, "cache must be set in cache_valid or above"
    scaler = cache["scaler"]
    # Transform finite cells per column so rows with NaNs in other properties
    # still receive z-scored values (pairwise-complete correlation under 'keep').
    matrix_normalized = matrix.copy()
    for j in range(matrix.shape[1]):
        ok = np.isfinite(matrix[:, j])
        if not ok.any():
            continue
        matrix_normalized[ok, j] = (matrix[ok, j] - scaler.mean_[j]) / scaler.scale_[j]
    matrix = matrix_normalized

    return matrix, cache


def handle_nan_matrix(
    matrix: np.ndarray,
    indices: list[int],
    keys: list[str],
    handle_nans: Literal["filter", "impute", "warn", "keep", "drop_columns"],
) -> tuple[np.ndarray, list[int], list[str]]:
    """Handle NaN values in property matrix.

    Returns:
        Tuple of (matrix, valid_indices, used_keys). ``used_keys`` may shrink when
        ``handle_nans='drop_columns'``.
    """
    has_nans = bool(np.isnan(matrix).any()) if matrix.size else False
    if not has_nans:
        return matrix, indices, keys

    nan_counts = np.isnan(matrix).sum(axis=0)
    nan_props = {keys[i]: int(nan_counts[i]) for i in range(len(keys)) if nan_counts[i] > 0}

    if handle_nans == "drop_columns":
        col_ok = ~np.isnan(matrix).any(axis=0)
        dropped = [keys[i] for i in range(len(keys)) if not col_ok[i]]
        kept_keys = [keys[i] for i in range(len(keys)) if col_ok[i]]
        matrix = matrix[:, col_ok] if matrix.ndim == 2 else matrix
        if dropped:
            logger.warning(
                "Dropped %d properties with NaN values (kept %d samples, %d properties). "
                "NaN counts: %s",
                len(dropped),
                len(indices),
                len(kept_keys),
                {k: nan_props[k] for k in dropped},
            )
        return matrix, indices, kept_keys

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
        return matrix, valid_indices, keys

    if handle_nans == "impute":
        matrix = np.nan_to_num(matrix, nan=0.0)
        logger.warning(
            "Imputed NaN values with 0.0 (this biases distances and correlations "
            "for missing-at-random data; consider 'filter' instead). NaN counts: %s",
            nan_props,
        )
        return matrix, indices, keys

    if handle_nans == "keep":
        return matrix, indices, keys

    logger.warning(
        "Matrix contains NaN values. NaN counts: %s. May cause errors in ML algorithms.",
        nan_props,
    )
    return matrix, indices, keys
