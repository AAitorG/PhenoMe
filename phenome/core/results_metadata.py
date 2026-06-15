"""
Results metadata access and filtering.

Provides functions to read metadata from pipeline results and filter indices.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .._logging import get_logger
from .pipeline_results import PhenoMeResults

logger = get_logger(__name__)


def _metadata_list(results: PhenoMeResults) -> list[dict[str, Any]]:
    return list(results.metadata)


def _img_path_list(results: PhenoMeResults) -> list[Any]:
    return list(results.img_path)


def get_metadata_value_from_dict(meta_dict: dict[str, Any], key: str) -> Any:
    """Get value from a single metadata dict with case-insensitive key matching.

    Args:
        meta_dict: Single metadata dict (e.g. results.metadata[i]).
        key: Metadata key name (case-insensitive).

    Returns:
        Value or None if not found.
    """
    return _get_value_case_insensitive(meta_dict, key)


def get_metadata_value(results: PhenoMeResults, idx: int, key: str) -> Any:
    """Safely fetch a metadata value for an image index.

    Args:
        results: PhenoMeResults instance.
        idx: Image index.
        key: Metadata key name (case-insensitive).

    Returns:
        Metadata value or None if not found.
    """
    metadata_list = _metadata_list(results)
    if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
        return _get_value_case_insensitive(metadata_list[idx], key)
    return None


def get_all_metadata_keys(results: PhenoMeResults) -> list[str]:
    """Get sorted unique metadata keys present in results.

    Deduplicates by lowercase so 'Drug' and 'drug' yield a single key.
    Uses first-occurrence casing for display.

    Args:
        results: PhenoMeResults instance.

    Returns:
        Sorted list of unique metadata keys.
    """
    metadata_list = _metadata_list(results)
    key_by_lower: dict[str, str] = {}
    for meta in metadata_list:
        if isinstance(meta, dict):
            for k in meta:
                kl = k.lower()
                if kl not in key_by_lower:
                    key_by_lower[kl] = k
    return sorted(key_by_lower.values())


def _row_matches_criteria(results: PhenoMeResults, idx: int, normalized: dict[str, Any]) -> bool:
    """Return True if row at idx matches all criteria (for include or exclude)."""
    for key_lower, allowed in normalized.items():
        if allowed is None:
            continue
        value = get_metadata_value(results, idx, key_lower)
        if isinstance(allowed, list):
            if value not in allowed:
                return False
        else:
            if value != allowed:
                return False
    return True


def filter_indices(
    results: PhenoMeResults,
    filters: dict[str, Any | list[Any]] | None = None,
    exclude: dict[str, Any | list[Any]] | None = None,
) -> list[int]:
    """Return indices that satisfy provided metadata filters and exclusions.

    Filters map metadata keys to allowed values. If a value is a list, the
    metadata value must match any element in the list. Exclude uses the same
    structure but removes matching rows. Empty or None filters return all
    indices. Key matching is case-insensitive.

    Order: apply filters first (include only matching), then remove rows
    matching exclude.

    Args:
        results: PhenoMeResults or pipeline results dict with img_path and metadata.
        filters: Optional dict mapping metadata keys to allowed value(s). Each
            value can be a single value (exact match) or a list (match any).
            Example: {'condition': 'Control', 'time': ['24h', '48h']}.
        exclude: Optional dict mapping metadata keys to excluded value(s).
            Same format as filters (single value or list). Can be used together
            with filters; applied after filters. Matching rows are removed.
            Example: {'condition': 'BadBatch', 'plate': 'P1'}.

    Returns:
        List[int]: Image indices that match filters and do not match exclude.

    Example:
        >>> filters = {'condition': ['Control', 'Treatment'], 'time': '24h'}
        >>> exclude = {'condition': 'BadBatch'}
        >>> indices = filter_indices(results, filters, exclude=exclude)
    """
    img_paths = _img_path_list(results)
    n_images = len(img_paths)
    indices = list(range(n_images))

    known_keys = {k.lower() for k in get_all_metadata_keys(results)}

    if not filters:
        filtered = indices
    else:
        normalized_filters = {key.lower(): v for key, v in filters.items()}
        unknown = set(normalized_filters) - known_keys
        if unknown:
            import warnings

            warnings.warn(
                f"Filter key(s) {unknown} not found in metadata. Known keys: {sorted(known_keys)}",
                stacklevel=2,
            )
        filtered = [i for i in indices if _row_matches_criteria(results, i, normalized_filters)]

    if not exclude:
        return filtered

    normalized_exclude = {key.lower(): v for key, v in exclude.items()}
    unknown_ex = set(normalized_exclude) - known_keys
    if unknown_ex:
        import warnings

        warnings.warn(
            f"Exclude key(s) {unknown_ex} not found in metadata. Known keys: {sorted(known_keys)}",
            stacklevel=2,
        )
    return [i for i in filtered if not _row_matches_criteria(results, i, normalized_exclude)]


def build_metadata_columns(
    results: PhenoMeResults,
    indices: list[int] | None = None,
    capitalize: bool = False,
    keys: list[str] | None = None,
) -> dict[str, list[Any]]:
    """Build dict of metadata columns suitable for DataFrame creation.

    Args:
        results: PhenoMeResults instance.
        indices: Optional list of image indices. If None, uses all images.
        capitalize: Whether to capitalize column names.
        keys: Optional list of metadata keys to include. If None, uses all keys.

    Returns:
        Dict mapping column names to value lists (length len(indices)).
    """
    img_paths = _img_path_list(results)
    n_images = len(img_paths)
    indices = indices if indices is not None else list(range(n_images))
    all_keys = get_all_metadata_keys(results) if keys is None else keys
    metadata_list = _metadata_list(results)

    columns: dict[str, list[Any]] = {}
    for key in all_keys:
        col_key = key.capitalize() if capitalize else key
        columns[col_key] = [
            _get_value_case_insensitive(metadata_list[i], key)
            if i < len(metadata_list) and isinstance(metadata_list[i], dict)
            else None
            for i in indices
        ]
    return columns


def _get_value_case_insensitive(meta_dict: dict[str, Any], key: str) -> Any:
    """Get value from dict with case-insensitive key matching."""
    if not isinstance(meta_dict, dict):
        return None
    key_lower = key.lower()
    for k, v in meta_dict.items():
        if k.lower() == key_lower:
            return v
    return None


def _find_result_key(results: PhenoMeResults, col: str) -> str | None:
    """Case-insensitive lookup of *col* in properties then metadata stores."""
    for store_key in ("properties", "metadata"):
        store = getattr(results, store_key, None) or []
        if not store or not isinstance(store[0], dict):
            continue
        for k in store[0]:
            if k.lower() == col.lower():
                return k
    return None


def resolve_result_keys(
    results: PhenoMeResults,
    requested: str | list[str],
    *,
    missing_key_log: str = "Grouping key '%s' not found.",
) -> list[str]:
    """Resolve case-insensitive keys across properties and metadata stores.

    Logs a warning and skips keys that are not found (same behavior as analysis mixins).
    """
    requested_cols = [requested] if isinstance(requested, str) else list(requested)
    found_cols: list[str] = []
    for col in requested_cols:
        found_key = _find_result_key(results, col)
        if found_key:
            found_cols.append(found_key)
        else:
            logger.warning(missing_key_log, col)
    return found_cols


def _get_value_from_stores(
    results: PhenoMeResults,
    idx: int,
    key: str,
    store_order: tuple[str, ...],
) -> Any:
    """Fetch *key* from the first store in *store_order* that has a non-None value."""
    for store_key in store_order:
        store = getattr(results, store_key, None) or []
        if idx >= len(store) or not isinstance(store[idx], dict):
            continue
        val = get_metadata_value_from_dict(store[idx], key)
        if val is not None:
            return val
    return None


def get_result_value(
    results: PhenoMeResults,
    idx: int,
    key: str,
    *,
    prefer: Literal["metadata", "properties"] = "metadata",
) -> Any:
    """Fetch a value from metadata or properties at *idx*.

    Args:
        prefer: ``"metadata"`` checks metadata then properties (outlier grouping).
            ``"properties"`` checks properties then metadata (composite prototypes).
    """
    store_order = ("metadata", "properties") if prefer == "metadata" else ("properties", "metadata")
    return _get_value_from_stores(results, idx, key, store_order)


def collect_metadata_labels(
    results: PhenoMeResults,
    indices: list[int],
    key: str,
) -> np.ndarray:
    """Collect per-sample metadata labels for stratified CV or similar."""
    return np.asarray(
        [get_metadata_value(results, idx, key) for idx in indices],
        dtype=object,
    )


def build_index_group_labels(
    results: PhenoMeResults,
    requested_cols: list[str],
    n_total: int,
) -> tuple[list[Any], list[str]]:
    """Build per-image group labels and resolved column names.

    Returns ``(labels_per_image, found_cols)`` where ``labels_per_image[i]`` is the
    group label for global image index *i*. Single-column groups use raw store values;
    multi-column groups use ``" | "``-joined strings with ``"N/A"`` for missing values.
    """
    found_cols = resolve_result_keys(
        results,
        requested_cols,
        missing_key_log="Column '%s' not found.",
    )
    group_labels: list[Any] = [None] * n_total

    if not found_cols:
        return group_labels, found_cols

    if len(found_cols) == 1:
        k = found_cols[0]
        for store_key in ("properties", "metadata"):
            store = getattr(results, store_key, None) or []
            if not store or not isinstance(store[0], dict) or k not in store[0]:
                continue
            for i, entry in enumerate(store):
                if i < n_total and isinstance(entry, dict):
                    group_labels[i] = entry.get(k)
            break
    else:
        for i in range(n_total):
            row_vals = []
            for k in found_cols:
                val = get_result_value(results, i, k, prefer="properties")
                row_vals.append(str(val) if val is not None else "N/A")
            group_labels[i] = " | ".join(row_vals)

    return group_labels, found_cols


def build_subset_groups(
    results: PhenoMeResults,
    valid_indices: list[int],
    found_cols: list[str],
) -> tuple[dict[Any, np.ndarray], str | list[str] | None]:
    """Map group values to relative row indices within *valid_indices*.

    Used by outlier detection where grouping is over the filtered sample matrix.
    """
    import pandas as pd

    n_samples = len(valid_indices)
    if not found_cols:
        return {"all": np.arange(n_samples)}, None

    group_data: dict[str, list[Any]] = {}
    for k in found_cols:
        group_data[k] = [get_result_value(results, i, k) for i in valid_indices]

    df = pd.DataFrame(group_data)
    groups: dict[Any, np.ndarray] = {}

    if len(found_cols) == 1:
        actual_group_by: str | list[str] | None = found_cols[0]
        for gv, gdf in df.groupby(actual_group_by, dropna=False):
            groups[gv] = gdf.index.to_numpy()
    else:
        actual_group_by = found_cols
        df["_composite_group"] = df[found_cols].fillna("N/A").astype(str).agg(" | ".join, axis=1)
        for gv, gdf in df.groupby("_composite_group", dropna=False):
            groups[gv] = gdf.index.to_numpy()

    return groups, actual_group_by
