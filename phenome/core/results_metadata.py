"""
Results metadata access and filtering.

Provides functions to read metadata from pipeline results and filter indices.
Accepts both :class:`PhenoMeResults` instances and legacy dicts.
"""

from __future__ import annotations

from typing import Any


def _metadata_list(results: Any) -> list[dict[str, Any]]:
    """Extract the metadata list from either a PhenoMeResults or a dict."""
    if hasattr(results, "metadata"):
        return list(results.metadata)
    return list(results.get("metadata", []))


def _img_path_list(results: Any) -> list[Any]:
    """Extract the img_path list from either a PhenoMeResults or a dict."""
    if hasattr(results, "img_path"):
        return list(results.img_path)
    return list(results.get("img_path", []))


def get_metadata_value_from_dict(meta_dict: dict[str, Any], key: str) -> Any:
    """Get value from a single metadata dict with case-insensitive key matching.

    Args:
        meta_dict: Single metadata dict (e.g. results.metadata[i]).
        key: Metadata key name (case-insensitive).

    Returns:
        Value or None if not found.
    """
    return _get_value_case_insensitive(meta_dict, key)


def get_metadata_value(results: Any, idx: int, key: str) -> Any:
    """Safely fetch a metadata value for an image index.

    Args:
        results: PhenoMeResults or pipeline results dict.
        idx: Image index.
        key: Metadata key name (case-insensitive).

    Returns:
        Metadata value or None if not found.
    """
    metadata_list = _metadata_list(results)
    if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
        return _get_value_case_insensitive(metadata_list[idx], key)
    return None


def get_all_metadata_keys(results: Any) -> list[str]:
    """Get sorted unique metadata keys present in results.

    Deduplicates by lowercase so 'Drug' and 'drug' yield a single key.
    Uses first-occurrence casing for display.

    Args:
        results: PhenoMeResults or pipeline results dict.

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


def _row_matches_criteria(results: Any, idx: int, normalized: dict[str, Any]) -> bool:
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
    results: Any,
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
    if not filters:
        filtered = indices
    else:
        normalized_filters = {key.lower(): v for key, v in filters.items()}
        filtered = [i for i in indices if _row_matches_criteria(results, i, normalized_filters)]

    if not exclude:
        return filtered

    normalized_exclude = {key.lower(): v for key, v in exclude.items()}
    return [i for i in filtered if not _row_matches_criteria(results, i, normalized_exclude)]


def build_metadata_columns(
    results: Any,
    indices: list[int] | None = None,
    capitalize: bool = False,
    keys: list[str] | None = None,
) -> dict[str, list[Any]]:
    """Build dict of metadata columns suitable for DataFrame creation.

    Args:
        results: PhenoMeResults or pipeline results dict.
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
