"""
Property and metadata key utilities.

Optimization and stable key generation for checkpoint matching.
"""

import json
from typing import Any

import numpy as np


def optimize_property_types(img_props: dict[str, Any]) -> dict[str, Any]:
    """Reduce memory usage of property values.

    Converts floats to integers if they have no fractional part, and uses
    float16 precision for other floats.

    Args:
        img_props: Dict[str, Any] mapping property names to values. Values may be
            float, int, numpy scalar, or other (passed through unchanged).

    Returns:
        Dict[str, Any]: Same keys as input. Values optimized: float->int if whole,
            float->float16 (as float) otherwise; int preserved; other unchanged.
    """
    # Reduce storage: whole-number floats -> int; fractional -> float16 (as float for JSON)
    optimized = {}
    for k, v in img_props.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                optimized[k] = v
            elif float(v).is_integer():
                optimized[k] = int(v)
            else:
                # Use float32 precision for storage (JSON serializable, avoids
                # float16 quantization artifacts on small values)
                optimized[k] = float(np.float32(v))
        elif isinstance(v, (int, np.integer)):
            optimized[k] = int(v)
        else:
            optimized[k] = v
    return optimized


def metadata_to_stable_key(
    meta: dict[str, Any],
    exclude_keys: set[str] | None = None,
    prefer_id_column: str | None = "id",
) -> str:
    """Build a deterministic, portable key from metadata for matching across devices.

    Paths vary across machines, so we exclude path-dependent keys and use only
    metadata that uniquely identifies the image (e.g. drug, time, crop_name, plate, well).
    The user must ensure the remaining metadata is atomic (uniquely identifies each image).

    When prefer_id_column is set and meta contains that key with a non-path-like value,
    that value is returned directly for simpler, portable matching.

    Args:
        meta: Metadata dict for one image.
        exclude_keys: Additional keys to exclude. Path-like values are always
            excluded regardless of key name (detected heuristically).
        prefer_id_column: If set (default "id"), use this column's value when
            present and non-path-like. Set to None for legacy behavior.

    Returns:
        str: Deterministic JSON string (sort_keys=True) suitable as a stable identifier,
            or the prefer_id_column value when applicable.
            Path-like values and exclude_keys are omitted.

    Raises:
        ValueError: If metadata is empty (cannot uniquely identify the image).
    """
    if not meta or not isinstance(meta, dict):
        raise ValueError(
            "Metadata must be a non-empty dict for stable matching. "
            "Ensure your metadata_fn extracts atomic identifiers (e.g. drug, time, crop_name)."
        )
    if prefer_id_column is not None:
        for k, v in meta.items():
            if str(k).lower() == prefer_id_column.lower() and v is not None:
                v_str = str(v).strip()
                if v_str and not _is_path_like_value(v_str):
                    return v_str
    exclude_keys_set = exclude_keys or set()
    exclude_keys_lower = {str(k).lower() for k in exclude_keys_set}
    # Keep only non-path, non-excluded keys; normalize to lowercase for portability
    filtered = {}
    for k, v in meta.items():
        k_lower = str(k).lower()
        if k_lower in exclude_keys_lower:
            continue
        if _is_path_like_value(v):
            continue
        filtered[k_lower] = _to_json_safe(v)
    if not filtered:
        raise ValueError(
            "Metadata has no non-path keys. Add atomic identifiers (e.g. drug, time, "
            "crop_name, plate, well) for checkpoint matching across devices."
        )
    return json.dumps(filtered, sort_keys=True)


def _is_path_like_value(val: Any) -> bool:
    """Heuristic: value looks like a filesystem path (varies across devices).

    Uses only structural cues—no format lists or key assumptions.
    Path-like if: starts with / or ~, or contains / or \\.
    """
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s:
        return False
    if s.startswith(("/", "~")):
        return True
    return bool("/" in s or "\\" in s)


def _to_json_safe(val: Any) -> Any:
    """Convert value to JSON-serializable form (handles numpy/pandas types).

    Args:
        val: Value of any type (numpy scalar, ndarray, dict, list, etc.).

    Returns:
        JSON-serializable equivalent (int, float, bool, list, dict, or unchanged).
    """
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    if isinstance(val, dict):
        return {k: _to_json_safe(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_json_safe(v) for v in val]
    return val
