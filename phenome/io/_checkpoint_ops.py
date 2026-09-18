"""
Low-level HDF5 operations for checkpoint read/write.

Extracted from checkpoint.py for separation of concerns.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

import h5py
import numpy as np

from ..core._processing_pipeline_persist import (
    _printed_step_fields,
    coalesce_pipeline_payload,
    slim_pipeline_payload,
)

# Constants for HDF5 schema (must match checkpoint.py)
_GZIP_LEVEL = 4
_PROP_CHUNK = 256
_VL_STR = h5py.special_dtype(vlen=str)
_METADATA_EXCLUDE_KEYS = frozenset({"mask_path"})
_PHENOME_DTYPE_ATTR = "phenome_dtype"
_BOOL_MISSING = np.int8(-1)


def _property_values_for_key(prop_dicts: list[dict[str, Any]], key: str) -> list[Any]:
    vals: list[Any] = []
    for p in prop_dicts:
        if isinstance(p, dict):
            vals.append(p.get(key, np.nan))
        else:
            vals.append(np.nan)
    return vals


def _is_bool_property_column(values: list[Any]) -> bool:
    non_missing = [
        v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    return bool(non_missing) and all(isinstance(v, bool) for v in non_missing)


def _serialize_property_column(
    prop_dicts: list[dict[str, Any]],
    key: str,
) -> tuple[np.ndarray, str | None]:
    """Return (column array, phenome_dtype hint) for HDF5 storage."""
    values = _property_values_for_key(prop_dicts, key)
    if _is_bool_property_column(values):
        col = np.array(
            [
                np.int8(1) if v is True else np.int8(0) if v is False else _BOOL_MISSING
                for v in values
            ],
            dtype=np.int8,
        )
        return col, "bool"
    col = np.array(
        [float(p.get(key, np.nan)) if isinstance(p, dict) else np.nan for p in prop_dicts],
        dtype=np.float32,
    )
    return col, None


def _restore_property_value(raw: Any, phenome_dtype: str | None) -> Any:
    if phenome_dtype == "bool":
        if raw == _BOOL_MISSING or raw is None:
            return np.nan
        return bool(int(raw))
    return float(raw)


def decode(value: Any) -> str:
    """Decode bytes to str if needed."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value) if value is not None else ""


def decode_list(arr: Any) -> list[str]:
    """Decode an HDF5 vlen array of image paths to Python strings."""
    return [decode(v) for v in arr]


def append_vlen_dataset(
    f: h5py.File,
    name: str,
    data: list[str],
    n_old: int,
    n_new_total: int,
) -> None:
    """Create (if needed) and append to a 1-D vlen UTF-8 dataset."""
    if name not in f:
        f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=_VL_STR)
    ds = cast(h5py.Dataset, f[name])
    ds.resize(n_new_total, axis=0)
    ds[n_old:n_new_total] = data


def append_2d_vlen_dataset(
    f: h5py.File,
    name: str,
    rows: list[list[str]],
    n_old: int,
    n_new_total: int,
    n_cols: int,
) -> None:
    """Create (if needed) and append to a 2-D (N, C) vlen UTF-8 dataset."""
    if name not in f:
        f.create_dataset(
            name,
            shape=(0, n_cols),
            maxshape=(None, n_cols),
            dtype=_VL_STR,
        )
    ds = cast(h5py.Dataset, f[name])
    existing_cols = ds.shape[1]
    if existing_cols != n_cols:
        raise ValueError(
            f"Channel count mismatch in '{name}': file has {existing_cols}, got {n_cols}."
        )
    ds.resize(n_new_total, axis=0)
    ds[n_old:n_new_total] = rows


def write_2d_vlen(f: h5py.File, name: str, rows: list[list[str]]) -> None:
    """Write a complete 2-D vlen string dataset."""
    n = len(rows)
    c = max(len(r) for r in rows) if rows else 0
    ds = f.create_dataset(
        name,
        shape=(n, c),
        maxshape=(None, c),
        dtype=_VL_STR,
    )
    ds[:] = rows


def meta_value_to_str(val: Any) -> str:
    """Convert a metadata value to a UTF-8 string for storage."""
    if val is None:
        return ""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return ""
    return str(val)


def append_metadata_group(
    f: h5py.File,
    group_name: str,
    meta_dicts: list[dict[str, Any]],
    n_old: int,
    n_new_total: int,
) -> None:
    """Append a batch of metadata dicts to the metadata group (columnar).

    Keys are sorted before touching HDF5 so dataset creation order in the
    file is deterministic across runs with the same logical metadata.
    """
    grp = f.require_group(group_name)
    all_keys: set[str] = set()
    for m in meta_dicts:
        if isinstance(m, dict):
            all_keys.update(k for k in m if k not in _METADATA_EXCLUDE_KEYS)

    for key in sorted(all_keys):
        col = [meta_value_to_str(m.get(key) if isinstance(m, dict) else None) for m in meta_dicts]

        if key not in grp:
            ds = grp.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=_VL_STR,
                chunks=(_PROP_CHUNK,),
            )
            ds.resize(n_old, axis=0)
        else:
            ds = cast(h5py.Dataset, grp[key])

        cur_size = ds.shape[0]
        ds.resize(n_new_total, axis=0)
        if cur_size < n_old:
            ds[cur_size:n_old] = [""] * (n_old - cur_size)

        ds[n_old:n_new_total] = col

    for key in sorted(grp):
        if key not in all_keys:
            ds = cast(h5py.Dataset, grp[key])
            cur_size = ds.shape[0]
            if cur_size < n_new_total:
                ds.resize(n_new_total, axis=0)
                ds[cur_size:n_new_total] = [""] * (n_new_total - cur_size)


def write_metadata_group(
    f: h5py.File,
    group_name: str,
    meta_dicts: list[dict[str, Any]],
    n: int,
) -> None:
    """Write complete metadata as a columnar group.

    ``n`` is the authoritative row count for the file; ``meta_dicts`` must
    match it exactly so columns stay aligned with paths/embeddings.
    """
    if len(meta_dicts) != n:
        raise ValueError(f"write_metadata_group: expected {n} rows, got {len(meta_dicts)}.")
    grp = f.create_group(group_name)
    all_keys: set[str] = set()
    for m in meta_dicts:
        if isinstance(m, dict):
            all_keys.update(k for k in m if k not in _METADATA_EXCLUDE_KEYS)
    for key in sorted(all_keys):
        col = [meta_value_to_str(m.get(key) if isinstance(m, dict) else None) for m in meta_dicts]
        grp.create_dataset(key, data=col, dtype=_VL_STR, maxshape=(None,))


def restore_meta_type(v: str) -> Any:
    """Try to restore the original Python type of a metadata string value."""
    try:
        i = int(v)
        if str(i) == v:
            return i
    except (ValueError, OverflowError):
        pass
    try:
        return float(v)
    except (ValueError, OverflowError):
        pass
    return v


def read_metadata_group(
    grp: h5py.Group,
    n: int,
) -> list[dict[str, Any]]:
    """Read a columnar metadata group back into a list of dicts."""
    result: list[dict[str, Any]] = [{} for _ in range(n)]
    for key in grp:
        ds = cast(h5py.Dataset, grp[key])
        actual = min(n, ds.shape[0])
        values = [decode(v) for v in ds[:actual]]
        for i, v in enumerate(values):
            if v:
                result[i][key] = restore_meta_type(v)
    return result


def append_properties_group(
    f: h5py.File,
    group_name: str,
    prop_dicts: list[dict[str, Any]],
    n_old: int,
    n_new_total: int,
) -> None:
    """Append a batch of property dicts to the properties group (columnar).

    Keys are sorted before touching HDF5 so dataset creation order in the
    file is deterministic across runs with the same logical properties.
    """
    grp = f.require_group(group_name)
    all_keys: set[str] = set()
    for p in prop_dicts:
        if isinstance(p, dict):
            all_keys.update(p.keys())

    for key in sorted(all_keys):
        col, phenome_dtype = _serialize_property_column(prop_dicts, key)
        if key not in grp:
            ds = grp.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=col.dtype,
                chunks=(_PROP_CHUNK,),
                compression="gzip",
                compression_opts=_GZIP_LEVEL,
                fillvalue=_BOOL_MISSING if phenome_dtype == "bool" else np.nan,
            )
            if phenome_dtype:
                ds.attrs[_PHENOME_DTYPE_ATTR] = phenome_dtype
            ds.resize(n_old, axis=0)
        else:
            ds = cast(h5py.Dataset, grp[key])

        cur_size = ds.shape[0]
        ds.resize(n_new_total, axis=0)
        if cur_size < n_old:
            fill_val = _BOOL_MISSING if phenome_dtype == "bool" else np.nan
            fill = np.full(n_old - cur_size, fill_val, dtype=col.dtype)
            ds[cur_size:n_old] = fill

        ds[n_old:n_new_total] = col

    for key in sorted(grp):
        if key not in all_keys:
            ds = cast(h5py.Dataset, grp[key])
            cur_size = ds.shape[0]
            if cur_size < n_new_total:
                fill = np.full(n_new_total - cur_size, np.nan, dtype=np.float32)
                ds.resize(n_new_total, axis=0)
                ds[cur_size:n_new_total] = fill


def write_properties_group(
    f: h5py.File,
    group_name: str,
    prop_dicts: list[dict[str, Any]],
    n: int,
) -> None:
    """Write complete properties as a columnar group.

    ``n`` is the authoritative row count for the file; ``prop_dicts`` must
    match it exactly so columns stay aligned with paths/embeddings.
    """
    if len(prop_dicts) != n:
        raise ValueError(f"write_properties_group: expected {n} rows, got {len(prop_dicts)}.")
    grp = f.create_group(group_name)
    all_keys: set[str] = set()
    for p in prop_dicts:
        if isinstance(p, dict):
            all_keys.update(p.keys())
    for key in sorted(all_keys):
        col, phenome_dtype = _serialize_property_column(prop_dicts, key)
        ds = grp.create_dataset(
            key,
            data=col,
            maxshape=(None,),
            chunks=(_PROP_CHUNK,),
            compression="gzip",
            compression_opts=_GZIP_LEVEL,
            fillvalue=_BOOL_MISSING if phenome_dtype == "bool" else np.nan,
        )
        if phenome_dtype:
            ds.attrs[_PHENOME_DTYPE_ATTR] = phenome_dtype


def read_properties_group(
    grp: h5py.Group,
    n: int,
) -> list[dict[str, Any]]:
    """Read a columnar properties group back into a list of dicts."""
    result: list[dict[str, Any]] = [{} for _ in range(n)]
    for key in grp:
        ds = cast(h5py.Dataset, grp[key])
        actual = min(n, ds.shape[0])
        values = ds[:actual]
        phenome_dtype = ds.attrs.get(_PHENOME_DTYPE_ATTR)
        if isinstance(phenome_dtype, bytes):
            phenome_dtype = phenome_dtype.decode("utf-8")
        for i, v in enumerate(values):
            result[i][key] = _restore_property_value(v, phenome_dtype)
    return result


def append_internal_group(
    f: h5py.File,
    group_name: str,
    internal_dicts: list[dict[str, Any]],
    n_old: int,
    n_new_total: int,
) -> None:
    """Append internal tracking dicts to a columnar float32 group (same layout as /properties)."""
    append_properties_group(f, group_name, internal_dicts, n_old, n_new_total)


def write_internal_group(
    f: h5py.File,
    group_name: str,
    internal_dicts: list[dict[str, Any]],
    n: int,
) -> None:
    """Write a complete internal group (one row per image, same as write_properties_group)."""
    write_properties_group(f, group_name, internal_dicts, n)


def read_internal_group(
    grp: h5py.Group,
    n: int,
) -> list[dict[str, Any]]:
    """Read a columnar internal group back into a list of dicts (same as properties)."""
    return read_properties_group(grp, n)


def truncate_group_datasets(grp: h5py.Group, n: int) -> None:
    """Truncate all datasets in group to at most n rows."""
    for key in grp:
        ds = cast(h5py.Dataset, grp[key])
        if ds.shape[0] > n:
            ds.resize(n, axis=0)


_PIPELINE_GROUP = "processing_pipeline"
_PIPELINE_SEGMENTS = ("embeddings", "properties")
_PIPELINE_STEP_WIDTH = 2
_PIPELINE_STEP_NAME_RE = re.compile(r"^(?:step[\s_]+)?(\d+)$", re.IGNORECASE)


def _h5_attr_text(value: Any) -> str | None:
    """Decode an HDF5 attribute to text, or None if empty."""
    if value is None:
        return None
    text = decode(value).strip()
    return text or None


def _pipeline_step_group_name(index: int) -> str:
    """HDF5 group name for a 0-based step (``step_01`` is first)."""
    return f"step_{index + 1:0{_PIPELINE_STEP_WIDTH}d}"


def _pipeline_step_sort_key(name: str) -> int | None:
    """Return the sort index for a step group name, or None if not a step."""
    match = _PIPELINE_STEP_NAME_RE.fullmatch(str(name).strip())
    if not match:
        return None
    return int(match.group(1))


def _write_pipeline_steps(grp: h5py.Group, steps: list[dict[str, str]]) -> None:
    """Write ``step_01``, ``step_02``, … groups with title/detail attributes."""
    for i, step in enumerate(steps):
        child = grp.create_group(_pipeline_step_group_name(i))
        title = step.get("title")
        detail = step.get("detail")
        kind = step.get("kind")
        if title:
            child.attrs["title"] = title
        if detail:
            child.attrs["detail"] = detail
        if kind:
            child.attrs["kind"] = kind


def _read_pipeline_steps(grp: h5py.Group) -> list[dict[str, str]] | None:
    """Read ``step_NN`` groups (``step_1``, ``step 1``, and ``0`` still load)."""
    named = [
        (key, str(name)) for name in grp if (key := _pipeline_step_sort_key(str(name))) is not None
    ]
    if not named:
        return None
    steps: list[dict[str, str]] = []
    for _, name in sorted(named, key=lambda item: item[0]):
        item = grp[name]
        if not isinstance(item, h5py.Group):
            continue
        step = _printed_step_fields(
            {
                "title": _h5_attr_text(item.attrs.get("title")),
                "detail": _h5_attr_text(item.attrs.get("detail")),
                "kind": _h5_attr_text(item.attrs.get("kind")),
            }
        )
        if step:
            steps.append(step)
    return steps or None


def _read_pipeline_group(grp: h5py.Group) -> dict[str, Any] | None:
    """Read slim embeddings/properties steps from one HDF5 group."""
    out: dict[str, Any] = {}
    for name in _PIPELINE_SEGMENTS:
        if name not in grp or not isinstance(grp[name], h5py.Group):
            continue
        steps = _read_pipeline_steps(cast(h5py.Group, grp[name]))
        if steps:
            out[name] = steps
    return out or None


def write_pipeline_to_h5(f: h5py.File, payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Write ``/processing_pipeline`` (ordered steps only) or remove it.

    When *payload* has only embeddings or only properties, the missing segment
    is kept from the file if it already exists. Pass ``None`` (or a payload
    with no printable steps) to delete the group.

    Args:
        f: Open HDF5 file.
        payload: Session pipeline, slim step lists, or None to delete the group.

    Returns:
        The slim payload that was written, or None when the group was removed.
    """
    incoming = slim_pipeline_payload(payload)
    if incoming is None:
        if _PIPELINE_GROUP in f:
            del f[_PIPELINE_GROUP]
        return None
    existing = read_pipeline_from_h5(f) if _PIPELINE_GROUP in f else None
    slim = coalesce_pipeline_payload(incoming, existing)
    if _PIPELINE_GROUP in f:
        del f[_PIPELINE_GROUP]
    if not slim:
        return None
    root = f.create_group(_PIPELINE_GROUP)
    for name in _PIPELINE_SEGMENTS:
        steps = slim.get(name)
        if not isinstance(steps, list) or not steps:
            continue
        _write_pipeline_steps(root.create_group(name), steps)
    return slim


def read_pipeline_from_h5(f: h5py.File) -> dict[str, Any] | None:
    """Load ``/processing_pipeline``.

    Args:
        f: Open HDF5 file.

    Returns:
        Slim ``{embeddings, properties}`` step lists, or None.
    """
    if _PIPELINE_GROUP not in f:
        return None
    return _read_pipeline_group(cast(h5py.Group, f[_PIPELINE_GROUP]))
