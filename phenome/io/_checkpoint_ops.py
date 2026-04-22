"""
Low-level HDF5 operations for checkpoint read/write.

Extracted from checkpoint.py for separation of concerns.
"""

from typing import Any, cast

import h5py
import numpy as np

# Constants for HDF5 schema (must match checkpoint.py)
_GZIP_LEVEL = 4
_PROP_CHUNK = 256
_VL_STR = h5py.special_dtype(vlen=str)
_METADATA_EXCLUDE_KEYS = frozenset({"mask_path"})


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
            if n_old > 0:
                ds[:n_old] = [""] * n_old
        else:
            ds = cast(h5py.Dataset, grp[key])

        ds.resize(n_new_total, axis=0)
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
        col = np.array(
            [float(p.get(key, np.nan)) if isinstance(p, dict) else np.nan for p in prop_dicts],
            dtype=np.float32,
        )
        if key not in grp:
            ds = grp.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(_PROP_CHUNK,),
                compression="gzip",
                compression_opts=_GZIP_LEVEL,
            )
            ds.resize(n_old, axis=0)
            if n_old > 0:
                fill = np.full(n_old, np.nan, dtype=np.float32)
                ds[:n_old] = fill
        else:
            ds = cast(h5py.Dataset, grp[key])

        ds.resize(n_new_total, axis=0)
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
        col = np.array(
            [float(p.get(key, np.nan)) if isinstance(p, dict) else np.nan for p in prop_dicts],
            dtype=np.float32,
        )
        grp.create_dataset(
            key,
            data=col,
            maxshape=(None,),
            chunks=(_PROP_CHUNK,),
            compression="gzip",
            compression_opts=_GZIP_LEVEL,
        )


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
        for i, v in enumerate(values):
            result[i][key] = float(v)
    return result


def truncate_group_datasets(grp: h5py.Group, n: int) -> None:
    """Truncate all datasets in group to at most n rows."""
    for key in grp:
        ds = cast(h5py.Dataset, grp[key])
        if ds.shape[0] > n:
            ds.resize(n, axis=0)
