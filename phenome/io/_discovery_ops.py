"""Auxiliary helpers for file discovery (grouping and id uniqueness)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .._logging import get_logger
from ..metadata.base import MetadataBase

logger = get_logger(__name__)


def _raise_on_duplicate_ids(rows: list[dict[str, Any]]) -> None:
    """Fail closed when two discovered samples share the same id."""
    seen: dict[str, Any] = {}
    for row in rows:
        rid = None
        for k, v in row.items():
            if str(k).lower() == "id" and v is not None and str(v).strip():
                rid = str(v)
                break
        if rid is None:
            continue
        if rid in seen:
            raise ValueError(
                f"Duplicate sample id {rid!r} for {row.get('file_path')!r} "
                f"and {seen[rid]!r}. Distinct files must have distinct ids."
            )
        seen[rid] = row.get("file_path")


def _resolve_channel_group_key(rows: list[dict[str, Any]], group_key: Any) -> Any:
    """Pick a sample-level grouping column; never ``channel_index``."""
    if not rows:
        return None
    if group_key not in rows[0]:
        group_key_lower = str(group_key).lower()
        matching = [k for k in rows[0] if k.lower() == group_key_lower]
        if matching:
            group_key = matching[0]
        else:
            sys_keys = {"channel_index", "file_path"}
            group_key = next((k for k in rows[0] if k not in sys_keys), None)
    if group_key is not None and str(group_key).lower() == "channel_index":
        sys_keys = {"channel_index", "file_path"}
        group_key = next((k for k in rows[0] if k not in sys_keys), None)
    return group_key


def _ordered_channel_group(group_id: Any, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort a channel group by integer index and require 0..n-1 with no gaps."""
    int_indices: list[int] = []
    for r in group:
        try:
            int_indices.append(int(r["channel_index"]))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"channel_index must be an integer, got {r['channel_index']!r} "
                f"in group {group_id!r}."
            ) from e
    if len(set(int_indices)) != len(int_indices):
        raise ValueError(f"Duplicate channel_index values in group {group_id!r}: {int_indices}.")
    expected = list(range(len(int_indices)))
    if sorted(int_indices) != expected:
        raise ValueError(
            f"Incomplete channel group {group_id!r}: got indices "
            f"{sorted(int_indices)}, expected {expected} (0..n-1, no gaps)."
        )
    return [r for _, r in sorted(zip(int_indices, group, strict=True))]


def _merge_multichannel_groups(
    rows: list[dict[str, Any]],
    fn: Any,
    metadata_fn: Any,
    data_dir: str | None,
) -> list[dict[str, Any]]:
    """Collapse per-channel files into one row per sample with ordered path lists."""
    group_key = _resolve_channel_group_key(rows, getattr(fn, "group_by", "filename"))
    if group_key is None:
        raise ValueError(
            "Multi-channel files detected (channel_index present) but no grouping "
            "column found. Set group_by on the metadata source (e.g. filename or id)."
        )
    n_files = len(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    merged_rows: list[dict[str, Any]] = []
    for group_id, group in groups.items():
        ordered = _ordered_channel_group(group_id, group)
        first = ordered[0].copy()
        paths_list = [r["file_path"] for r in ordered]
        first["file_path"] = paths_list
        first.pop("channel_index", None)
        meta_src = getattr(metadata_fn, "_metadata_source", metadata_fn)
        if isinstance(meta_src, MetadataBase):
            meta_src.ensure_id(first, paths_list, data_dir)
        merged_rows.append(first)
    logger.info(
        "Grouped %d channel files into %d samples (multi-channel mode).",
        n_files,
        len(merged_rows),
    )
    return merged_rows
