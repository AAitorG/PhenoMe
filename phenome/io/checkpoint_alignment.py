"""
Shared checkpoint alignment utilities for metadata vs path-based matching.

Used by both the embedding pipeline and the properties mixin to determine
which items are already in a checkpoint and which need processing.
"""

from collections.abc import Callable
from typing import Any

from .._logging import get_logger
from ..core import metadata_to_stable_key

logger = get_logger(__name__)


def get_already_committed_metadata_keys(
    ckpt_metadata: list[dict[str, Any]],
    n_committed: int,
) -> set[str]:
    """Extract set of metadata stable keys for first n_committed rows in checkpoint."""
    keys: set[str] = set()
    for i in range(min(n_committed, len(ckpt_metadata))):
        try:
            meta = ckpt_metadata[i] if isinstance(ckpt_metadata[i], dict) else {}
            keys.add(metadata_to_stable_key(meta))
        except ValueError:
            pass
    return keys


def get_already_committed_paths(
    ckpt_paths: list[str],
    n_committed: int,
) -> set[str]:
    """Extract set of path representations for first n_committed rows in checkpoint."""
    return set(ckpt_paths[:n_committed])


def get_requested_metadata_keys(
    items: list[dict[str, Any]],
) -> set[str] | None:
    """Build set of metadata stable keys from items. Returns None if any item fails."""
    try:
        return {metadata_to_stable_key(d.get("metadata", {})) for d in items}
    except ValueError:
        return None


def get_requested_paths(
    items: list[dict[str, Any]],
    path_repr_fn: Callable[[Any], str],
) -> set[str]:
    """Build set of path representations from items."""
    return {path_repr_fn(d["file_path"]) for d in items}


def filter_items_not_in_checkpoint(
    items: list[dict[str, Any]],
    already_meta_keys: set[str] | None = None,
    already_paths: set[str] | None = None,
    path_repr_fn: Callable[[Any], str] | None = None,
) -> list[dict[str, Any]]:
    """Filter items that are not already in the checkpoint.

    Uses metadata keys when already_meta_keys is provided and non-empty;
    otherwise uses path-based filtering with already_paths and path_repr_fn.

    Args:
        items: List of dicts with 'file_path' and optionally 'metadata'.
        already_meta_keys: Set of metadata stable keys already in checkpoint.
        already_paths: Set of path reprs already in checkpoint (used when metadata fails).
        path_repr_fn: Function to convert file_path to comparable string.

    Returns:
        List of items that need processing.
    """
    if already_meta_keys is not None and already_meta_keys:
        result = []
        for d in items:
            try:
                key = metadata_to_stable_key(d.get("metadata", {}))
                if key not in already_meta_keys:
                    result.append(d)
            except ValueError:
                result.append(d)  # Include if metadata invalid
        return result

    if already_paths is not None and path_repr_fn is not None:
        return [d for d in items if path_repr_fn(d["file_path"]) not in already_paths]

    return list(items)


def compute_path_alignment(
    image_paths: list[str | list[str]],
    ckpt_paths: list[str],
    n_committed: int,
    path_repr_fn: Callable[[Any], str],
) -> tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]:
    """Compute path-based alignment for properties checkpoint resume.

    Used when metadata-based matching is not available.

    Args:
        image_paths: List of image paths (str or list of str per image).
        ckpt_paths: Paths from checkpoint (primary path per row).
        n_committed: Number of rows committed in checkpoint.
        path_repr_fn: Function to get comparable path string (e.g. primary_path).

    Returns:
        Tuple of (path_to_ckpt_idx, paths_with_data, paths_to_compute, identifier_list).
        paths_to_compute is [(index, path), ...] for images needing processing.
    """
    path_to_ckpt_idx: dict[str, int] = {p: i for i, p in enumerate(ckpt_paths)}
    primary_list = [path_repr_fn(p) for p in image_paths]
    paths_with_data: set[str] = {
        pp for pp in primary_list if pp in path_to_ckpt_idx and path_to_ckpt_idx[pp] < n_committed
    }
    paths_to_compute: list[tuple[int, str | list[str]]] = [
        (i, p) for i, p in enumerate(image_paths) if path_repr_fn(p) not in paths_with_data
    ]
    return path_to_ckpt_idx, paths_with_data, paths_to_compute, primary_list


def compute_metadata_alignment(
    image_paths: list[str | list[str]],
    image_metadata: list[dict[str, Any]],
    ckpt_paths: list[str],
    ckpt_metadata: list[dict[str, Any]],
    n_committed: int,
    path_repr_fn: Callable[[Any], str],
) -> tuple[
    dict[str, int],
    set[str],
    list[tuple[int, str | list[str]]],
    list[str],
]:
    """Compute metadata-based alignment for checkpoint resume.

    Prefer this over path-based when metadata provides atomic identifiers.

    Args:
        image_paths: List of image paths.
        image_metadata: Metadata dict per image (same order as image_paths).
        ckpt_paths: Paths from checkpoint (for fallback).
        ckpt_metadata: Metadata from checkpoint.
        n_committed: Number of rows committed.
        path_repr_fn: For fallback to path-based alignment.

    Returns:
        Tuple of (key_to_ckpt_idx, keys_with_data, paths_to_compute, identifier_list).
    """
    meta_key_to_ckpt_idx: dict[str, int] = {}
    for i, meta in enumerate(ckpt_metadata):
        try:
            key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
            meta_key_to_ckpt_idx[key] = i
        except ValueError:
            pass

    identifiers_with_data: set[str] = get_already_committed_metadata_keys(
        ckpt_metadata, n_committed
    )

    identifier_list: list[str] = []
    paths_to_compute: list[tuple[int, str | list[str]]] = []
    for i, (path, meta) in enumerate(zip(image_paths, image_metadata, strict=False)):
        try:
            key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
            identifier_list.append(key)
            if key not in identifiers_with_data:
                paths_to_compute.append((i, path))
        except ValueError:
            logger.warning(
                "Metadata lacks atomic identifiers. "
                "Falling back to path-based matching (not portable).",
            )
            return compute_path_alignment(image_paths, ckpt_paths, n_committed, path_repr_fn)

    return meta_key_to_ckpt_idx, identifiers_with_data, paths_to_compute, identifier_list


def merge_metadata_from_current_run(
    ckpt_paths: list[Any],
    ckpt_metadata: list[dict[str, Any]],
    current_requested_data: list[dict[str, Any]],
    path_repr_fn: Callable[[Any], str],
) -> list[dict[str, Any]]:
    """Merge checkpoint metadata with current run metadata so new keys are not lost.

    When the checkpoint was created with a file_df that had fewer metadata columns
    than the current file_df, this enriches each row with the extra keys from the
    current run. For each row, result[i] = {**ckpt_metadata[i], **current_meta}.
    Current run keys overwrite when both have the same key.

    Matching is by metadata stable key first, then by path. If no current item
    matches, the checkpoint metadata is returned unchanged for that row.

    Args:
        ckpt_paths: Paths from checkpoint (same order and length as ckpt_metadata).
        ckpt_metadata: Metadata dicts from checkpoint.
        current_requested_data: List of dicts with 'file_path' and 'metadata' from
            the current run's file_df (e.g. full list before filtering by checkpoint).
        path_repr_fn: Function to get comparable path string for matching.

    Returns:
        List of merged metadata dicts, same length as ckpt_metadata.
    """
    n = len(ckpt_metadata)
    if n == 0:
        return []
    if not current_requested_data:
        return list(ckpt_metadata)

    # Build lookups from current run: key -> metadata, path_repr -> metadata
    by_key: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for d in current_requested_data:
        meta = d.get("metadata") or {}
        if isinstance(meta, dict):
            try:
                key = metadata_to_stable_key(meta)
                by_key[key] = meta
            except ValueError:
                pass
        path_repr = path_repr_fn(d.get("file_path", ""))
        by_path[path_repr] = meta if isinstance(meta, dict) else {}

    merged: list[dict[str, Any]] = []
    for i in range(n):
        ckpt_meta = ckpt_metadata[i] if isinstance(ckpt_metadata[i], dict) else {}
        current_meta: dict[str, Any] | None = None
        try:
            key = metadata_to_stable_key(ckpt_meta)
            current_meta = by_key.get(key)
        except ValueError:
            pass
        if current_meta is None and i < len(ckpt_paths):
            path_repr = path_repr_fn(ckpt_paths[i])
            current_meta = by_path.get(path_repr)
        if current_meta is None:
            current_meta = {}
        # Checkpoint base, then overlay current so new columns and updates are kept
        merged.append({**ckpt_meta, **current_meta})
    return merged


def rebase_paths_from_current_run(
    ckpt_paths: list[Any],
    ckpt_metadata: list[dict[str, Any]],
    current_requested_data: list[dict[str, Any]],
    path_repr_fn: Callable[[Any], str],
) -> list[Any]:
    """Replace checkpoint paths with file_df paths when loading from checkpoint.

    When loading a checkpoint on a different machine or after moving the dataset,
    checkpoint paths (relative or from the old storage root) may be invalid.
    This function matches each checkpoint row to current_requested_data by
    metadata stable key (or path as fallback) and uses the current run's
    file_path so results.img_path points to valid files on this machine.

    Args:
        ckpt_paths: Paths from checkpoint (same order and length as ckpt_metadata).
        ckpt_metadata: Metadata dicts from checkpoint.
        current_requested_data: List of dicts with 'file_path' and 'metadata' from
            the current run's file_df.
        path_repr_fn: Function to get comparable path string for matching.

    Returns:
        List of paths (str or list of str per image), same length as ckpt_paths.
        Uses file_path from current_requested_data when a match is found.
    """
    n = len(ckpt_paths)
    if n == 0:
        return []
    if not current_requested_data:
        return list(ckpt_paths)

    # Build lookups: metadata key -> item (with file_path), path_repr -> item
    by_key: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for d in current_requested_data:
        meta = d.get("metadata") or {}
        if isinstance(meta, dict):
            try:
                key = metadata_to_stable_key(meta)
                by_key[key] = d
            except ValueError:
                pass
        path_repr = path_repr_fn(d.get("file_path", ""))
        by_path[path_repr] = d

    result: list[Any] = []
    for i in range(n):
        ckpt_meta = ckpt_metadata[i] if isinstance(ckpt_metadata[i], dict) else {}
        current_item: dict[str, Any] | None = None
        try:
            key = metadata_to_stable_key(ckpt_meta)
            current_item = by_key.get(key)
        except ValueError:
            pass
        if current_item is None and i < len(ckpt_paths):
            path_repr = path_repr_fn(ckpt_paths[i])
            current_item = by_path.get(path_repr)
        if current_item is not None:
            fpath = current_item.get("file_path")
            if fpath is not None:
                result.append(fpath)
                continue
        result.append(ckpt_paths[i])
    return result
