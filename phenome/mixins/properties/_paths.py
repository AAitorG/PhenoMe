"""
Path resolution for property computation.

Functions to resolve image and mask paths for loading during property computation.
"""

import contextlib
import os
from glob import glob
from typing import Any, cast

import pandas as pd

from ..._logging import get_logger
from ...utils.path_utils import path_repr as _path_repr

logger = get_logger(__name__)


def build_file_df_lookup(file_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build lookup from path_repr to row dict for file_df alignment.

    Used to align property computation with the original file_df (e.g. after
    checkpoint resume or filtered subsets).

    Args:
        file_df: DataFrame with a 'file_path' column (str or list of str).

    Returns:
        Dict mapping path_repr key to full row dict. Skips rows with missing path.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in file_df.iterrows():
        fpath = row.get("file_path")
        if fpath is None or (isinstance(fpath, (list, tuple)) and not fpath):
            continue
        key = _path_repr(fpath)
        lookup[key] = row.to_dict()
    return lookup


def resolve_image_paths(
    img_paths: list,
    file_df_lookup: dict[str, dict[str, Any]],
) -> list[str | list[str]]:
    """Build list of absolute image paths for loading from file_df.

    file_df is the single source of truth. All paths must be present in file_df;
    if a path is not in the lookup, it cannot be used.

    Args:
        img_paths: Raw paths (str or list of str for multi-channel) from results.
        file_df_lookup: Lookup from path_repr to row dict (from build_file_df_lookup).

    Returns:
        List of absolute paths (str or list per image). Same length as img_paths.

    Raises:
        ValueError: If any path is not found in file_df_lookup.
    """
    full_paths: list[str | list[str]] = []
    missing: list[str] = []
    for p in img_paths:
        path_repr_key = _path_repr(p)
        if path_repr_key in file_df_lookup:
            row = file_df_lookup[path_repr_key]
            fpath = row.get("file_path")
            if fpath is not None:
                full_paths.append(list(fpath) if isinstance(fpath, (list, tuple)) else str(fpath))
                continue
        missing.append(path_repr_key[:100] + ("..." if len(path_repr_key) > 100 else ""))

    if missing:
        raise ValueError(
            f"Path(s) not found in file_df; cannot resolve. "
            f"file_df must contain all file paths. "
            f"Missing (first 5): {missing[:5]}"
        )

    return full_paths


def resolve_mask_paths(
    results_img_path: list,
    mask_dir: str | None,
    image_paths: list[str | list[str]],
    image_dir: str | None,
    metadata_list: list[dict[str, Any]] | None = None,
    metadata_config: Any | None = None,
    file_df_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[str | None]:
    """Build list of mask paths matched to image_paths.

    Resolution order: (1) file_df_lookup / metadata mask_path, (2) metadata_config
    (mask_dir + mask_filename_column), (3) mask_dir discovery (glob + basename match).

    Args:
        results_img_path: Original image paths from results (for lookup keys).
        mask_dir: Base dir to search for mask files.
        image_paths: Resolved image paths (same order as results).
        image_dir: Image base dir for mirror-path construction in mask_dir.
        metadata_list: Per-image metadata dicts (may contain mask_path).
        metadata_config: Object with mask_dir, mask_filename_column, get_mask_path().
        file_df_lookup: path_repr -> row dict; row may have mask_path.

    Returns:
        List of mask paths (or None) aligned to image_paths.
    """
    metadata_list = metadata_list or []

    def _get_mask_for_index(i: int) -> str | None:
        path_repr_key = _path_repr(results_img_path[i] if i < len(results_img_path) else "")
        if file_df_lookup and path_repr_key in file_df_lookup:
            row = file_df_lookup[path_repr_key]
            for k, v in row.items():
                if k.lower() == "mask_path" and v and isinstance(v, str):
                    if os.path.isfile(v):
                        return cast(str | None, v)
                    break
        meta = (
            metadata_list[i]
            if i < len(metadata_list) and isinstance(metadata_list[i], dict)
            else {}
        )
        for k, v in meta.items():
            if k.lower() == "mask_path" and v and isinstance(v, str):
                return cast(str | None, v) if os.path.isfile(v) else None
        return None

    # Fast path: no mask_dir and no explicit config → use file_df/metadata only.
    if mask_dir is None and not (
        metadata_config
        and getattr(metadata_config, "mask_dir", None)
        and getattr(metadata_config, "mask_filename_column", None)
    ):
        pre_resolved = [_get_mask_for_index(i) for i in range(len(image_paths))]
        if any(m is not None for m in pre_resolved):
            n_found = sum(1 for m in pre_resolved if m is not None)
            logger.info(
                "Masks: %d/%d images.",
                n_found,
                len(image_paths),
            )
            return pre_resolved
        return [None] * len(image_paths)

    result: list[str | None] = []
    needs_resolution: list[int] = []
    for i in range(len(image_paths)):
        mp = _get_mask_for_index(i)
        if mp is not None:
            result.append(mp)
        else:
            result.append(None)
            needs_resolution.append(i)

    if not needs_resolution:
        n_found = sum(1 for m in result if m is not None)
        logger.info(
            "Masks: %d/%d found.",
            n_found,
            len(image_paths),
        )
        return result

    # Explicit config: metadata_config.get_mask_path(meta) with fallback to mask_dir glob.
    use_explicit = (
        metadata_config
        and getattr(metadata_config, "mask_dir", None)
        and getattr(metadata_config, "mask_filename_column", None)
    )
    if use_explicit and metadata_config is not None:
        explicit_resolved: list[str | None] = []
        for i, _ip in enumerate(image_paths):
            meta = (
                metadata_list[i]
                if i < len(metadata_list) and isinstance(metadata_list[i], dict)
                else {}
            )
            mp = metadata_config.get_mask_path(meta)
            explicit_resolved.append(mp)
        needs_fallback = [i for i, m in enumerate(explicit_resolved) if m is None]
        if needs_fallback and mask_dir:
            mask_fnames = sorted(glob(os.path.join(mask_dir, "**", "*.*"), recursive=True))
            fallback_lookup: dict[str, str] = {}
            for mp in mask_fnames:
                fallback_lookup[os.path.basename(mp)] = mp
                fallback_lookup[mp] = mp
                with contextlib.suppress(ValueError):
                    fallback_lookup[os.path.relpath(mp, mask_dir)] = mp
            for i in needs_fallback:
                ip = image_paths[i]
                ip_basename = ip[0] if isinstance(ip, list) else ip
                matched = fallback_lookup.get(os.path.basename(ip_basename)) or fallback_lookup.get(
                    ip_basename
                )
                if image_dir and matched is None:
                    try:
                        candidate = os.path.join(mask_dir, os.path.relpath(ip_basename, image_dir))
                        if os.path.exists(candidate):
                            matched = candidate
                    except (ValueError, OSError):
                        pass
                explicit_resolved[i] = matched if matched and os.path.exists(matched) else None
        for i in needs_resolution:
            result[i] = explicit_resolved[i]
        n_found = sum(1 for m in result if m is not None)
        logger.info(
            "Masks: %d/%d images matched (%d missing).",
            n_found,
            len(image_paths),
            len(image_paths) - n_found,
        )
        return result

    if mask_dir is None:
        return result

    # Discovery: glob mask_dir, match by basename or relpath from image_dir.
    mask_fnames = sorted(glob(os.path.join(mask_dir, "**", "*.*"), recursive=True))
    logger.info("Found %d mask files in %s", len(mask_fnames), mask_dir)

    lookup: dict[str, str] = {}
    for mp in mask_fnames:
        lookup[os.path.basename(mp)] = mp
        lookup[mp] = mp
        with contextlib.suppress(ValueError):
            lookup[os.path.relpath(mp, mask_dir)] = mp

    result_list: list[str | None] = []
    for ip in image_paths:
        ip_primary = ip[0] if isinstance(ip, (list, tuple)) else ip
        mask_path: str | None = None

        if image_dir is not None:
            try:
                candidate = os.path.join(mask_dir, os.path.relpath(ip_primary, image_dir))
                if os.path.exists(candidate):
                    mask_path = candidate
            except (ValueError, OSError):
                pass

        if mask_path is None:
            mask_path = lookup.get(os.path.basename(ip_primary)) or lookup.get(ip_primary)

        if mask_path and os.path.exists(mask_path):
            result_list.append(mask_path)
        else:
            result_list.append(None)

    for i in needs_resolution:
        result[i] = result_list[i]
    n_found = sum(1 for m in result if m is not None)
    logger.info(
        "Masks: %d/%d images matched (%d missing).",
        n_found,
        len(image_paths),
        len(image_paths) - n_found,
    )
    return result
