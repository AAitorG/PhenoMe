"""Internal Implementation: Orchestration for property computation.

Provides functions for path resolution, checkpoint resume logic,
parallel worker coordination, and result finalization. These functions
are intended for internal use by the `PhenoMeProperties` mixin.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..._logging import get_logger
from ...core import metadata_to_stable_key, optimize_property_types
from ...io import CheckpointManager, align_by_metadata
from ...io.checkpoint_alignment import (
    _build_unique_path_lookup,
    _canonical_path,
    _match_path_index,
    _union_prop_keys_sample,
    build_metadata_key_index,
)
from ...utils.path_utils import primary_path as _primary_path
from ...utils.progress import ProgressCallback, report_progress
from ...utils.property_factories import get_preset_property_functions
from ._paths import build_file_df_lookup, resolve_image_paths, resolve_mask_paths
from ._workers import (
    _determine_channel_range,
    _load_image_and_mask_stacks,
    _should_call_property_function,
    collect_property_names_for_stacks,
    compute_properties_worker,
    format_channel_property_name,
    validate_channel_names,
)

if TYPE_CHECKING:
    from ...core.pipeline_results import PhenoMeResults

logger = get_logger(__name__)


def resolve_properties_paths(
    results: "PhenoMeResults",
    file_df: pd.DataFrame,
    metadata_config: Any | None,
) -> tuple[list[str | list[str]], list[str | None]]:
    """Resolve image and mask paths for property computation.

    Uses the provided ``file_df`` as the single source of truth.
    """
    file_df_lookup = build_file_df_lookup(file_df)
    image_paths = resolve_image_paths(
        results.img_path,
        file_df_lookup=file_df_lookup,
    )
    metadata_list = results.metadata or []
    mask_paths = resolve_mask_paths(
        results.img_path,
        mask_dir=None,
        image_paths=image_paths,
        image_dir=None,
        metadata_list=metadata_list,
        metadata_config=metadata_config,
        file_df_lookup=file_df_lookup,
    )
    return image_paths, mask_paths


def normalize_property_functions(
    property_preset: str | None = None,
    additional_property_functions: dict[str, Callable | list[Callable]] | None = None,
) -> dict[str, list[Callable]]:
    """Normalize property functions input to standardized dict format."""
    base: dict[str, list[Callable]] = {}
    if property_preset is not None and property_preset != "none":
        if property_preset not in (
            "basic",
            "shape",
            "intensity",
            "standard",
            "complete",
        ):
            raise ValueError(
                f"Unknown preset '{property_preset}'. Valid presets: "
                "'none', 'basic', 'shape', 'intensity', 'standard', 'complete'"
            )
        base = get_preset_property_functions(property_preset)

    extras_normalized = (
        normalize_property_functions_dict(additional_property_functions)
        if additional_property_functions
        else {}
    )

    if not base and not extras_normalized:
        logger.warning("No property functions provided, nothing to compute.")
        return {}

    return merge_property_dicts(base, extras_normalized)


def normalize_property_functions_dict(
    property_functions: str | dict[str, Callable | list[Callable]],
) -> dict[str, list[Callable]]:
    """Normalize a property_functions dict or preset string to standardized format."""
    if isinstance(property_functions, str):
        resolved: dict[str, list[Callable]] = get_preset_property_functions(property_functions)
    else:
        resolved = property_functions

    if not isinstance(resolved, dict):
        raise TypeError(
            f"property_functions must be a dict or preset string, got: {type(property_functions)}"
        )

    valid_requirements = {"image", "mask", "both", "any"}
    bad_keys = [k for k in resolved if k not in valid_requirements]
    if bad_keys:
        raise ValueError(
            f"Invalid requirement keys: {bad_keys}. Must be one of: {valid_requirements}"
        )

    normalized: dict[str, list[Callable]] = {}
    for req, val in resolved.items():
        if callable(val):
            normalized[req] = [val]
        elif isinstance(val, list):
            bad_idx = [i for i, fn in enumerate(val) if not callable(fn)]
            if bad_idx:
                raise TypeError(
                    f"property_functions['{req}'] has non-callable at indices {bad_idx}"
                )
            normalized[req] = val
        else:
            raise TypeError(
                f"property_functions['{req}'] must be callable or list, got: {type(val)}"
            )

    return normalized


def merge_property_dicts(
    base: dict[str, list[Callable]],
    extras: dict[str, list[Callable]],
) -> dict[str, list[Callable]]:
    """Merge two property dicts by concatenating lists for each requirement key."""
    merged: dict[str, list[Callable]] = dict(base)
    for req, fns in extras.items():
        merged[req] = merged.get(req, []) + fns
    return merged


def infer_expected_property_keys(
    property_functions: dict[str, list[Callable]],
    image_paths: list[str | list[str]],
    mask_paths: list[str | None],
    channel_names: list[str] | None = None,
) -> set[str]:
    """Infer property column names using the first image/mask."""
    if not image_paths:
        return set()
    any_requires_image = any(r in ("image", "both", "any") for r in property_functions)
    any_requires_mask = any(r in ("mask", "both", "any") for r in property_functions)
    img0 = image_paths[0]
    mask0: str | None = None
    if mask_paths and len(mask_paths) > 0:
        mask0 = mask_paths[0]
    img_stack, mask_stack = _load_image_and_mask_stacks(
        img0, mask0, any_requires_image, any_requires_mask, property_functions
    )
    return collect_property_names_for_stacks(
        img_stack, mask_stack, property_functions, channel_names=channel_names
    )


def checkpoint_rows_align_for_property_save(
    image_paths: list[str | list[str]],
    image_metadata: list[dict[str, Any]],
    ckpt_paths: list[str],
    ckpt_metadata: list[dict[str, Any]],
) -> bool:
    """Return True if session rows can be written into the checkpoint.

    Exact match and checkpoint-as-prefix of the session are accepted. A longer
    checkpoint is also accepted when every session sample maps to a checkpoint
    row (extra checkpoint rows are left in place).
    """
    n_ckpt = len(ckpt_paths)
    n_session = len(image_paths)
    if n_session == 0 or n_ckpt == 0:
        return False

    if n_ckpt > n_session:
        try:
            id_to_idx = build_metadata_key_index(ckpt_metadata, on_duplicate="first")
            for meta in image_metadata:
                key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
                if key not in id_to_idx:
                    return False
            return True
        except ValueError:
            ckpt_set = {_canonical_path(p) for p in ckpt_paths}
            return all(_canonical_path(_primary_path(p)) in ckpt_set for p in image_paths)

    if len(image_metadata) >= n_ckpt and len(ckpt_metadata) >= n_ckpt:
        meta_aligned = True
        for i in range(n_ckpt):
            im = image_metadata[i] if isinstance(image_metadata[i], dict) else {}
            cm = ckpt_metadata[i] if isinstance(ckpt_metadata[i], dict) else {}
            try:
                if metadata_to_stable_key(im) != metadata_to_stable_key(cm):
                    meta_aligned = False
                    break
            except ValueError:
                meta_aligned = False
                break
        if meta_aligned:
            return True

    primary_paths = [_primary_path(p) for p in image_paths[:n_ckpt]]
    return all(
        _canonical_path(str(a)) == _canonical_path(str(b))
        for a, b in zip(primary_paths, ckpt_paths, strict=True)
    )


def _merge_prop_row(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Keep extra checkpoint keys; overwrite with newly computed values."""
    out = dict(old) if isinstance(old, dict) else {}
    if isinstance(new, dict):
        out.update(new)
    return out


def align_session_properties_to_checkpoint(
    session_props: list[dict[str, Any]],
    session_internal: list[dict[str, Any]],
    image_paths: list[str | list[str]],
    image_metadata: list[dict[str, Any]],
    ckpt: Any,
    id_to_ckpt_idx: dict[str, int] | None = None,
    identifier_list: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Place session property rows onto checkpoint indices; extra ckpt rows are kept."""
    n_ckpt = int(ckpt.n_committed)
    if n_ckpt <= 0:
        return list(session_props), list(session_internal)

    aligned_props: list[dict[str, Any]] = [{} for _ in range(n_ckpt)]
    aligned_int: list[dict[str, Any]] = [{} for _ in range(n_ckpt)]
    if ckpt.has_property_content():
        existing = ckpt.load_committed_properties()
        existing_int = ckpt.load_internal_all()
        for i, p in enumerate(existing):
            if i < n_ckpt and isinstance(p, dict):
                aligned_props[i] = dict(p)
        for i, p in enumerate(existing_int):
            if i < n_ckpt and isinstance(p, dict):
                aligned_int[i] = dict(p)

    if id_to_ckpt_idx is None:
        ckpt_metadata = ckpt.get_committed_metadata_list()
        id_to_ckpt_idx = build_metadata_key_index(ckpt_metadata, on_duplicate="first")
    ckpt_paths = ckpt.get_committed_paths_list()
    path_to_idx = {_canonical_path(p): i for i, p in enumerate(ckpt_paths)}
    ckpt_lookup = _build_unique_path_lookup(ckpt_paths)
    image_lookup = _build_unique_path_lookup(image_paths)

    for i, prop in enumerate(session_props):
        ident = None
        if identifier_list is not None and i < len(identifier_list):
            ident = identifier_list[i]
        else:
            meta = image_metadata[i] if i < len(image_metadata) else {}
            try:
                ident = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
            except ValueError:
                ident = None
        idx = id_to_ckpt_idx.get(ident, -1) if ident is not None else -1
        if idx < 0 and i < len(image_paths):
            idx = path_to_idx.get(_canonical_path(image_paths[i]), -1)
            if idx < 0:
                matched = _match_path_index(image_paths[i], ckpt_lookup, image_lookup, i)
                if matched is not None:
                    idx = matched
        if not (0 <= idx < n_ckpt):
            continue
        aligned_props[idx] = _merge_prop_row(aligned_props[idx], prop)
        if i < len(session_internal) and session_internal[i]:
            aligned_int[idx] = dict(session_internal[i])
    return aligned_props, aligned_int


def _rewrite_checkpoint_properties(
    ckpt: Any,
    aligned_props: list[dict[str, Any]],
    aligned_internal: list[dict[str, Any]],
) -> int:
    """Replace committed property columns with an aligned full table."""
    ckpt.clear_properties()
    if not aligned_props:
        return 0
    ckpt.buffer_properties(aligned_props)
    if len(aligned_internal) == len(aligned_props):
        ckpt.buffer_internal(aligned_internal)
    else:
        ckpt.buffer_internal([{} for _ in aligned_props])
    return ckpt.commit_properties()


def flush_computed_properties_to_checkpoint(
    ckpt: Any,
    feature_buffers: dict[str, list[float]],
    all_property_names: set,
    internal_buffer: list[dict[str, Any]],
    paths_list: list[tuple[int, str | list[str]]],
    n_computed: int,
    last_committed: int,
    image_paths: list[str | list[str]],
    *,
    aligned: bool,
    id_to_ckpt_idx: dict[str, int] | None,
    identifier_list: list[str] | None,
) -> int:
    """Persist newly computed rows. Returns the new last_committed count."""
    if ckpt is None or n_computed <= last_committed:
        return last_committed
    if aligned and id_to_ckpt_idx is not None:
        computed = feature_buffers_to_dicts(feature_buffers, all_property_names, 0, n_computed)
        session_props: list[dict[str, Any]] = [{} for _ in range(len(image_paths))]
        session_int: list[dict[str, Any]] = [{} for _ in range(len(image_paths))]
        for buf_i, prop in enumerate(computed):
            img_idx = paths_list[buf_i][0]
            if 0 <= img_idx < len(session_props):
                session_props[img_idx] = prop
                if buf_i < len(internal_buffer):
                    session_int[img_idx] = dict(internal_buffer[buf_i])
        aligned_p, aligned_i = align_session_properties_to_checkpoint(
            session_props,
            session_int,
            image_paths,
            [],
            ckpt,
            id_to_ckpt_idx,
            identifier_list,
        )
        n_total = _rewrite_checkpoint_properties(ckpt, aligned_p, aligned_i)
        logger.debug(
            "Checkpoint: incremental write %d newly computed / %d rows.",
            n_computed - last_committed,
            n_total,
        )
        return n_computed

    remaining = feature_buffers_to_dicts(
        feature_buffers, all_property_names, last_committed, n_computed
    )
    if remaining:
        rem_int = internal_buffer[last_committed:n_computed]
        if len(rem_int) != len(remaining):
            rem_int = [{} for _ in remaining]
        ckpt.buffer_properties(remaining)
        ckpt.buffer_internal(rem_int)
        n_total = ckpt.commit_properties()
        logger.debug(
            "Checkpoint: incremental write %d newly computed / %d rows.",
            n_computed - last_committed,
            n_total,
        )
    return n_computed


def warn_checkpoint_missing_requested_keys(
    checkpoint_path: str,
    expected: set[str],
    props_list: list[dict],
) -> None:
    """Log a warning if the checkpoint does not define all requested property keys."""
    if not expected:
        return
    union_ckpt: set[str] = set()
    for p in props_list:
        if isinstance(p, dict):
            union_ckpt.update(str(k) for k in p)
    have = {k.lower() for k in union_ckpt}
    missing = [k for k in expected if str(k).lower() not in have]
    if not missing:
        return
    logger.warning(
        "Checkpoint %s does not store all requested properties; missing %d key(s): %s. "
        "This often happens after changing property_preset or custom functions. "
        "Call compute_properties(..., force_update=True) with the same checkpoint_path "
        "to recompute all properties and refresh the file, or use a new path / delete "
        "the file to start fresh.",
        checkpoint_path,
        len(missing),
        sorted(missing),
    )


def is_property_dict_unprocessed(prop: dict, internal: dict | None = None) -> bool:
    """Return True if this row has not been successfully written yet.

    A finite ``_properties_attempted`` flag means the row was computed even when
    every phenotype is NaN. All-NaN rows without that flag are unfinished
    (including HDF5 placeholder rows created so extra checkpoint samples keep
    alignment).
    """
    if internal:
        w = internal.get("_properties_attempted", np.nan)
        if isinstance(w, (float, int, np.floating, np.integer)) and not (
            isinstance(w, (float, np.floating)) and bool(np.isnan(float(w)))
        ):
            return False
    if not prop:
        return True
    for v in prop.values():
        if v is None:
            continue
        if isinstance(v, (int, bool, np.integer)):
            return False
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                continue
            return False
        return False
    return True


def row_missing_expected_keys(prop: dict, expected_property_keys: set[str] | None) -> bool:
    """Return True if the checkpoint row lacks any currently requested property key.

    Comparison is case-insensitive. Extra stored keys are ignored. NaN values
    still count as present.
    """
    if not expected_property_keys or not isinstance(prop, dict) or not prop:
        return bool(expected_property_keys) and not bool(prop)
    have = {str(k).lower() for k in prop}
    return any(str(k).lower() not in have for k in expected_property_keys)


def refine_path_alignment_for_unprocessed(
    path_alignment: tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]],
    image_paths: list[str | list[str]],
    existing_props: list[dict],
    expected_property_keys: set[str] | None = None,
    existing_internal: list[dict] | None = None,
) -> tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]:
    """Move never-written rows, or rows missing requested keys, to paths_to_compute.

    Extra checkpoint columns are left in place. Stored NaNs count as already
    computed; only missing requested keys cause a recompute so they can be added.
    """
    id_to_ckpt_idx, identifiers_with_props, paths_to_compute, identifier_list = path_alignment
    paths_to_compute_dict: dict[int, str | list[str]] = dict(paths_to_compute)
    for i, identifier in enumerate(identifier_list):
        if identifier in identifiers_with_props:
            idx = id_to_ckpt_idx.get(identifier, -1)
            if 0 <= idx < len(existing_props):
                prop = existing_props[idx]
                int_row = (
                    existing_internal[idx]
                    if existing_internal and 0 <= idx < len(existing_internal)
                    else None
                )

                is_missing_expected = row_missing_expected_keys(prop, expected_property_keys)

                if is_property_dict_unprocessed(prop, int_row) or is_missing_expected:
                    identifiers_with_props.discard(identifier)
                    paths_to_compute_dict[i] = image_paths[i]
            else:
                identifiers_with_props.discard(identifier)
                paths_to_compute_dict[i] = image_paths[i]
    return (
        id_to_ckpt_idx,
        identifiers_with_props,
        sorted(paths_to_compute_dict.items(), key=lambda x: x[0]),
        identifier_list,
    )


def feature_buffers_to_dicts(
    feature_buffers: dict[str, list[float]],
    all_property_names: set,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Convert slice of feature_buffers into list of property dictionaries."""
    result: list[dict[str, Any]] = []
    for buf_idx in range(start, end):
        props: dict[str, Any] = {}
        for pn in all_property_names:
            if pn in feature_buffers and buf_idx < len(feature_buffers[pn]):
                v = feature_buffers[pn][buf_idx]
                props[pn] = float(v) if not np.isnan(v) else np.nan
            else:
                props[pn] = np.nan
        result.append(optimize_property_types(props))
    return result


def compute_properties_for_channel(
    ch: int,
    img_stack: np.ndarray | None,
    mask_stack: np.ndarray | None,
    n_img_ch: int,
    n_mask_ch: int,
    property_functions: dict[str, list[Callable]],
    feature_buffers: dict[str, list[float]],
    img_idx: int,
    n_channels: int,
    start_ch: int = 0,
    mask_properties_computed: set | None = None,
    channel_names: list[str] | None = None,
) -> tuple[set, set, set]:
    """Compute properties for one channel, populating feature_buffers."""
    has_img_fn = any(r in ("image", "both") for r in property_functions)
    has_mask_fn = any(r in ("mask", "both", "any") for r in property_functions)
    only_mask = "mask" in property_functions and not has_img_fn and "any" not in property_functions

    image2d: np.ndarray | None = None
    if (
        img_stack is not None
        and has_img_fn
        and not only_mask
        and (
            (n_channels == n_img_ch and ch < n_img_ch)
            or (n_channels == n_mask_ch and n_img_ch > 0 and ch < n_img_ch)
        )
    ):
        image2d = img_stack[..., ch]

    mask2d: np.ndarray | None = None
    if mask_stack is not None and has_mask_fn:
        if only_mask:
            mask2d = mask_stack[..., 0] if mask_stack.ndim == 3 else mask_stack
        elif n_mask_ch == 1:
            mask2d = mask_stack[..., 0]
        elif n_channels == n_mask_ch and ch < n_mask_ch:
            mask2d = mask_stack[..., ch]
        elif n_channels == n_img_ch:
            if ch < n_mask_ch:
                mask2d = mask_stack[..., ch]
            elif n_mask_ch == 1:
                mask2d = mask_stack[..., 0]

    base_names: set = set()
    computed: set = set()
    mask_base: set = set()
    if mask_properties_computed is None:
        mask_properties_computed = set()

    for req_type, fn_list in property_functions.items():
        if req_type == "mask" and ch != start_ch:
            continue
        if not _should_call_property_function(req_type, image2d, mask2d):
            continue

        for fn in fn_list:
            try:
                feat_dict = fn(image2d, mask2d)
                if not isinstance(feat_dict, dict):
                    logger.warning(
                        "Property function %s did not return a dict, skipping.",
                        getattr(fn, "__name__", "unknown"),
                    )
                    continue
                base_names.update(feat_dict.keys())
                if req_type == "mask":
                    mask_base.update(feat_dict.keys())
            except Exception as exc:
                logger.warning(
                    "Property function %s failed: %s", getattr(fn, "__name__", "unknown"), exc
                )
                feat_dict = {}

            for bname, value in feat_dict.items():
                if req_type == "mask":
                    name = bname
                    if name in mask_properties_computed:
                        continue
                    mask_properties_computed.add(name)
                else:
                    name = format_channel_property_name(bname, ch, n_channels, channel_names)

                if name not in feature_buffers:
                    feature_buffers[name] = [np.nan] * img_idx
                feature_buffers[name].append(float(value) if value is not None else np.nan)
                computed.add(name)

    return base_names, computed, mask_base


def setup_properties_checkpoint_resume(
    checkpoint_path: str | None,
    results: "PhenoMeResults",
    active_db: Any | None = None,
    image_paths: list[str | list[str]] | None = None,
    lazy: bool = True,
    expected_property_keys: set[str] | None = None,
) -> tuple[
    Any | None,
    int,
    tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]] | None,
]:
    """Setup checkpoint resume for properties computation."""
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return None, 0, None

    if active_db is not None and os.path.abspath(active_db.path) == os.path.abspath(
        checkpoint_path
    ):
        ckpt = active_db
    else:
        ckpt = CheckpointManager(checkpoint_path, lazy=lazy)

    ckpt.reset_empty_property_counter()
    n_already = ckpt.n_committed_props if ckpt.has_property_content() else 0
    n_with_emb = ckpt.n_committed
    if image_paths is None:
        image_paths = list(results.img_path)
    image_metadata = (
        list(results.metadata)
        if results.metadata is not None
        else [{} for _ in range(len(image_paths))]
    )
    ckpt_paths = ckpt.get_committed_paths_list()
    ckpt_metadata = ckpt.get_committed_metadata_list()

    try:
        if n_already >= n_with_emb > 0:
            loaded = ckpt.load_committed_results()
            all_props = loaded.properties
            all_internal = ckpt.load_internal_all()
            if expected_property_keys:
                warn_checkpoint_missing_requested_keys(
                    checkpoint_path, expected_property_keys, list(all_props)
                )

            filtered_props = []
            filtered_internal: list[dict] = []
            id_to_idx = build_metadata_key_index(ckpt_metadata, on_duplicate="error")
            try:
                for _i, meta in enumerate(image_metadata):
                    key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
                    idx = id_to_idx.get(key, -1)
                    if 0 <= idx < len(all_props):
                        filtered_props.append(all_props[idx])
                        filtered_internal.append(
                            all_internal[idx] if 0 <= idx < len(all_internal) else {}
                        )
                    else:
                        filtered_props.append({})
                        filtered_internal.append({})
            except ValueError:
                path_to_idx = {_canonical_path(p): i for i, p in enumerate(ckpt_paths)}
                ckpt_lookup = _build_unique_path_lookup(ckpt_paths)
                image_lookup = _build_unique_path_lookup(image_paths)
                for i, p in enumerate(image_paths):
                    idx = path_to_idx.get(_canonical_path(p), -1)
                    if idx < 0:
                        matched = _match_path_index(p, ckpt_lookup, image_lookup, i)
                        if matched is not None:
                            idx = matched
                    if 0 <= idx < len(all_props):
                        filtered_props.append(all_props[idx])
                        filtered_internal.append(
                            all_internal[idx] if 0 <= idx < len(all_internal) else {}
                        )
                    else:
                        filtered_props.append({})
                        filtered_internal.append({})

            unprocessed_indices = [
                i
                for i, prop in enumerate(filtered_props)
                if is_property_dict_unprocessed(
                    prop, filtered_internal[i] if i < len(filtered_internal) else None
                )
                or row_missing_expected_keys(prop, expected_property_keys)
            ]
            if not unprocessed_indices:
                n_total = len(image_paths)
                n_prop = (
                    len(expected_property_keys)
                    if expected_property_keys
                    else (
                        len(filtered_props[0].keys())
                        if filtered_props and isinstance(filtered_props[0], dict)
                        else 0
                    )
                )
                logger.info(
                    "Properties: %d/%d matched in checkpoint (%d columns).",
                    n_total,
                    n_total,
                    n_prop,
                )
                results.properties = filtered_props
                if ckpt is not active_db:
                    ckpt.close()
                return None, n_already, None

            # Calculate initial alignment before refinement for better logging
            path_alignment = align_by_metadata(
                image_paths, image_metadata, ckpt_paths, ckpt_metadata, n_already, _primary_path
            )
            n_total = len(image_paths)

            existing_props = ckpt.load_committed_properties()
            existing_int = ckpt.load_internal_all()

            # Count properties in checkpoint
            n_props_in_ckpt = len(ckpt.property_column_names()) or len(
                _union_prop_keys_sample(existing_props)
            )

            # Refine alignment (this modifies path_alignment[1] set)
            path_alignment = refine_path_alignment_for_unprocessed(
                path_alignment,
                image_paths,
                existing_props,
                expected_property_keys,
                existing_internal=existing_int,
            )

            n_fully_matched = len(path_alignment[1])
            n_to_compute = n_total - n_fully_matched

            if n_fully_matched > 0:
                logger.info(
                    "Properties: %d/%d matched in checkpoint (%d columns). Computing %d remaining.",
                    n_fully_matched,
                    n_total,
                    n_props_in_ckpt,
                    n_to_compute,
                )
            else:
                logger.info("Properties: 0/%d matched in checkpoint. Computing all.", n_total)

            return ckpt, n_already, path_alignment

        elif n_already > 0:
            # Calculate initial alignment before refinement for better logging
            path_alignment = align_by_metadata(
                image_paths, image_metadata, ckpt_paths, ckpt_metadata, n_already, _primary_path
            )
            n_total = len(image_paths)

            existing_props = ckpt.load_committed_properties()
            existing_int = ckpt.load_internal_all()
            if expected_property_keys:
                warn_checkpoint_missing_requested_keys(
                    checkpoint_path, expected_property_keys, list(existing_props)
                )

            # Count properties in checkpoint
            n_props_in_ckpt = len(ckpt.property_column_names()) or len(
                _union_prop_keys_sample(existing_props)
            )

            # Refine alignment (this modifies path_alignment[1] set)
            path_alignment = refine_path_alignment_for_unprocessed(
                path_alignment,
                image_paths,
                existing_props,
                expected_property_keys,
                existing_internal=existing_int,
            )

            n_fully_matched = len(path_alignment[1])
            n_to_compute = n_total - n_fully_matched

            if n_fully_matched > 0:
                logger.info(
                    "Properties: %d/%d matched in checkpoint (%d columns). Computing %d remaining.",
                    n_fully_matched,
                    n_total,
                    n_props_in_ckpt,
                    n_to_compute,
                )
            else:
                logger.info("Properties: 0/%d matched in checkpoint. Computing all.", n_total)

            return ckpt, n_already, path_alignment

        path_alignment = align_by_metadata(
            image_paths, image_metadata, ckpt_paths, ckpt_metadata, n_already, _primary_path
        )
        return ckpt, n_already, path_alignment
    except Exception:
        raise


def process_all_images(
    image_paths: list[str | list[str]],
    mask_paths: list[str | None],
    property_functions: dict[str, list[Callable]],
    ckpt: Any | None,
    checkpoint_path: str | None,
    save_every: int,
    path_alignment: tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]
    | None,
    n_jobs: int = 1,
    expected_property_keys: set[str] | None = None,
    channel_names: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[set, dict[str, list[float]], int, list[dict[str, Any]]]:
    """Process all images and compute properties."""
    any_requires_image = any(r in ("image", "both", "any") for r in property_functions)
    any_requires_mask = any(r in ("mask", "both", "any") for r in property_functions)

    id_to_ckpt_idx: dict[str, int] | None = None
    identifier_list: list[str] | None = None
    if path_alignment is None:
        paths_to_compute = [(i, p) for i, p in enumerate(image_paths)]
    else:
        id_to_ckpt_idx, _, paths_to_compute, identifier_list = path_alignment

    all_property_names: set = set(expected_property_keys) if expected_property_keys else set()
    feature_buffers: dict[str, list[float]] = {}
    internal_buffer: list[dict[str, Any]] = []
    images_computed = 0
    last_committed = 0
    use_ckpt = checkpoint_path is not None
    use_ckpt_incremental = use_ckpt and ckpt is not None
    use_aligned_rewrite = bool(
        use_ckpt_incremental
        and id_to_ckpt_idx is not None
        and ckpt is not None
        and ckpt.n_committed > 0
    )
    n_to_process = len(paths_to_compute)
    paths_list = list(paths_to_compute)
    desc = "Computing properties"
    disable_tqdm = progress_callback is not None

    if n_jobs != 1:
        batch_size = (
            save_every
            if (use_ckpt_incremental and ckpt is not None)
            else max(1, min(50, n_to_process))
        )
        with tqdm(
            total=n_to_process,
            desc=desc,
            unit="img",
            disable=disable_tqdm,
        ) as pbar:
            for batch_start in range(0, n_to_process, batch_size):
                batch_end = min(batch_start + batch_size, n_to_process)
                batch_tasks = [
                    (
                        buf_idx,
                        paths_list[buf_idx][1],
                        mask_paths[paths_list[buf_idx][0]],
                    )
                    for buf_idx in range(batch_start, batch_end)
                ]
                results = list(
                    Parallel(n_jobs=n_jobs)(
                        delayed(compute_properties_worker)(
                            buf_idx,
                            img_path,
                            mask_path,
                            property_functions,
                            any_requires_image,
                            any_requires_mask,
                            channel_names,
                        )
                        for buf_idx, img_path, mask_path in batch_tasks
                    )
                )
                results.sort(key=lambda x: x[0])
                batch_prop_names = set()
                for _, props_dict, _ in results:
                    batch_prop_names.update(props_dict.keys())
                all_property_names.update(batch_prop_names)

                for buf_idx, props_dict, _ in results:
                    for pn in all_property_names:
                        val = props_dict.get(pn, np.nan)
                        if pn not in feature_buffers:
                            feature_buffers[pn] = [np.nan] * buf_idx
                        feature_buffers[pn].append(val)
                for _, _props, int_d in results:
                    internal_buffer.append(dict(int_d))
                images_computed = batch_end
                if use_ckpt_incremental and ckpt is not None and not use_aligned_rewrite:
                    last_committed = flush_computed_properties_to_checkpoint(
                        ckpt,
                        feature_buffers,
                        all_property_names,
                        internal_buffer,
                        paths_list,
                        images_computed,
                        last_committed,
                        image_paths,
                        aligned=use_aligned_rewrite,
                        id_to_ckpt_idx=id_to_ckpt_idx,
                        identifier_list=identifier_list,
                    )
                pbar.update(batch_end - batch_start)
                report_progress(progress_callback, batch_end, n_to_process, desc)
    else:
        iterable: Any = paths_list
        if n_to_process > 0:
            iterable = tqdm(
                paths_list,
                desc=desc,
                total=n_to_process,
                unit="img",
                disable=disable_tqdm,
            )
        for buf_idx, (img_idx, img_path) in enumerate(iterable):
            mask_path = mask_paths[img_idx]

            img_stack, mask_stack = _load_image_and_mask_stacks(
                img_path,
                mask_path,
                any_requires_image,
                any_requires_mask,
                property_functions=property_functions,
            )
            n_img_ch = 0 if img_stack is None else img_stack.shape[-1]
            n_mask_ch = 0 if mask_stack is None else mask_stack.shape[-1]
            start_ch, end_ch = _determine_channel_range(property_functions, n_img_ch, n_mask_ch)
            n_channels = end_ch - start_ch
            validate_channel_names(channel_names, n_channels)

            all_computed: set = set()
            mask_props_done: set = set()
            for ch in range(start_ch, end_ch):
                _, computed, _ = compute_properties_for_channel(
                    ch,
                    img_stack,
                    mask_stack,
                    n_img_ch,
                    n_mask_ch,
                    property_functions,
                    feature_buffers,
                    buf_idx,
                    n_channels,
                    start_ch=start_ch,
                    mask_properties_computed=mask_props_done,
                    channel_names=channel_names,
                )
                all_computed.update(computed)

            internal_buffer.append({"_properties_attempted": 1.0})

            for pn in list(feature_buffers):
                if pn not in all_computed:
                    feature_buffers[pn].append(np.nan)

            all_property_names.update(feature_buffers.keys())
            images_computed += 1

            if (
                use_ckpt_incremental
                and ckpt is not None
                and not use_aligned_rewrite
                and images_computed % save_every == 0
            ):
                last_committed = flush_computed_properties_to_checkpoint(
                    ckpt,
                    feature_buffers,
                    all_property_names,
                    internal_buffer,
                    paths_list,
                    images_computed,
                    last_committed,
                    image_paths,
                    aligned=use_aligned_rewrite,
                    id_to_ckpt_idx=id_to_ckpt_idx,
                    identifier_list=identifier_list,
                )

            report_progress(progress_callback, images_computed, n_to_process, desc)

    return all_property_names, feature_buffers, last_committed, internal_buffer


def finalize_properties_computation(
    results: "PhenoMeResults",
    feature_buffers: dict[str, list[float]],
    all_property_names: set,
    path_alignment: tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]
    | None,
    last_committed: int,
    ckpt: Any | None,
    checkpoint_path: str | None,
    image_paths: list[str | list[str]],
    processing_params: dict[str, Any] | None = None,
    lazy: bool = True,
    internal_buffer: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Finalize property computation and handle checkpoint commit."""
    ib = internal_buffer or []
    images_computed = max((len(buf) for buf in feature_buffers.values()), default=0)
    new_props = feature_buffers_to_dicts(feature_buffers, all_property_names, 0, images_computed)
    full_internal: list[dict[str, Any]] = []

    if path_alignment is not None and ckpt is not None:
        id_to_ckpt_idx, identifiers_with_props, paths_to_compute, identifier_list = path_alignment
        existing_props = ckpt.load_committed_properties()
        existing_internal = ckpt.load_internal_all()
        img_idx_to_new_idx = {img_idx: i for i, (img_idx, _) in enumerate(paths_to_compute)}
        ckpt_lookup = _build_unique_path_lookup(ckpt.get_committed_paths_list())
        image_lookup = _build_unique_path_lookup(image_paths)

        all_keys = set(all_property_names)
        for prop_dict in existing_props:
            if isinstance(prop_dict, dict):
                all_keys.update(prop_dict.keys())

        full_props = []
        for i in range(len(image_paths)):
            identifier = (
                identifier_list[i]
                if i < len(identifier_list)
                else _canonical_path(image_paths[i] if i < len(image_paths) else "")
            )
            if identifier in identifiers_with_props:
                idx = id_to_ckpt_idx.get(identifier, -1)
                if idx < 0:
                    matched = _match_path_index(image_paths[i], ckpt_lookup, image_lookup, i)
                    if matched is not None:
                        idx = matched
                if 0 <= idx < len(existing_props):
                    prop = dict(existing_props[idx]) if existing_props[idx] else {}
                else:
                    prop = {}
                if 0 <= idx < len(existing_internal):
                    int_row = dict(existing_internal[idx]) if existing_internal[idx] else {}
                else:
                    int_row = {}
            else:
                new_idx = img_idx_to_new_idx.get(i, -1)
                if 0 <= new_idx < len(new_props):
                    prop = dict(new_props[new_idx]) if new_props[new_idx] else {}
                else:
                    prop = {}
                if 0 <= new_idx < len(ib):
                    int_row = dict(ib[new_idx]) if ib[new_idx] else {}
                else:
                    int_row = {}

                # If this sample was in the checkpoint but needed recomputation (e.g. missing columns),
                # merge the old properties so we don't lose them if the new run only computed a subset
                # of properties or failed to compute some.
                if identifier in id_to_ckpt_idx:
                    old_idx = id_to_ckpt_idx[identifier]
                    if 0 <= old_idx < len(existing_props):
                        old_p = existing_props[old_idx]
                        if isinstance(old_p, dict):
                            for k, v in old_p.items():
                                # Fill missing or NaN values from checkpoint
                                if k not in prop or (
                                    isinstance(prop[k], (float, np.floating)) and np.isnan(prop[k])
                                ):
                                    prop[k] = v
                    if not int_row and 0 <= old_idx < len(existing_internal):
                        int_row = (
                            dict(existing_internal[old_idx]) if existing_internal[old_idx] else {}
                        )

            for key in all_keys:
                if key not in prop:
                    prop[key] = np.nan
            full_props.append(optimize_property_types(prop))
            full_internal.append(int_row)
    else:
        full_props = new_props
        full_internal = [dict(d) for d in ib[: len(new_props)]] if ib else []
        while len(full_internal) < len(full_props):
            full_internal.append({})

    results.properties = full_props

    ckpt_created_this_run = False
    if checkpoint_path is not None:
        if ckpt is None:
            eager_emb = results.embeddings
            emb_dim = (
                int(eager_emb.shape[1])
                if isinstance(eager_emb, np.ndarray) and eager_emb.ndim == 2
                else None
            )
            ckpt = CheckpointManager(
                checkpoint_path,
                embedding_dim=emb_dim,
                processing_params=processing_params,
                lazy=lazy,
            )

            img_paths_list = list(results.img_path)
            metadata_list = (
                list(results.metadata)
                if results.metadata is not None
                else [{} for _ in img_paths_list]
            )

            if emb_dim is not None:
                ckpt.buffer_embeddings(eager_emb, img_paths_list, metadata_list)
                ckpt.commit_embeddings()
            else:
                ckpt.buffer_paths_and_metadata(img_paths_list, metadata_list)
                ckpt.commit_embeddings()
            ckpt_created_this_run = True

        if ckpt is not None:
            ckpt_paths = ckpt.get_committed_paths_list()
            ckpt_metadata_list = ckpt.get_committed_metadata_list()
            session_metadata = (
                list(results.metadata)
                if results.metadata is not None
                else [{} for _ in range(len(image_paths))]
            )
            paths_match = ckpt_created_this_run or checkpoint_rows_align_for_property_save(
                image_paths,
                session_metadata,
                ckpt_paths,
                ckpt_metadata_list,
            )
            have_full_props = len(full_props) == len(image_paths)

            if paths_match and have_full_props:
                # If the checkpoint is a compatible prefix of the session, we MUST
                # extend it with the missing session paths/metadata first, so that
                # committing properties (which are aligned to image_paths) doesn't
                # create misaligned datasets.
                if len(image_paths) > len(ckpt_paths):
                    raise RuntimeError(
                        f"Session has {len(image_paths)} images but checkpoint has "
                        f"{len(ckpt_paths)} committed rows. Run process_images() for new "
                        "samples before saving properties to checkpoint. Temporal images "
                        "from process_temporal_images() are not persisted to checkpoint."
                    )

                if path_alignment is None:
                    images_computed = max((len(buf) for buf in feature_buffers.values()), default=0)
                    remaining = feature_buffers_to_dicts(
                        feature_buffers, all_property_names, last_committed, images_computed
                    )
                    if remaining:
                        rem_int = ib[last_committed:images_computed]
                        if len(rem_int) != len(remaining):
                            rem_int = [{} for _ in remaining]
                        ckpt.buffer_properties(remaining)
                        ckpt.buffer_internal(rem_int)
                        n_total = ckpt.commit_properties()
                        logger.info("Checkpoint: saved %d properties.", n_total)
                else:
                    id_to_ckpt_idx, _, _, identifier_list = path_alignment
                    aligned_props, aligned_int = align_session_properties_to_checkpoint(
                        full_props,
                        full_internal,
                        image_paths,
                        session_metadata,
                        ckpt,
                        id_to_ckpt_idx,
                        identifier_list,
                    )
                    n_total = _rewrite_checkpoint_properties(ckpt, aligned_props, aligned_int)
                    logger.info(
                        "Checkpoint: saved %d properties (refreshed).",
                        n_total,
                    )
            else:
                if not paths_match:
                    logger.warning(
                        "Skipping checkpoint persistence: current images do not align with "
                        "checkpoint rows (different count or sample order/metadata vs paths)."
                    )
                else:
                    logger.warning(
                        "Skipping checkpoint persistence: properties count (%d) "
                        "does not match image count (%d).",
                        len(full_props),
                        len(image_paths),
                    )
            # We don't close ckpt here if it was passed in as active_db,
            # but finalize_properties_computation in main.py will handle it.
            # However, for new checkpoints created here, we should return it or close it.
            # The original code closed it if self._db is None or self._db is not ckpt.

    return full_props, full_internal
