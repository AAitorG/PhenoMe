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
    _canonical_path,
    _union_prop_keys_sample,
    build_metadata_key_index,
)
from ...utils.path_utils import primary_path as _primary_path
from ...utils.property_factories import get_preset_property_functions
from ._paths import build_file_df_lookup, resolve_image_paths, resolve_mask_paths
from ._workers import (
    _determine_channel_range,
    _load_image_and_mask_stacks,
    _should_call_property_function,
    collect_property_names_for_stacks,
    compute_properties_worker,
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
    return collect_property_names_for_stacks(img_stack, mask_stack, property_functions)


def checkpoint_rows_align_for_property_save(
    image_paths: list[str | list[str]],
    image_metadata: list[dict[str, Any]],
    ckpt_paths: list[str],
    ckpt_metadata: list[dict[str, Any]],
) -> bool:
    """Return True if checkpoint rows correspond to the current session (exact match or prefix).

    HDF5 properties must stay row-aligned with ``embeddings`` / ``img_path``. Session paths
    often differ from stored paths (relative vs absolute, storage-root rebase), while each
    row still refers to the same sample — detect via metadata stable keys when possible,
    then canonical primary-path comparison.
    """
    n_ckpt = len(ckpt_paths)
    if n_ckpt > len(image_paths):
        return False

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
            union_ckpt.update(p.keys())
    missing = expected - union_ckpt
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
    """Return True if property dict is empty or all values are NaN/None (unprocessed)."""
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


def refine_path_alignment_for_unprocessed(
    path_alignment: tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]],
    image_paths: list[str | list[str]],
    existing_props: list[dict],
    expected_property_keys: set[str] | None = None,
    existing_internal: list[dict] | None = None,
) -> tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]:
    """Move unprocessed (missing or all-NaN) entries to paths_to_compute."""
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

                is_missing_expected = False
                if expected_property_keys is not None:
                    is_missing_expected = any(k not in prop for k in expected_property_keys)

                if is_property_dict_unprocessed(prop, int_row) or is_missing_expected:
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
                    name = f"{bname}_ch{ch}" if n_channels > 1 else bname

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

    n_already = ckpt.n_committed_props
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
                for p in image_paths:
                    cp = _canonical_path(p)
                    idx = path_to_idx.get(cp, -1)
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
                or (expected_property_keys and any(k not in prop for k in expected_property_keys))
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
            n_props_in_ckpt = len(_union_prop_keys_sample(existing_props))

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
            n_props_in_ckpt = len(_union_prop_keys_sample(existing_props))

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
) -> tuple[set, dict[str, list[float]], int, list[dict[str, Any]]]:
    """Process all images and compute properties."""
    any_requires_image = any(r in ("image", "both", "any") for r in property_functions)
    any_requires_mask = any(r in ("mask", "both", "any") for r in property_functions)

    if path_alignment is None:
        paths_to_compute = [(i, p) for i, p in enumerate(image_paths)]
    else:
        _, _, paths_to_compute, _ = path_alignment

    all_property_names: set = set(expected_property_keys) if expected_property_keys else set()
    feature_buffers: dict[str, list[float]] = {}
    internal_buffer: list[dict[str, Any]] = []
    images_computed = 0
    last_committed = 0
    use_ckpt = checkpoint_path is not None
    use_ckpt_incremental = use_ckpt and path_alignment is None
    n_to_process = len(paths_to_compute)
    paths_list = list(paths_to_compute)

    if n_jobs != 1:
        batch_size = (
            save_every
            if (use_ckpt_incremental and ckpt is not None)
            else max(1, min(50, n_to_process))
        )
        with tqdm(
            total=n_to_process,
            desc="Computing properties",
            unit="img",
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
                if use_ckpt_incremental and ckpt is not None:
                    new_dicts = feature_buffers_to_dicts(
                        feature_buffers, all_property_names, last_committed, images_computed
                    )
                    if new_dicts:
                        int_slice = internal_buffer[last_committed:images_computed]
                        ckpt.buffer_properties(new_dicts)
                        ckpt.buffer_internal(int_slice)
                        ckpt.commit_properties()
                    last_committed = images_computed
                pbar.update(batch_end - batch_start)
    else:
        if n_to_process > 0:
            paths_list = tqdm(
                paths_list,
                desc="Computing properties",
                total=n_to_process,
                unit="img",
            )
        for buf_idx, (img_idx, img_path) in enumerate(paths_list):
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
                )
                all_computed.update(computed)

            internal_buffer.append({"_properties_attempted": 1.0})

            for pn in list(feature_buffers):
                if pn not in all_computed:
                    feature_buffers[pn].append(np.nan)

            all_property_names.update(feature_buffers.keys())
            images_computed += 1

            if use_ckpt_incremental and ckpt is not None and images_computed % save_every == 0:
                new_dicts = feature_buffers_to_dicts(
                    feature_buffers, all_property_names, last_committed, images_computed
                )
                if new_dicts:
                    int_slice = internal_buffer[last_committed:images_computed]
                    ckpt.buffer_properties(new_dicts)
                    ckpt.buffer_internal(int_slice)
                    ckpt.commit_properties()
                last_committed = images_computed

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

    if path_alignment is not None and ckpt is not None and path_alignment[1]:
        id_to_ckpt_idx, identifiers_with_props, paths_to_compute, identifier_list = path_alignment
        existing_props = ckpt.load_committed_properties()
        existing_internal = ckpt.load_internal_all()
        img_idx_to_new_idx = {img_idx: i for i, (img_idx, _) in enumerate(paths_to_compute)}

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
                    ckpt.clear_properties()
                    ckpt.buffer_properties(full_props)
                    if full_internal and len(full_internal) == len(full_props):
                        ckpt.buffer_internal(full_internal)
                    elif not full_internal and full_props:
                        ckpt.buffer_internal([{} for _ in full_props])
                    n_total = ckpt.commit_properties()
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
