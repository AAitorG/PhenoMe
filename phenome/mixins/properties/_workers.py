"""
Worker and channel logic for property computation.

Module-level functions for parallel processing (picklable).
"""

from collections.abc import Callable

import numpy as np

from ..._logging import get_logger
from ...io import ensure_hwc, read_image
from ...utils.transforms import normalize_by_dtype_max

logger = get_logger(__name__)


def _load_image_and_mask_stacks(
    img_path: str | list[str] | None,
    mask_path: str | None,
    any_requires_image: bool,
    any_requires_mask: bool,
    property_functions: dict[str, list[Callable]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load image/mask arrays, only reading what is needed."""
    img_stack: np.ndarray | None = None
    mask_stack: np.ndarray | None = None

    only_mask = False
    if property_functions is not None:
        has_img_fn = any(r in ("image", "both") for r in property_functions)
        only_mask = (
            "mask" in property_functions and not has_img_fn and "any" not in property_functions
        )

    if (
        any_requires_image
        and img_path is not None
        and not only_mask
        and (img_path if isinstance(img_path, str) else len(img_path) > 0)
    ):
        try:
            img_stack = read_image(img_path)
            img_stack = ensure_hwc(img_stack)
            img_stack = normalize_by_dtype_max(img_stack)
        except Exception as exc:
            logger.warning("Failed to read image %s: %s", img_path, exc)

    if any_requires_mask and mask_path is not None:
        try:
            mask_stack = read_image(mask_path)
            mask_stack = ensure_hwc(mask_stack)
        except Exception as exc:
            logger.warning("Failed to read mask %s: %s", mask_path, exc)

    return img_stack, mask_stack


def _determine_channel_range(
    property_functions: dict[str, list[Callable]],
    n_img_ch: int,
    n_mask_ch: int,
) -> tuple[int, int]:
    """Determine channel range to iterate based on property function requirements."""
    has_img = any(r in ("image", "both") for r in property_functions)
    has_mask = any(r in ("mask", "both") for r in property_functions)
    has_any = "any" in property_functions
    only_mask = "mask" in property_functions and not has_img and not has_any
    only_img = "image" in property_functions and not has_mask and not has_any

    if only_mask:
        return (0, 1)
    if only_img:
        return (0, n_img_ch)
    if has_img:
        return (0, n_img_ch)
    if has_any:
        return (0, n_img_ch) if n_img_ch > 0 else (0, n_mask_ch)
    if has_mask:
        return (0, n_mask_ch)
    return (0, max(n_img_ch, n_mask_ch, 1))


def _should_call_property_function(
    req_type: str,
    image2d: np.ndarray | None,
    mask2d: np.ndarray | None,
) -> bool:
    """Check if property function should be called based on requirement type and available data."""
    if req_type == "image":
        return image2d is not None
    if req_type == "mask":
        return mask2d is not None
    if req_type == "both":
        return image2d is not None and mask2d is not None
    if req_type == "any":
        return image2d is not None or mask2d is not None
    return False


def collect_property_names_for_stacks(
    img_stack: np.ndarray | None,
    mask_stack: np.ndarray | None,
    property_functions: dict[str, list[Callable]],
) -> set:
    """Property column names produced for these stacks (same rules as compute_properties_worker).

    Uses the first image/mask pair's channel layout to infer names (including ``_ch{k}`` suffixes).
    """
    n_img_ch = 0 if img_stack is None else img_stack.shape[-1]
    n_mask_ch = 0 if mask_stack is None else mask_stack.shape[-1]
    start_ch, end_ch = _determine_channel_range(property_functions, n_img_ch, n_mask_ch)
    n_channels = end_ch - start_ch

    names: set = set()
    mask_props_done: set = set()
    has_img_fn = any(r in ("image", "both") for r in property_functions)
    has_mask_fn = any(r in ("mask", "both", "any") for r in property_functions)
    only_mask = "mask" in property_functions and not has_img_fn and "any" not in property_functions

    for ch in range(start_ch, end_ch):
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

        for req_type, fn_list in property_functions.items():
            if req_type == "mask" and ch != start_ch:
                continue
            if not _should_call_property_function(req_type, image2d, mask2d):
                continue

            for fn in fn_list:
                try:
                    feat_dict = fn(image2d, mask2d)
                    if not isinstance(feat_dict, dict):
                        continue
                except Exception:
                    feat_dict = {}

                for bname, _value in feat_dict.items():
                    if req_type == "mask":
                        name = bname
                        if name in mask_props_done:
                            continue
                        mask_props_done.add(name)
                    else:
                        name = f"{bname}_ch{ch}" if n_channels > 1 else bname

                    names.add(name)

    return names


def compute_properties_worker(
    buf_idx: int,
    img_path: str | list[str],
    mask_path: str | None,
    property_functions: dict[str, list[Callable]],
    any_requires_image: bool,
    any_requires_mask: bool,
) -> tuple[int, dict[str, float], dict[str, float]]:
    """Worker for parallel property computation (module-level, picklable).

    Returns
    -------
    buf_idx
        Local buffer index in the current batch.
    result
        Phenotypic scalars {prop_name: value}.
    internal
        Per-row checkpoint state (e.g. ``_properties_attempted``), stored in HDF5 ``/internal``.
    """
    img_stack, mask_stack = _load_image_and_mask_stacks(
        img_path, mask_path, any_requires_image, any_requires_mask, property_functions
    )
    n_img_ch = 0 if img_stack is None else img_stack.shape[-1]
    n_mask_ch = 0 if mask_stack is None else mask_stack.shape[-1]
    start_ch, end_ch = _determine_channel_range(property_functions, n_img_ch, n_mask_ch)
    n_channels = end_ch - start_ch

    result: dict[str, float] = {}
    mask_props_done: set = set()
    has_img_fn = any(r in ("image", "both") for r in property_functions)
    has_mask_fn = any(r in ("mask", "both", "any") for r in property_functions)
    only_mask = "mask" in property_functions and not has_img_fn and "any" not in property_functions

    for ch in range(start_ch, end_ch):
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

        for req_type, fn_list in property_functions.items():
            if req_type == "mask" and ch != start_ch:
                continue
            if not _should_call_property_function(req_type, image2d, mask2d):
                continue

            for fn in fn_list:
                try:
                    feat_dict = fn(image2d, mask2d)
                    if not isinstance(feat_dict, dict):
                        continue
                except Exception:
                    feat_dict = {}

                for bname, value in feat_dict.items():
                    if req_type == "mask":
                        name = bname
                        if name in mask_props_done:
                            continue
                        mask_props_done.add(name)
                    else:
                        name = f"{bname}_ch{ch}" if n_channels > 1 else bname

                    result[name] = float(value) if value is not None else np.nan

    internal = {"_properties_attempted": 1.0}
    return (buf_idx, result, internal)
