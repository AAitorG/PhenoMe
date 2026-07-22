"""
File discovery and data inspecting for the phenotyping pipeline.
"""

import os
from collections import Counter, defaultdict
from collections.abc import Callable
from glob import glob
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .._logging import get_logger
from ..metadata.base import MetadataBase
from ..utils.path_utils import filename_identifier_keys

__all__ = ["FileDiscovery", "ensure_hwc", "read_image"]

logger = get_logger(__name__)


def read_image(path: str | list[str]) -> np.ndarray:
    """
    @section Image I/O and checkpoints
    @order 40

    Read an image and return as a numpy array.

    Supports TIFF (via tifffile), NIfTI (via nibabel), Numpy arrays (.npy, .npz), and
    common formats (PNG, JPEG, BMP, etc. via OpenCV).
    When ``path`` is a list of paths (e.g. one per channel), each file is read and
    concatenated along the channel axis, yielding shape (H, W, C_total).

    Args:
        path: Path to the image file, or list of paths (one per channel) for
            multi-channel datasets where each channel is in a separate file.

    Returns:
        np.ndarray: Shape (H, W) or (H, W, C), dtype float32. All channels returned.
    """
    if isinstance(path, list):
        if not path:
            raise ValueError("read_image(path): path list cannot be empty.")
        imgs = [read_image(p) for p in path]
        # Ensure each is (H, W, C) with C at least 1, then concat on last axis
        hwc = [ensure_hwc(np.array(im, dtype=np.float32)) for im in imgs]
        return np.concatenate(hwc, axis=-1)

    ext = os.path.splitext(path)[1].lower()
    path_lower = path.lower()

    if ext == ".npy":
        img = np.load(path, allow_pickle=False)
        if img.dtype == object:
            raise ValueError(f".npy at {path} has object dtype; only numeric arrays are supported.")
    elif ext == ".npz":
        with np.load(path) as data:
            img = data[data.files[0]]
    elif path_lower.endswith(".nii.gz") or path_lower.endswith(".nii"):
        try:
            import nibabel as nib
            from nibabel.spatialimages import SpatialImage

            nii = nib.load(path)
            if not isinstance(nii, SpatialImage):
                raise TypeError(f"Unsupported NIfTI type: {type(nii).__name__}")
            img = nii.get_fdata()
        except ImportError as e:
            raise ImportError("nibabel is required to read NIfTI files.") from e
        except Exception as e:
            raise ValueError(f"Could not read NIfTI image at {path}: {e}") from e
    elif ext in {".tif", ".tiff"}:
        try:
            import tifffile

            img = tifffile.imread(path)
        except Exception as e:
            raise ValueError(f"Could not read TIFF image at {path}: {e}") from e
    else:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image at {path}")
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    img = np.array(img, dtype=np.float32)

    return img


def ensure_hwc(img: np.ndarray) -> np.ndarray:
    """@section Image I/O and checkpoints
    @order 50

    Ensure image is in (H, W, C) format.

    Args:
        img: np.ndarray. Supported shapes: (H, W), (C, H, W), (H, W, C).
            For (C, H, W), transposes to (H, W, C) using smallest dim as C.

    Returns:
        np.ndarray: Shape (H, W, C). Single channel gets (H, W, 1).
    """
    if img.ndim == 2:
        return img[..., np.newaxis]
    if img.ndim == 3:
        # Heuristic: the channel axis is the smallest dimension, but only
        # transpose when the smallest dimension is clearly a channel count
        # (≤ 16) to avoid misinterpreting a narrow spatial dimension.
        min_dim = int(np.argmin(img.shape))
        min_size = img.shape[min_dim]
        if min_size <= 16:
            if min_dim == 0:
                return np.transpose(img, (1, 2, 0))
            if min_dim == 1:
                return np.transpose(img, (0, 2, 1))
    return img


class FileDiscovery:
    """@section File discovery
    @order 80

    Finds image files and extracts metadata from directories."""

    def _find_files_from_dir(
        self,
        data_dir: str,
        extensions: list[str] | None = None,
        metadata_fn: Callable[[str], dict[str, Any]] | MetadataBase | None = None,
        on_missing_metadata: str = "drop",
        mask_dir: str | None = None,
        mask_filename_column: str | None = None,
        mask_extensions: list[str] | None = None,
    ) -> pd.DataFrame:
        """Discover image files from a directory (internal use only)."""
        # Resolve actual dataset root for robust ID generation when parsing globs
        effective_data_dir = data_dir
        if "*" in data_dir or "?" in data_dir:
            effective_data_dir = data_dir.split("*")[0].split("?")[0]
            if not effective_data_dir.endswith("/"):
                effective_data_dir = os.path.dirname(effective_data_dir)

        if metadata_fn is None:
            from ..metadata.sources import DefaultMetadata

            _default_meta = DefaultMetadata()

            def fn(p: str) -> dict[str, Any]:
                return _default_meta.metadata_fn(p, data_dir=effective_data_dir)

        elif isinstance(metadata_fn, MetadataBase):

            def _metadata_extractor(p: str) -> dict[str, Any]:
                return metadata_fn.metadata_fn(p, data_dir=effective_data_dir)

            fn = _metadata_extractor
            fn.group_by = metadata_fn.group_by  # type: ignore[attr-defined]
        elif hasattr(metadata_fn, "_metadata_source"):
            meta_src = metadata_fn._metadata_source

            def _metadata_extractor(p: str) -> dict[str, Any]:
                return meta_src.metadata_fn(p, data_dir=effective_data_dir)

            fn = _metadata_extractor
            fn.group_by = meta_src.group_by  # type: ignore[attr-defined]
        else:
            fn = metadata_fn

        # MetadataBase alignment: use instance mask_dir/mask_filename_column if not passed
        base_meta = getattr(metadata_fn, "_metadata_source", metadata_fn)
        if isinstance(base_meta, MetadataBase):
            if mask_dir is None and getattr(base_meta, "mask_dir", None):
                mask_dir = base_meta.mask_dir
            if mask_filename_column is None and getattr(base_meta, "mask_filename_column", None):
                mask_filename_column = base_meta.mask_filename_column

        # Support both directory (recursive walk) and glob pattern
        if "*" in data_dir or "?" in data_dir:
            raw_fnames = sorted(glob(data_dir))
            ext_set = None if extensions is None else {e.lower() for e in extensions}
            fnames = []
            for fp in raw_fnames:
                if not os.path.isfile(fp):
                    continue
                if ext_set is not None and os.path.splitext(fp)[1].lower() not in ext_set:
                    continue
                if os.path.islink(fp):
                    logger.warning("Skipping symlink: %s", fp)
                    continue
                fnames.append(fp)
        else:
            fnames = []
            for root, _, files in os.walk(data_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.islink(fp):
                        logger.warning("Skipping symlink: %s", fp)
                        continue
                    if extensions is None or os.path.splitext(f)[1].lower() in {
                        e.lower() for e in extensions
                    }:
                        fnames.append(fp)
            fnames.sort()

        logger.info("Found %d files total.", len(fnames))

        if len(fnames) == 0:
            logger.warning(
                "No files found. Please ensure that the filename column in the metadata "
                "does not contain the file format/extension (e.g., use 'image_001' instead of 'image_001.tif')."
            )

        # Extract metadata per file; drop or keep based on on_missing_metadata
        rows: list[dict[str, Any]] = []
        bad: list[str] = []
        for path in fnames:
            try:
                meta = fn(path)
                if metadata_fn is not None:
                    is_missing = not meta or not any(k.lower() != "file_path" for k in meta)
                else:
                    is_missing = not meta
            except (ValueError, KeyError, FileNotFoundError):
                meta = {}
                is_missing = True
            except Exception as exc:
                raise RuntimeError(f"Metadata extraction failed for {path!r}: {exc}") from exc

            if is_missing:
                bad.append(path)
                if on_missing_metadata == "drop":
                    continue

            row: dict[str, Any] = {"file_path": meta.get("file_path", path)}
            for k, v in meta.items():
                if k.lower() != "file_path":
                    row[k.lower()] = v
            rows.append(row)

        if bad:
            logger.warning(
                "%d files have missing/failed metadata (%s).",
                len(bad),
                on_missing_metadata,
            )

        # If metadata includes channel_index, group into one row per sample
        # with file_path = list of paths in channel order. Use the column specified
        # by metadata_fn.group_by (set by make_dataframe_metadata_fn or
        # get_metadata_from_path), default "filename". If absent, deduce from row keys.
        # Note: row keys are lowercased; resolve group_key case-insensitively.
        if rows and "channel_index" in rows[0]:
            group_key = getattr(fn, "group_by", "filename")
            if group_key not in rows[0]:
                # Resolve case-insensitively (row keys are lowercased)
                group_key_lower = group_key.lower()
                matching = [k for k in rows[0] if k.lower() == group_key_lower]
                if matching:
                    group_key = matching[0]
                else:
                    # Deduce: first metadata key that is not a system key
                    sys_keys = {"channel_index", "file_path"}
                    group_key = next(
                        (k for k in rows[0] if k not in sys_keys),
                        None,
                    )
            if group_key is None:
                raise ValueError(
                    "Multi-channel files detected (channel_index present) but no grouping "
                    "column found. Set group_by on the metadata source (e.g. filename or id)."
                )
            if group_key is not None:
                n_files = len(rows)
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    groups[row[group_key]].append(row)
                merged_rows: list[dict[str, Any]] = []
                for _, group in groups.items():
                    group.sort(key=lambda r: r["channel_index"])
                    first = group[0].copy()
                    paths_list = [r["file_path"] for r in group]
                    first["file_path"] = paths_list
                    # Drop channel_index from merged row (no longer per-file)
                    first.pop("channel_index", None)
                    # Re-ensure composite ID from all channel paths when using MetadataBase
                    meta_src = getattr(metadata_fn, "_metadata_source", metadata_fn)
                    if isinstance(meta_src, MetadataBase):
                        meta_src.ensure_id(first, paths_list, data_dir)
                    merged_rows.append(first)
                rows = merged_rows
                logger.info(
                    "Grouped %d channel files into %d samples (multi-channel mode).",
                    n_files,
                    len(rows),
                )

        # Mask discovery
        rows = self._resolve_mask_paths_for_rows(
            rows=rows,
            data_dir=data_dir,
            fnames=fnames,
            mask_dir=mask_dir,
            mask_filename_column=mask_filename_column,
            mask_extensions=mask_extensions,
        )

        df = pd.DataFrame(rows)
        logger.info("DataFrame: %d files, %d columns.", len(df), len(df.columns))

        # Warn if files were found but none matched metadata
        if len(fnames) > 0 and len(df) == 0 and metadata_fn is not None:
            logger.warning(
                "Files were found (%d) but none matched the metadata. "
                "Please ensure that the filename column in the metadata "
                "does not contain the file format/extension (e.g., use 'image_001' instead of 'image_001.tif').",
                len(fnames),
            )

        return df

    def _resolve_mask_paths_for_rows(
        self,
        rows: list[dict[str, Any]],
        data_dir: str,
        fnames: list[str],
        mask_dir: str | None,
        mask_filename_column: str | None,
        mask_extensions: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Resolve mask paths for each row and add mask_path column."""
        if mask_dir is None or not rows:
            for row in rows:
                row["mask_path"] = None
            return rows

        # Infer effective data_dir for relative path matching (handles glob patterns)
        if "*" in data_dir or "?" in data_dir:
            if fnames:
                try:
                    dirs = [os.path.dirname(p) for p in fnames if p]
                    effective_data_dir = (
                        os.path.commonpath(dirs)
                        if len(dirs) > 1
                        else (dirs[0] if dirs else os.path.dirname(fnames[0]))
                    )
                except (ValueError, OSError):
                    effective_data_dir = os.path.dirname(fnames[0])
            else:
                effective_data_dir = data_dir.split("*")[0].rstrip("/\\") or "."
        else:
            effective_data_dir = data_dir

        # Build portable lookup keys. Root-relative keys disambiguate repeated
        # basenames; basename keys are used only when unambiguous.
        mask_fnames = sorted(glob(os.path.join(mask_dir, "**", "*.*"), recursive=True))
        mask_lookup: dict[str, str | None] = {}
        for mp in mask_fnames:
            for key in filename_identifier_keys(mp, mask_dir):
                existing = mask_lookup.get(key)
                if existing is None and key in mask_lookup:
                    continue
                if existing is not None and existing != mp:
                    mask_lookup[key] = None
                else:
                    mask_lookup[key] = mp

        def _lookup_mask(value: Any) -> str | None:
            for key in filename_identifier_keys(value):
                matched = mask_lookup.get(key)
                if matched:
                    return matched
            return None

        n_found = 0
        for row in rows:
            # Prefer existing mask_path from metadata (e.g. custom metadata_fn)
            existing = None
            for k, v in row.items():
                if k.lower() == "mask_path" and v and isinstance(v, str):
                    existing = v
                    break
            if existing and os.path.isfile(existing):
                row["mask_path"] = existing
                n_found += 1
                continue
            if existing:
                matched_existing = _lookup_mask(existing)
                if matched_existing and os.path.isfile(matched_existing):
                    row["mask_path"] = matched_existing
                    n_found += 1
                    continue

            # Resolve mask path
            row["mask_path"] = None
            primary_path = row["file_path"]
            if isinstance(primary_path, list):
                primary_path = primary_path[0]

            matched: str | None = None

            # Custom column lookup
            if mask_filename_column:
                mask_fname = None
                for k, v in row.items():
                    if k.lower() == mask_filename_column.lower() and v:
                        mask_fname = str(v).strip()
                        break
                if mask_fname:
                    full = os.path.join(mask_dir, mask_fname)
                    if os.path.isfile(full):
                        matched = full
                    if matched is None:
                        matched = _lookup_mask(mask_fname)

            # Default path-based matching
            if matched is None:
                try:
                    rel = os.path.relpath(primary_path, effective_data_dir)
                    candidate = os.path.join(mask_dir, rel)
                    if os.path.isfile(candidate):
                        matched = candidate
                except (ValueError, OSError):
                    pass

            # Try alternative extensions
            if matched is None and mask_extensions:
                try:
                    rel = os.path.relpath(primary_path, effective_data_dir)
                    stem, _ = os.path.splitext(rel)
                    for ext in mask_extensions:
                        ext = ext if ext.startswith(".") else f".{ext}"
                        candidate = os.path.join(mask_dir, stem + ext)
                        if os.path.isfile(candidate):
                            matched = candidate
                            break
                except (ValueError, OSError):
                    pass

            # Basename fallback
            if matched is None:
                matched = _lookup_mask(primary_path)

            if matched and os.path.isfile(matched):
                row["mask_path"] = matched
                n_found += 1

        if n_found > 0:
            logger.info(
                "Matched masks: %d/%d images (%d missing).",
                n_found,
                len(rows),
                len(rows) - n_found,
            )
        return rows

    def inspect_data(self, file_df: pd.DataFrame) -> pd.DataFrame:
        """Inspect image and mask dimensions, shapes, and data ranges.

        Reads each image (and mask when present) and reports: shape statistics,
        image-mask shape matching, and mask dtype/range. Masks are expected in
        the ``mask_path`` column of file_df when present.

        Args:
            file_df: pd.DataFrame from find_files(). Must have 'file_path' column.
                When masks exist, includes 'mask_path' column.

        Returns:
            pd.DataFrame with columns: file_path, height, width, channels, min, max,
            error, shape_match (True/False/None), mask_height, mask_width, mask_dtype,
            mask_min, mask_max, mask_error.
        """
        if not isinstance(file_df, pd.DataFrame):
            raise ValueError(f"file_df must be a DataFrame, got: {type(file_df)}")
        if "file_path" not in file_df.columns:
            raise ValueError("file_df must contain 'file_path' column.")

        mask_col = "mask_path" if "mask_path" in file_df.columns else None
        has_masks = False
        if mask_col is not None:
            valid_mask_vals = file_df[mask_col].dropna()
            has_masks = any(str(v).strip() for v in valid_mask_vals)

        group_counts: Counter = Counter()
        shape_counts: Counter = Counter()
        mm_counts: Counter = Counter()
        dtype_counts: Counter = Counter()
        hs, ws, cs = [], [], []
        rows = []
        n_shape_mismatch = 0
        mask_dtypes: Counter = Counter()
        mask_mins: list[float] = []
        mask_maxs: list[float] = []

        logger.info("Inspecting %d images%s...", len(file_df), " (with masks)" if has_masks else "")

        for _, row in tqdm(file_df.iterrows(), total=len(file_df), desc="Reading data"):
            fp_raw = row["file_path"]
            fp = fp_raw if isinstance(fp_raw, list) else str(fp_raw)
            mask_path: str | None = None
            if mask_col:
                mp = row.get(mask_col)
                if mp is not None and not (isinstance(mp, float) and np.isnan(mp)):
                    s = str(mp).strip()
                    if s:
                        mask_path = s

            img_h, img_w, img_c = np.nan, np.nan, np.nan
            img_lo, img_hi = np.nan, np.nan
            img_error = None
            shape_match = None
            mask_h, mask_w = np.nan, np.nan
            mask_dtype_val = None
            mask_lo, mask_hi = np.nan, np.nan
            mask_error = None

            try:
                img = ensure_hwc(read_image(fp))
                img_lo, img_hi = float(img.min()), float(img.max())
                dtype = img.dtype
                mm_counts[(img_lo, img_hi)] += 1
                dtype_counts[dtype] += 1
                img_h, img_w = img.shape[0], img.shape[1]
                img_c = img.shape[2] if img.ndim >= 3 else 1
                shape = (img_h, img_w, img_c)
                shape_counts[shape] += 1
                group_counts[(shape, img_lo, img_hi)] += 1
                hs.append(img_h)
                ws.append(img_w)
                cs.append(img_c)
            except Exception as exc:
                img_error = str(exc)
                logger.warning("Failed to read image %s: %s", fp, exc)

            if mask_path and os.path.isfile(mask_path):
                try:
                    mask = ensure_hwc(read_image(mask_path))
                    mask_h, mask_w = mask.shape[0], mask.shape[1]
                    mask_dtype_val = mask.dtype
                    mask_lo, mask_hi = float(mask.min()), float(mask.max())
                    mask_dtypes[mask_dtype_val] += 1
                    mask_mins.append(mask_lo)
                    mask_maxs.append(mask_hi)
                    if img_error is None and not (np.isnan(img_h) or np.isnan(img_w)):
                        match = (int(img_h), int(img_w)) == (int(mask_h), int(mask_w))
                        shape_match = bool(match)
                        if not match:
                            n_shape_mismatch += 1
                except Exception as exc:
                    mask_error = str(exc)
                    logger.warning("Failed to read mask %s: %s", mask_path, exc)

            rows.append(
                {
                    "file_path": fp if isinstance(fp, list) else str(fp),
                    "height": img_h,
                    "width": img_w,
                    "channels": img_c,
                    "min": img_lo,
                    "max": img_hi,
                    "error": img_error,
                    "shape_match": shape_match,
                    "mask_height": mask_h,
                    "mask_width": mask_w,
                    "mask_dtype": mask_dtype_val,
                    "mask_min": mask_lo,
                    "mask_max": mask_hi,
                    "mask_error": mask_error,
                }
            )

        if not group_counts:
            logger.warning("No valid images found.")
            return pd.DataFrame(rows)

        # Enhanced logging with detailed statistics
        if hs and ws and cs:
            logger.info(
                "Unique shapes: %d  |  H: %d-%d  |  W: %d-%d  |  C: %d-%d",
                len(shape_counts),
                min(hs),
                max(hs),
                min(ws),
                max(ws),
                min(cs),
                max(cs),
            )

            if shape_counts:
                logger.info("Top 5 shapes:")
                for i, (shape, count) in enumerate(shape_counts.most_common(5), 1):
                    h, w, c = shape
                    logger.info("  %d. (%d, %d, %d)  →  %d images", i, h, w, c, count)

            if shape_counts:
                biggest_shape = max(shape_counts.keys(), key=lambda s: s[0] * s[1] * s[2])
                smallest_shape = min(shape_counts.keys(), key=lambda s: s[0] * s[1] * s[2])
                logger.info(
                    "Biggest:   (%d, %d, %d)  [%d px]",
                    biggest_shape[0],
                    biggest_shape[1],
                    biggest_shape[2],
                    biggest_shape[0] * biggest_shape[1] * biggest_shape[2],
                )
                logger.info(
                    "Smallest:  (%d, %d, %d)  [%d px]",
                    smallest_shape[0],
                    smallest_shape[1],
                    smallest_shape[2],
                    smallest_shape[0] * smallest_shape[1] * smallest_shape[2],
                )

            if dtype_counts:
                logger.info("Top 3 dtypes:")
                for i, (dtype, count) in enumerate(dtype_counts.most_common(3), 1):
                    logger.info("  %d. %s  →  %d images", i, dtype, count)

            if mm_counts:
                logger.info("Top 3 data ranges [min, max]:")
                for i, ((lo, hi), count) in enumerate(mm_counts.most_common(3), 1):
                    logger.info("  %d. [%.2f, %.2f]  →  %d images", i, lo, hi, count)

            if rows:
                valid_mins = [r["min"] for r in rows if r["error"] is None]
                valid_maxs = [r["max"] for r in rows if r["error"] is None]
                if valid_mins and valid_maxs:
                    logger.info(
                        "Global intensity range: min=%.2f  max=%.2f",
                        min(valid_mins),
                        max(valid_maxs),
                    )

        if has_masks:
            logger.info("Image-mask shape mismatches: %d", n_shape_mismatch)
            if mask_dtypes:
                dtype_str = ", ".join(
                    f"{dtype} (n={cnt})" for dtype, cnt in mask_dtypes.most_common()
                )
                logger.info("Mask dtypes: %s", dtype_str)
            if mask_mins and mask_maxs:
                logger.info(
                    "Mask data range: min=%.2f  max=%.2f",
                    min(mask_mins),
                    max(mask_maxs),
                )

        return pd.DataFrame(rows)
