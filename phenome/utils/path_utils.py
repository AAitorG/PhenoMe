"""
Path utilities for checkpoint/dedup and file handling.

Consolidates path representation, basename extraction, and primary path logic.
"""

import contextlib
import json
import os
import posixpath
from typing import Any

# Only these suffixes are treated as file extensions when building match keys.
# Other dots (e.g. ``my.image``, ``plate.A01.well``) stay part of the stem.
_KNOWN_FILE_EXTENSIONS = frozenset(
    {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".webp",
        ".npy",
        ".npz",
        ".czi",
        ".nd2",
        ".nii",
        ".hdf5",
        ".h5",
        # Common bioimaging / slide formats
        ".lsm",
        ".oib",
        ".oif",
        ".ims",
        ".dv",
        ".mrc",
        ".st",
        ".ndpi",
        ".svs",
        ".jp2",
        ".fits",
        ".dcm",
        ".lif",
        ".vsi",
        ".flex",
    }
)
_COMPOUND_FILE_EXTENSIONS = (
    ".nii.gz",
    ".tiff.gz",
    ".tif.gz",
    ".ome.tiff",
    ".ome.tif",
)


def path_repr(fname: str | list[str] | tuple) -> str:
    """Stable string representation for checkpoint deduplication and file_df alignment.

    Handles single and multi-channel paths; multi-channel uses sorted JSON for consistency.
    """
    if isinstance(fname, (list, tuple)):
        # Preserve channel order (sorting can make distinct orderings collide).
        return json.dumps([str(x) for x in fname])
    return str(fname)


def primary_path(p: Any) -> str:
    """Return primary (first) path from a path or list of paths.

    For multi-channel paths, returns the first channel path.
    """
    return p[0] if isinstance(p, (list, tuple)) else str(p)


def path_basename(p: str | list[str]) -> str:
    """Return basename of primary path."""
    primary = p[0] if isinstance(p, list) else p
    return posixpath.basename(normalize_identifier_path(primary))


def normalize_identifier_path(path: Any) -> str:
    """Normalize a filename/path identifier using POSIX separators.

    This is for matching only. It accepts Windows, Unix, or mixed separators
    and avoids relying on the current OS path parser for metadata values.
    """
    s = str(path).strip().replace("\\", "/")
    if not s:
        return ""
    normalized = posixpath.normpath(s)
    if normalized == ".":
        return ""
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _looks_like_short_file_extension(ext: str) -> bool:
    """True for short alphanumeric suffixes (e.g. ``.lsm``), not dotted names like ``.image``."""
    if not ext or not ext.startswith("."):
        return False
    body = ext[1:]
    return 1 <= len(body) <= 4 and body.isalnum()


def strip_known_extension(name: str) -> str:
    """Strip a known image/file extension; leave other dots in the name intact.

    ``my.image.tif`` -> ``my.image``; ``my.image`` stays ``my.image``.
    """
    if not name:
        return name
    lower = name.lower()
    for compound in _COMPOUND_FILE_EXTENSIONS:
        if lower.endswith(compound):
            return name[: -len(compound)]
    root, ext = posixpath.splitext(name)
    if ext.lower() in _KNOWN_FILE_EXTENSIONS:
        return root
    return name


def _basename_stem_keys(basename: str) -> list[str]:
    """Primary allowlist stem plus legacy aliases for matching.

    - Compound names (e.g. ``sample.ome.tif``) also emit the last-suffix stem
      (``sample.ome``) so older dataframe keys still match.
    - Unknown short extensions (e.g. ``sample.xyz``) also emit the last-dot stem
      (``sample``) for the same reason. Dotted identifiers without a short file
      suffix (e.g. ``my.image``) are left intact.
    """
    if not basename:
        return []
    keys: list[str] = []
    primary = strip_known_extension(basename)
    if primary:
        keys.append(primary)

    lower = basename.lower()
    compound_hit = any(lower.endswith(c) for c in _COMPOUND_FILE_EXTENSIONS)
    root, ext = posixpath.splitext(basename)
    if not root:
        return keys

    if compound_hit:
        if root not in keys:
            keys.append(root)
    elif (
        ext
        and ext.lower() not in _KNOWN_FILE_EXTENSIONS
        and _looks_like_short_file_extension(ext)
        and root not in keys
    ):
        keys.append(root)
    return keys


def stem_identifier_path(path: Any) -> str:
    """Return a normalized path identifier without a known file extension.

    Unlike ``os.path.splitext``, only recognized image/data extensions are
    removed, so names that contain dots but no extension (e.g. ``my.image``)
    are preserved.
    """
    normalized = normalize_identifier_path(path)
    if not normalized:
        return ""
    parent = posixpath.dirname(normalized)
    stem = strip_known_extension(posixpath.basename(normalized))
    return f"{parent}/{stem}" if parent else stem


def filename_identifier_keys(path: Any, root: str | None = None) -> list[str]:
    """Build ordered portable keys for filename matching.

    Keys prefer root-relative paths, then the provided value, then basename
    fallbacks. Absolute roots are used only in memory to derive relative keys.
    Stem aliases include compound and unknown-extension legacy forms.
    """
    raw = str(path).strip()
    if not raw:
        return []

    keys: list[str] = []

    def add(value: Any) -> None:
        key = normalize_identifier_path(value)
        if key and key not in keys:
            keys.append(key)
        normalized = normalize_identifier_path(value)
        if not normalized:
            return
        parent = posixpath.dirname(normalized)
        base = posixpath.basename(normalized)
        for stem in _basename_stem_keys(base):
            stem_key = f"{parent}/{stem}" if parent else stem
            if stem_key and stem_key not in keys:
                keys.append(stem_key)

    if root:
        with contextlib.suppress(ValueError, OSError):
            rel = os.path.relpath(os.path.abspath(raw), os.path.abspath(root))
            rel_key = normalize_identifier_path(rel)
            if rel_key and not rel_key.startswith("../") and rel_key != "..":
                add(rel)

    add(raw)

    basename = posixpath.basename(normalize_identifier_path(raw))
    if basename:
        add(basename)

    return keys
