"""
Path utilities for checkpoint/dedup and file handling.

Consolidates path representation, basename extraction, and primary path logic.
"""

import contextlib
import json
import os
import posixpath
from typing import Any


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


def stem_identifier_path(path: Any) -> str:
    """Return a normalized path identifier without the final extension."""
    normalized = normalize_identifier_path(path)
    if not normalized:
        return ""
    parent = posixpath.dirname(normalized)
    stem = posixpath.splitext(posixpath.basename(normalized))[0]
    return f"{parent}/{stem}" if parent else stem


def filename_identifier_keys(path: Any, root: str | None = None) -> list[str]:
    """Build ordered portable keys for filename matching.

    Keys prefer root-relative paths, then the provided value, then basename
    fallbacks. Absolute roots are used only in memory to derive relative keys.
    """
    raw = str(path).strip()
    if not raw:
        return []

    keys: list[str] = []

    def add(value: Any) -> None:
        key = normalize_identifier_path(value)
        if key and key not in keys:
            keys.append(key)
        stem_key = stem_identifier_path(value)
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
