"""
Path utilities for checkpoint/dedup and file handling.

Consolidates path representation, basename extraction, and primary path logic.
"""

import json
import os
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
    return os.path.basename(primary)
