"""
PhenoMeResults — typed container for phenotyping pipeline results.

Replaces the plain ``Dict[str, Any]`` that was previously used as
``PhenoMe.results``.  It exposes a fully dict-compatible
interface (``__getitem__``, ``__setitem__``, ``get``, ``__contains__``,
``__iter__``) so that all existing mixin code keeps working without any
changes.  On top of that it provides typed attributes and convenience
properties for cleaner client code.

Embeddings lifecycle
--------------------
``embeddings`` is **always** ``None`` inside ``PhenoMeResults``.

* **Not yet computed** — ``embeddings is None``, ``pipeline._db is None``
* **Lazy-backed (HDF5)** — ``embeddings is None``, ``pipeline._db is not None``
* **Eagerly computed** — ``embeddings`` is an ``np.ndarray`` of shape ``(N, D)``

The former ``[]``-sentinel that mixed "uninitialised" with "lazy" with
"being filled during extraction" has been removed.  In-progress embeddings
during extraction live in ``pipeline._emb_buffer`` (a plain ``List[np.ndarray]``)
and never touch this container until they are finalised.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

# Keys recognised by the dict-compatible interface.
_VALID_KEYS = frozenset({"img_path", "metadata", "properties", "embeddings"})


class PhenoMeResults:
    """@section Results container
    @order 5

    Typed container for per-image phenotyping data.

    Parameters
    ----------
    img_path : list of str or list of list[str]
        Per-image file path(s).  Single-channel images use a plain ``str``;
        multi-channel images use a ``List[str]`` with one path per channel.
    metadata : list of dict
        Per-image metadata dicts (e.g. ``{"drug": "DMSO", "time": "24h"}``).
    properties : list of dict
        Per-image computed scalar properties
        (e.g. ``{"intensity_mean": 0.42, "area": 1024}``).
    embeddings : np.ndarray or None
        Shape ``(N, D)`` float32 array, or ``None`` when not available /
        when embeddings are stored lazily in an HDF5 file.
    """

    __slots__ = ("embeddings", "img_path", "metadata", "properties")

    def __init__(
        self,
        img_path: Sequence[str | Sequence[str]] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        properties: Sequence[dict[str, Any]] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self.img_path: list[str | list[str]] = (
            [list(p) if isinstance(p, (list, tuple)) else str(p) for p in img_path]
            if img_path is not None
            else []
        )
        self.metadata: list[dict[str, Any]] = list(metadata) if metadata is not None else []
        self.properties: list[dict[str, Any]] = list(properties) if properties is not None else []
        self.embeddings: np.ndarray | None = embeddings

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def n_images(self) -> int:
        """Number of images stored."""
        return len(self.img_path)

    @property
    def has_embeddings(self) -> bool:
        """True if embeddings are stored eagerly (not lazy or absent)."""
        return self.embeddings is not None and len(self.embeddings) > 0

    @property
    def has_properties(self) -> bool:
        """True if at least one image has computed properties."""
        return bool(self.properties) and any(bool(p) for p in self.properties)

    @property
    def property_keys(self) -> list[str]:
        """Sorted list of property names from the first non-empty properties dict."""
        for p in self.properties:
            if isinstance(p, dict) and p:
                return sorted(p.keys())
        return []

    @property
    def metadata_keys(self) -> list[str]:
        """Sorted unique metadata keys across all images (case-preserving)."""
        seen_lower: dict[str, str] = {}
        for m in self.metadata:
            if isinstance(m, dict):
                for k in m:
                    kl = k.lower()
                    if kl not in seen_lower:
                        seen_lower[kl] = k
        return sorted(seen_lower.values())

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality, or 0 if no embeddings are stored."""
        if self.embeddings is not None and self.embeddings.ndim == 2:
            return int(self.embeddings.shape[1])
        return 0

    # ------------------------------------------------------------------
    # Dict-compatible interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        if key == "img_path":
            return self.img_path
        if key == "metadata":
            return self.metadata
        if key == "properties":
            return self.properties
        if key == "embeddings":
            # Legacy code expects [] when embeddings are absent/lazy.
            # Return the array if present, else an empty list for compatibility.
            return self.embeddings if self.embeddings is not None else []
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "img_path":
            self.img_path = value
        elif key == "metadata":
            self.metadata = value
        elif key == "properties":
            self.properties = value
        elif key == "embeddings":
            # Accept np.ndarray or the empty-list sentinel from legacy code.
            if isinstance(value, np.ndarray):
                self.embeddings = value
            elif value is None or (isinstance(value, list) and len(value) == 0):
                self.embeddings = None
            else:
                # Non-empty list (legacy in-progress buffer) — kept for compatibility,
                # but warn: use pipeline._emb_buffer instead.
                self.embeddings = None
        else:
            raise KeyError(f"Unknown results key '{key}'. Valid keys: {sorted(_VALID_KEYS)}")

    def __contains__(self, key: object) -> bool:
        return key in _VALID_KEYS

    def __iter__(self) -> Iterator[str]:
        return iter(_VALID_KEYS)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style .get() with default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> Iterable[str]:
        """Return the valid dict-style keys for this results container."""
        return _VALID_KEYS

    def items(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(key, value)`` pairs for each valid key."""
        for k in _VALID_KEYS:
            yield k, self[k]

    def values(self) -> Iterator[Any]:
        """Yield values for each valid key."""
        for k in _VALID_KEYS:
            yield self[k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all fields to empty state."""
        self.img_path = []
        self.metadata = []
        self.properties = []
        self.embeddings = None

    def primary_path(self, idx: int) -> str:
        """Return the primary (or only) file path for image *idx*."""
        p = self.img_path[idx]
        return p[0] if isinstance(p, list) else p

    def image_name(self, idx: int) -> str:
        """Return the basename of the primary path for image *idx*."""
        return os.path.basename(self.primary_path(idx))

    def metadata_value(self, idx: int, key: str) -> Any:
        """Case-insensitive metadata lookup for image *idx*."""
        if idx >= len(self.metadata):
            return None
        m = self.metadata[idx]
        if not isinstance(m, dict):
            return None
        key_lower = key.lower()
        for k, v in m.items():
            if k.lower() == key_lower:
                return v
        return None

    def property_value(self, idx: int, key: str) -> float | None:
        """Return a scalar property for image *idx*, or None if absent."""
        if idx >= len(self.properties):
            return None
        p = self.properties[idx]
        return p.get(key) if isinstance(p, dict) else None

    def rebase_paths(self, data_dir: str, old_data_dir: str | None = None) -> None:
        """Update all image paths by replacing *old_data_dir* with *data_dir*.

        This is useful when moving a checkpoint and its dataset to a different
        machine or directory.

        Parameters
        ----------
        data_dir : str
            The new base directory where the images are located.
        old_data_dir : str, optional
            The old base directory to be replaced. If not provided, it is
            automatically detected by finding the common prefix of all
            stored paths.  Special value "relative" indicates that stored
            paths are already relative and just need to be joined with
            *data_dir*.
        """
        if not self.img_path:
            return

        if old_data_dir == "relative":
            # Stored paths are relative, just prepend the new data_dir
            new_paths: list[str | list[str]] = []
            for p in self.img_path:
                if isinstance(p, list):
                    new_paths.append([os.path.join(data_dir, ch) for ch in p])
                else:
                    new_paths.append(os.path.join(data_dir, str(p)))
            self.img_path = new_paths
            return

        if old_data_dir is None:
            # Detect old root: find common prefix of all primary paths
            primary_paths = [self.primary_path(i) for i in range(self.n_images)]
            # Filter out empty paths for commonpath
            valid_dirs = [os.path.dirname(p) for p in primary_paths if p]
            if len(valid_dirs) > 1:
                try:
                    old_data_dir = os.path.commonpath(valid_dirs)
                except ValueError:
                    old_data_dir = ""
            elif len(valid_dirs) == 1:
                old_data_dir = valid_dirs[0]
            else:
                old_data_dir = ""

        rebased_paths: list[str | list[str]] = []
        for p in self.img_path:
            if isinstance(p, list):
                new_p: str | list[str] = [
                    os.path.join(data_dir, os.path.relpath(ch, old_data_dir))
                    if old_data_dir and os.path.isabs(ch)
                    else os.path.join(data_dir, ch)
                    for ch in p
                ]
                rebased_paths.append(new_p)
            else:
                if old_data_dir and os.path.isabs(str(p)):
                    new_p = os.path.join(data_dir, os.path.relpath(str(p), old_data_dir))
                else:
                    new_p = os.path.join(data_dir, str(p))
                rebased_paths.append(new_p)

        self.img_path = rebased_paths

    def __len__(self) -> int:
        return self.n_images

    def __repr__(self) -> str:
        emb_info = (
            f"embeddings=({self.n_images}, {self.embedding_dim})"
            if self.has_embeddings
            else "embeddings=None"
        )
        prop_keys = self.property_keys
        prop_info = f"{len(prop_keys)} properties" if prop_keys else "no properties"
        return (
            f"PhenoMeResults("
            f"n_images={self.n_images}, "
            f"{emb_info}, "
            f"metadata_keys={self.metadata_keys}, "
            f"{prop_info})"
        )
