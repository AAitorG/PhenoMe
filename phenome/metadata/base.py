"""
Base metadata class for the phenotyping pipeline.

Provides configurable column mappings, auto-generated unique IDs, and mask
path resolution. Specialized metadata sources inherit from MetadataBase.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

from ..core.property_utils import _is_path_like_value
from ..utils.path_utils import strip_known_extension


def _sanitize_id(raw: str) -> str:
    """Sanitize string for use as filesystem/JSON-safe unique ID."""
    if not raw or not isinstance(raw, str):
        return "unknown"
    # Replace path separators and problematic chars with underscore
    s = re.sub(r"[/\\:\s]+", "_", raw.strip())
    s = re.sub(r"_+", "_", s)  # collapse multiple underscores
    s = s.strip("_")
    return s if s else "unknown"


def _path_to_id_single(path: str, data_dir: str | None) -> str:
    """Generate human-readable ID from single file path."""
    path_norm = str(path).replace("\\", "/")

    if data_dir:
        try:
            data_dir_norm = str(data_dir).replace("\\", "/")
            rel = os.path.relpath(path_norm, data_dir_norm)
            rel = rel.replace("\\", "/")

            # If relpath escapes out completely using many '../' or is absolute, skip it
            if not rel.startswith("../") and not os.path.isabs(rel) and rel != ".":
                # Stricter validation to catch subtle traversal attempts
                if os.path.normpath(rel) != rel:
                    pass
                else:
                    parts = rel.split("/")
                    if parts:
                        parts[-1] = strip_known_extension(parts[-1])
                    raw = "/".join(parts)
                    return _sanitize_id(raw)
        except ValueError:
            pass

    # Fallback: include parent directory + stem for disambiguation without full abs path.
    parent = os.path.basename(os.path.dirname(path_norm))
    fallback_stem = strip_known_extension(os.path.basename(path_norm))
    if parent and parent not in (".", ""):
        return _sanitize_id(f"{parent}/{fallback_stem}")
    return _sanitize_id(fallback_stem)


def _paths_to_id_multi(paths: str | list[str], data_dir: str | None) -> str:
    """Generate composite ID from multi-channel paths (sorted for determinism)."""
    if isinstance(paths, str):
        return _path_to_id_single(paths, data_dir)
    stems = []
    for p in paths:
        id_part = _path_to_id_single(str(p), data_dir)
        stems.append(id_part)
    stems = sorted(stems)
    composite = "|".join(stems)
    return _sanitize_id(composite)


class MetadataBase(ABC):
    """@section Metadata classes
    @order 10

    Base class for metadata extraction with configurable columns and ID handling.

    Subclasses implement `_extract` to produce raw metadata; the base
    class ensures a unique ``id`` is always present and provides stable key
    generation for checkpoint matching.

    Parameters
    ----------
    filename_columns : str or list of str
        Column(s) for image filenames. List for multi-channel.
    unique_id_column : str
        Column name for unique sample ID (default ``"id"``).
    mask_filename_column : str, optional
        Column name for mask filename when metadata has explicit mask.
    mask_dir : str, optional
        Root directory for mask resolution.
    data_dir : str, optional
        Base directory for relative path ID generation.
    """

    def __init__(
        self,
        filename_columns: str | list[str] = "filename",
        unique_id_column: str = "id",
        mask_filename_column: str | None = None,
        mask_dir: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        if isinstance(filename_columns, str):
            self._filename_columns: list[str] = [filename_columns]
        else:
            self._filename_columns = list(filename_columns)
        if not self._filename_columns:
            raise ValueError("filename_columns must be non-empty.")
        self._unique_id_column = unique_id_column
        self._mask_filename_column = mask_filename_column
        self._mask_dir = mask_dir
        self._data_dir = data_dir
        self._group_by = self._filename_columns[0]

    @property
    def group_by(self) -> str:
        """Column used for multi-channel grouping in find_files."""
        return self._group_by

    @property
    def mask_dir(self) -> str | None:
        """Root directory for mask files."""
        return self._mask_dir

    @property
    def mask_filename_column(self) -> str | None:
        """Column containing mask filename when using explicit mask lookup."""
        return self._mask_filename_column

    def get_id(self, meta: dict[str, Any]) -> str:
        """Return the unique ID from metadata.

        Raises
        ------
        KeyError
            If the unique ID column is missing from meta.
        """
        key = self._unique_id_column
        for k, v in meta.items():
            if k.lower() == key.lower():
                return str(v)
        raise KeyError(f"Metadata missing unique ID column '{self._unique_id_column}'.")

    def ensure_id(
        self,
        meta: dict[str, Any],
        paths: str | list[str],
        data_dir: str | None = None,
    ) -> str:
        """Inject or return unique ID in meta. Auto-generate if not present.

        Parameters
        ----------
        meta : dict
            Metadata dict (modified in place).
        paths : str or list of str
            File path(s) for this sample.
        data_dir : str, optional
            Base directory for ID generation. Uses instance data_dir if None.

        Returns
        -------
        str
            The unique ID (existing or newly generated).
        """
        base_dir = data_dir if data_dir is not None else self._data_dir
        key = self._unique_id_column
        key_lower = key.lower()

        # Check if meta already has a valid ID
        for k, v in meta.items():
            if k.lower() == key_lower and v is not None:
                v_str = str(v).strip()
                if v_str and not _is_path_like_value(v_str):
                    return v_str

        # Auto-generate from path(s)
        generated = _paths_to_id_multi(paths, base_dir)
        meta[key] = generated
        return generated

    def to_stable_key(self, meta: dict[str, Any]) -> str:
        """Build stable key for checkpoint matching.

        Uses the unique ID when present and non-path-like; otherwise delegates
        to metadata_to_stable_key for a JSON-based stable key.
        """
        from ..core.property_utils import metadata_to_stable_key

        key = self._unique_id_column
        for k, v in meta.items():
            if k.lower() == key.lower() and v is not None:
                v_str = str(v).strip()
                if v_str and not _is_path_like_value(v_str):
                    return v_str

        return metadata_to_stable_key(meta)

    def get_mask_path(self, meta: dict[str, Any]) -> str | None:
        """Resolve mask path from metadata when mask_filename_column and mask_dir are set.

        Returns
        -------
        str or None
            Full path to mask file, or None if not resolvable.
        """
        if not self._mask_dir or not self._mask_filename_column:
            return None

        mask_fname = None
        for k, v in meta.items():
            if k.lower() == self._mask_filename_column.lower() and v:
                mask_fname = str(v).strip()
                break

        if not mask_fname:
            return None

        full = os.path.join(self._mask_dir, mask_fname)
        return full if os.path.isfile(full) else None

    def metadata_fn(
        self,
        path: str,
        data_dir: str | None = None,
    ) -> dict[str, Any]:
        """Extract metadata for a path. Guarantees 'id' is present.

        Parameters
        ----------
        path : str
            File path.
        data_dir : str, optional
            Base directory for ID generation.

        Returns
        -------
        dict
            Metadata with file_path, metadata keys, and guaranteed id.
        """
        meta = self._extract(path)
        if not meta:
            return meta
        paths = meta.get("file_path", path)
        self.ensure_id(meta, paths, data_dir)
        return meta

    @abstractmethod
    def _extract(self, path: str) -> dict[str, Any]:
        """Extract raw metadata for path. Subclasses implement this."""
        pass
