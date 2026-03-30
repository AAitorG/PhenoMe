"""
Phenotyping Checkpoint Manager — HDF5 schema
=============================================

Provides crash-safe, incremental persistence for embeddings, metadata,
and properties using HDF5.  The core safety invariant is a **committed count**
attribute that is updated *after* all data for a batch has been flushed to
disk.  On reload, only the first ``n_committed`` rows are trusted; any
trailing rows left behind by an interrupted write are silently discarded.

All compression is **lossless** (gzip).  Float32 embedding data is stored
bit-for-bit exactly; gzip only removes redundancy without altering values.

Paths are always stored as POSIX relative (forward slashes) regardless of OS.

Version configuration (Constants section):
  CHECKPOINT_FORMAT_VERSION: version written to new checkpoints
  CHECKPOINT_SUPPORTED_VERSIONS: set of versions that can be loaded

HDF5 schema
-----------
/embeddings              (N, D)  float32, chunked, lossless gzip
/img_path                (N,)    vlen UTF-8  — primary/only path per image
/img_path_channels       (N, C)  vlen UTF-8  — present only for multi-channel images;
                                              each row lists the C channel paths
/metadata/
    {key}                (N,)    vlen UTF-8 or float32 — one dataset per metadata key
/properties/
    {name}               (N,)    float32     — one dataset per property name
/config/                                    — processing parameters (typed attributes)
    channel_mode         str  attribute
    resize_size          str  attribute  ("none" when absent)
    pad_size             str  attribute  ("none" when absent)
    force_rgb            bool attribute
    channels             (C,) int8 dataset — absent when all channels used

Root attributes
    version              str   — CHECKPOINT_FORMAT_VERSION
    n_committed          int   — rows committed for paths + embeddings + metadata
    n_committed_props    int   — rows committed for properties (may lag n_committed)
    embedding_dim        int   — 0 when no embeddings
    is_multichannel      bool  — True when /img_path_channels is present

External access (no custom code required)
-----------------------------------------
    import h5py, numpy as np
    with h5py.File("results.h5", "r") as f:
        paths      = f["img_path"][:]
        drug       = f["metadata/drug"][:]          # vlen UTF-8 array
        intensity  = f["properties/intensity_mean_ch0"][:]  # float32 array
        embeddings = f["embeddings"][:]             # (N, D) float32
        channel_mode = f["config"].attrs["channel_mode"]
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from types import TracebackType
from typing import Any, Literal, cast

import h5py
import numpy as np

from ..core.pipeline_results import PhenoMeResults
from ._checkpoint_ops import (
    append_2d_vlen_dataset,
    append_metadata_group,
    append_properties_group,
    append_vlen_dataset,
    decode,
    decode_list,
    read_metadata_group,
    read_properties_group,
    truncate_group_datasets,
    write_2d_vlen,
    write_metadata_group,
    write_properties_group,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Version configuration: change these to bump the format or adjust compatibility.
CHECKPOINT_FORMAT_VERSION = "2.0"  # Version written to new checkpoints
CHECKPOINT_SUPPORTED_VERSIONS = frozenset(
    {"2.0"}
)  # Versions that can be loaded; add older (e.g. "2.1") for backward compat

_CHUNK_ROWS = 128  # HDF5 chunk size (rows) for embeddings
_GZIP_LEVEL = 4  # compression level (1-9)
_PROP_CHUNK = 256  # chunk size for property / metadata columns
_VL_STR = h5py.special_dtype(vlen=str)
# HDF5 root attribute name for portable path resolution. Stored in f.attrs.
# Possible values: str (absolute base dir) | absent (None) — when present, image
# paths in the checkpoint are stored relative to this root; on load they are
# joined with it to resolve to absolute paths.
_STORAGE_ROOT_ATTR = "_storage_root"

# Metadata keys excluded from checkpoint (redundant or not useful for persistence)
_METADATA_EXCLUDE_KEYS = frozenset({"mask_path"})


def _is_checkpoint_version_supported(version: str) -> bool:
    """Return True if the given checkpoint version can be loaded."""
    return version in CHECKPOINT_SUPPORTED_VERSIONS


def _normalize_path_for_storage(path: str) -> str:
    """Normalize a path for portable storage: POSIX separators, no leading slash.

    Ensures checkpoints are portable across Windows and Unix. Replaces platform
    separators with '/', collapses redundant slashes, and strips leading slash
    for relative paths.
    """
    if not path:
        return path
    # Use posixpath for consistent / handling regardless of os.sep
    normalized = path.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    # Strip leading slash for relative paths
    return normalized.lstrip("/")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConcurrentCheckpointAccessError(BlockingIOError):
    """Raised when a checkpoint cannot be opened because it is locked by another process.

    HDF5 files use file locking; only one process (e.g. one notebook kernel or script)
    can have a checkpoint open for writing or lazy-loading at a time. If you see this
    error, another notebook or script has the checkpoint open — close it first, or use
    a different checkpoint path.
    """

    def __init__(self, path: str, cause: BaseException | None = None) -> None:
        self.path = path
        msg = (
            "The file is currently locked by another instance (another notebook, script, or kernel). \n"
            "Please close the other session that has it open, or use a different checkpoint path. \n"
            "File"
        )
        super().__init__(11, msg, path)
        if cause is not None:
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Crash-safe incremental checkpoint backed by a single HDF5 file.

    Parameters
    ----------
    path : str
        Filesystem path for the checkpoint file.  If it already exists it is
        opened in append mode and validated; otherwise a new file is created.
    embedding_dim : int or None
        Dimensionality of the embedding vectors.  Required when *creating* a
        new file.  Ignored when opening an existing one.
    processing_params : dict or None
        Pipeline parameters used to produce embeddings
        (``channel_mode``, ``channels``, ``resize_size``, ``pad_size``,
        ``force_rgb``).  Stored in ``/config`` so that a resumed run can
        verify the same settings are being used.
    """

    FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION  # Same as module constant; used when writing

    _PARAM_KEYS = (
        "channel_mode",
        "channels",
        "resize_size",
        "pad_size",
        "force_rgb",
    )

    # Path-like keys never written to /config (safeguard if _PARAM_KEYS is extended).
    _CONFIG_EXCLUDED_KEYS = frozenset({"data_dir", "checkpoint_path", "output_dir"})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        path: str,
        embedding_dim: int | None = None,
        processing_params: dict[str, Any] | None = None,
        lazy: bool = True,
    ) -> None:
        self.path = path
        self.lazy = lazy
        self._file: h5py.File | None = None
        self._embedding_dim: int | None = embedding_dim
        # Inferred root for resolving relative paths (set when writing or from HDF5)
        self._storage_root: str | None = None
        self._ram_data: PhenoMeResults | None = None
        self._n_committed_props_ram: int = 0
        self._is_multichannel_ram: bool = False
        self._processing_params_ram: dict[str, Any] | None = None

        # In-memory buffers (flushed on commit)
        self._buf_paths: list[str] = []
        self._buf_channels: list[list[str] | None] = []  # None = single-channel
        self._buf_meta: list[dict[str, Any]] = []
        self._buf_embeddings: list[np.ndarray] = []
        self._buf_props: list[dict[str, Any]] = []

        if os.path.isfile(path):
            if self.detect_format(path) != "hdf5":
                raise ValueError(f"Unrecognised checkpoint format for '{path}'. Expected HDF5.")
            self._open_existing()
        else:
            self._create_new(embedding_dim, processing_params=processing_params)

        if not self.lazy:
            self._load_into_ram()

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    @property
    def n_committed(self) -> int:
        """Rows committed for paths + embeddings + metadata."""
        if not self.lazy and self._ram_data is not None:
            return self._ram_data.n_images
        if self._file is None:
            return 0
        return int(self._file.attrs.get("n_committed", 0))

    @property
    def n_committed_props(self) -> int:
        """Rows committed for properties (may lag n_committed)."""
        if not self.lazy and self._ram_data is not None:
            # PhenoMeResults.properties is always padded to n_images.
            # We want the original n_committed_props.
            if self._file is not None:
                return int(self._file.attrs.get("n_committed_props", 0))
            # If closed and in RAM mode, we should have stored this.
            # For simplicity, we can look at the properties in ram_data.
            # PhenoMeResults doesn't explicitly store n_committed_props.
            # Let's add an attribute to CheckpointManager for this.
            return self._n_committed_props_ram
        if self._file is None:
            return 0
        return int(self._file.attrs.get("n_committed_props", 0))

    @property
    def embedding_dim(self) -> int | None:
        """Embedding width *D* from file attrs, or ``None`` when not set."""
        return self._embedding_dim

    @property
    def is_multichannel(self) -> bool:
        """Whether the checkpoint stores separate paths per channel."""
        if not self.lazy and self._ram_data is not None:
            return self._is_multichannel_ram
        if self._file is None:
            return False
        return bool(self._file.attrs.get("is_multichannel", False))

    # ------------------------------------------------------------------
    # Processing params (stored in /config group)
    # ------------------------------------------------------------------

    def get_processing_params(self) -> dict[str, Any] | None:
        """Return processing parameters from ``/config`` group, or None."""
        if not self.lazy and self._ram_data is not None:
            return self._processing_params_ram
        if self._file is None or "config" not in self._file:
            return None
        cfg = cast(h5py.Group, self._file["config"])
        params: dict[str, Any] = {}
        for key in self._PARAM_KEYS:
            if key == "channels":
                if "channels" in cfg:
                    params["channels"] = cast(h5py.Dataset, cfg["channels"])[:].tolist()
                else:
                    params["channels"] = None
            else:
                val = cfg.attrs.get(key, None)
                if val is not None:
                    if val == "none":
                        params[key] = None
                    elif isinstance(val, (np.bool_, bool)):
                        params[key] = bool(val)
                    elif isinstance(val, (np.integer,)):
                        params[key] = int(val)
                    else:
                        params[key] = val
        return params if params else None

    def set_processing_params(self, params: dict[str, Any]) -> None:
        """Write processing params to ``/config`` group (creates if needed)."""
        opened_here = False
        if self._file is None:
            self._ensure_open()
            opened_here = True

        f = self._file
        assert f is not None
        cfg = f.require_group("config")
        for key in self._PARAM_KEYS:
            if key in self._CONFIG_EXCLUDED_KEYS:
                continue
            val = params.get(key)
            if key == "channels":
                if "channels" in cfg:
                    del cfg["channels"]
                if val is not None:
                    cfg.create_dataset("channels", data=np.array(val, dtype=np.int32))
            else:
                cfg.attrs[key] = "none" if val is None else val
        f.flush()

        if not self.lazy:
            self._processing_params_ram = self.get_processing_params()

        if opened_here:
            f.close()
            self._file = None

    def validate_processing_params(
        self, current_params: dict[str, Any]
    ) -> list[tuple[str, Any, Any]]:
        """Compare *current_params* against stored params.

        Returns list of ``(key, stored_value, current_value)`` for mismatches.
        """
        stored = self.get_processing_params()
        if stored is None:
            return []
        mismatches = []
        for key in self._PARAM_KEYS:
            sv = stored.get(key)
            cv = current_params.get(key)

            # Normalize sequence types for safe comparison
            if isinstance(sv, (list, tuple, np.ndarray)):
                sv = list(sv)
            if isinstance(cv, (list, tuple, np.ndarray)):
                cv = list(cv)

            if sv is None and cv is None:
                continue

            try:
                is_diff = bool(sv != cv)
            except ValueError:
                # Fallback for ambiguous numpy array comparisons if normalization missed anything
                is_diff = not np.array_equal(sv, cv)

            if is_diff:
                mismatches.append((key, sv, cv))
        return mismatches

    # ------------------------------------------------------------------
    # Path / metadata queries
    # ------------------------------------------------------------------

    def get_committed_paths(self) -> set[str]:
        """Return the set of committed primary ``img_path`` values (resolved to absolute when possible)."""
        paths_list = self.get_committed_paths_list()
        return set(paths_list)

    def get_committed_paths_list(self) -> list[str]:
        """Return committed primary paths in order (resolved to absolute when storage root available)."""
        if not self.lazy and self._ram_data is not None:
            paths: list[str | list[str]] = list(self._ram_data.img_path)
            primary = [(p[0] if isinstance(p, list) else str(p)) for p in paths]
            return primary

        n = self.n_committed
        if self._file is None or n == 0 or "img_path" not in self._file:
            return []
        loaded_paths: list[str | list[str]] = cast(
            list[str | list[str]],
            decode_list(cast(h5py.Dataset, self._file["img_path"])[:n]),
        )
        root = self._storage_root or (
            self._file.attrs.get(_STORAGE_ROOT_ATTR) if self._file else None
        )
        if root:
            if isinstance(root, bytes):
                root = root.decode("utf-8")
            tmp_res = PhenoMeResults(img_path=list(loaded_paths))
            tmp_res.rebase_paths(str(root), old_data_dir="relative")
            loaded_paths = tmp_res.img_path
        primary = [(p[0] if isinstance(p, list) else str(p)) for p in loaded_paths]
        return primary

    def get_committed_metadata_list(self) -> list[dict[str, Any]]:
        """Return committed metadata dicts in order."""
        if not self.lazy and self._ram_data is not None:
            return self._ram_data.metadata
        n = self.n_committed
        if self._file is None or n == 0 or "metadata" not in self._file:
            return []
        return read_metadata_group(cast(h5py.Group, self._file["metadata"]), n)

    # ------------------------------------------------------------------
    # Property management
    # ------------------------------------------------------------------

    def clear_properties(self) -> None:
        """Truncate all property datasets to 0 and reset n_committed_props."""
        opened_here = False
        if self._file is None:
            self._ensure_open()
            opened_here = True

        f = self._file
        assert f is not None
        if "properties" in f:
            props_grp = cast(h5py.Group, f["properties"])
            for name in list(props_grp.keys()):
                cast(h5py.Dataset, props_grp[name]).resize(0, axis=0)
        f.attrs["n_committed_props"] = 0
        self._buf_props.clear()
        f.flush()

        if not self.lazy and self._ram_data is not None:
            self._ram_data.properties = [{} for _ in range(self._ram_data.n_images)]
            self._n_committed_props_ram = 0

        if opened_here:
            f.close()
            self._file = None

    # ------------------------------------------------------------------
    # Buffered embedding append
    # ------------------------------------------------------------------

    def _infer_storage_root(self, img_paths: list[Any]) -> None:
        """Infer common root from absolute paths for portable storage."""
        if self._storage_root is not None:
            return
        primaries = [
            p[0] if isinstance(p, list) else p
            for p in img_paths
            if (p[0] if isinstance(p, list) else p)
        ]
        abs_paths = [os.path.abspath(str(p)) for p in primaries if p]
        if len(abs_paths) > 1:
            with contextlib.suppress(ValueError, OSError):
                self._storage_root = os.path.commonpath([os.path.dirname(p) for p in abs_paths])
        elif len(abs_paths) == 1:
            self._storage_root = os.path.dirname(abs_paths[0])

    def buffer_embeddings(
        self,
        embeddings: np.ndarray,
        img_paths: list[Any],
        metadata_dicts: list[dict[str, Any]],
    ) -> None:
        """Add a batch of embeddings to the in-memory buffer.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(B, D)`` float32.
        img_paths : list
            Length B.  Each element is a ``str`` (single-channel) or a
            ``List[str]`` (multi-channel).
        metadata_dicts : list of dict
            Length B.
        """
        self._infer_storage_root(img_paths)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (batch, dim), got shape {embeddings.shape}")
        n = embeddings.shape[0]
        if len(img_paths) != n or len(metadata_dicts) != n:
            raise ValueError("embeddings, img_paths, metadata_dicts must have same length.")

        # In non-lazy mode, we might not have a file open.
        # We'll defer dataset creation to commit_embeddings.
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
        elif embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._embedding_dim}, got {embeddings.shape[1]}."
            )

        self._buf_embeddings.append(embeddings.astype(np.float32, copy=False))

        for p in img_paths:
            p_rel = self._relativize_path(p)
            if isinstance(p_rel, list):
                self._buf_paths.append(p_rel[0])  # primary path
                self._buf_channels.append(p_rel)
            else:
                self._buf_paths.append(str(p_rel))
                self._buf_channels.append(None)

        self._buf_meta.extend(metadata_dicts)

    def buffer_paths_and_metadata(
        self,
        img_paths: list[Any],
        metadata_dicts: list[dict[str, Any]],
    ) -> None:
        """Buffer paths and metadata without embeddings (property-only run)."""
        self._infer_storage_root(img_paths)
        n = len(img_paths)
        if len(metadata_dicts) != n:
            raise ValueError("img_paths and metadata_dicts must have same length.")
        for p in img_paths:
            p_rel = self._relativize_path(p)
            if isinstance(p_rel, list):
                self._buf_paths.append(p_rel[0])
                self._buf_channels.append(p_rel)
            else:
                self._buf_paths.append(str(p_rel))
                self._buf_channels.append(None)
        self._buf_meta.extend(metadata_dicts)

    def _relativize_path(self, path: Any) -> Any:
        """Convert path to relative if _storage_root is set. Normalizes to POSIX for portability."""
        root = self._storage_root
        if root is None:
            return path

        def _to_rel(p: Any) -> Any:
            if not p:
                return p
            abs_p = os.path.abspath(str(p))
            rel = os.path.relpath(abs_p, root)
            return _normalize_path_for_storage(rel)

        if isinstance(path, list):
            return [_to_rel(p) for p in path]
        return _to_rel(path)

    def commit_embeddings(self) -> int:
        """Flush embedding + path + metadata buffer to HDF5 atomically.

        Returns
        -------
        int
            Total n_committed after this commit.
        """
        if not self._buf_paths:
            return self.n_committed

        opened_here = False
        if self._file is None:
            self._ensure_open()
            opened_here = True

        f = self._file
        assert f is not None

        n_old = self.n_committed
        n_new = len(self._buf_paths)
        n_total = n_old + n_new

        # --- embeddings ---
        if self._buf_embeddings:
            all_emb = np.concatenate(self._buf_embeddings, axis=0)
            dim = all_emb.shape[1]
            if "embeddings" not in f:
                f.create_dataset(
                    "embeddings",
                    shape=(0, dim),
                    maxshape=(None, dim),
                    dtype=np.float32,
                    chunks=(_CHUNK_ROWS, dim),
                    compression="gzip",
                    compression_opts=_GZIP_LEVEL,
                )
                f.attrs["embedding_dim"] = dim
                self._embedding_dim = dim

            expected_dim = int(f.attrs.get("embedding_dim", dim))
            if dim != expected_dim:
                raise ValueError(f"Embedding dim mismatch: expected {expected_dim}, got {dim}.")
            ds = cast(h5py.Dataset, f["embeddings"])
            ds.resize(n_total, axis=0)
            ds[n_old:n_total] = all_emb

        # --- primary paths ---
        append_vlen_dataset(f, "img_path", self._buf_paths, n_old, n_total)

        # --- multi-channel paths ---
        has_multi = any(ch is not None for ch in self._buf_channels)
        if has_multi or self.is_multichannel:
            max_c = max(
                (len(ch) for ch in self._buf_channels if ch is not None),
                default=1,
            )
            # Existing file may already have a column count
            if "img_path_channels" in f:
                existing_c = cast(h5py.Dataset, f["img_path_channels"]).shape[1]
                max_c = max(max_c, existing_c)
            # Build (n_new, max_c) array of strings
            rows = []
            for ch in self._buf_channels:
                if ch is None:
                    rows.append([""] * max_c)
                else:
                    padded = list(ch) + [""] * (max_c - len(ch))
                    rows.append(padded)
            append_2d_vlen_dataset(f, "img_path_channels", rows, n_old, n_total, max_c)
            f.attrs["is_multichannel"] = True

        # --- metadata (columnar) ---
        append_metadata_group(f, "metadata", self._buf_meta, n_old, n_total)

        # --- atomic commit ---
        f.flush()
        f.attrs["n_committed"] = n_total
        if self._storage_root and _STORAGE_ROOT_ATTR not in f.attrs:
            f.attrs[_STORAGE_ROOT_ATTR] = self._storage_root
        f.flush()

        if not self.lazy:
            # Update RAM data before clearing buffers
            self._update_ram_from_buffers(embeddings=True)

        self._buf_embeddings.clear()
        self._buf_paths.clear()
        self._buf_channels.clear()
        self._buf_meta.clear()

        if opened_here:
            f.close()
            self._file = None

        return n_total

    @property
    def embeddings_buffered(self) -> int:
        """Number of paths/embeddings currently in the write buffer."""
        return len(self._buf_paths)

    # ------------------------------------------------------------------
    # Buffered property append
    # ------------------------------------------------------------------

    def buffer_properties(self, properties_dicts: list[dict[str, Any]]) -> None:
        """Add property dicts to the in-memory buffer."""
        self._buf_props.extend(p if isinstance(p, dict) else {} for p in properties_dicts)

    def commit_properties(self) -> int:
        """Flush property buffer to HDF5 atomically.

        Returns
        -------
        int
            Total n_committed_props after this commit.
        """
        if not self._buf_props:
            return self.n_committed_props

        opened_here = False
        if self._file is None:
            self._ensure_open()
            opened_here = True

        f = self._file
        assert f is not None

        n_old = self.n_committed_props
        n_new = len(self._buf_props)
        n_total = n_old + n_new

        append_properties_group(f, "properties", self._buf_props, n_old, n_total)

        f.flush()
        f.attrs["n_committed_props"] = n_total
        f.flush()

        if not self.lazy:
            self._update_ram_from_buffers(properties=True)

        self._buf_props.clear()

        if opened_here:
            f.close()
            self._file = None

        return n_total

    @property
    def properties_buffered(self) -> int:
        """Number of property dicts in the write buffer."""
        return len(self._buf_props)

    # ------------------------------------------------------------------
    # Lazy / indexed read-back
    # ------------------------------------------------------------------

    def load_embeddings_by_indices(self, indices: np.ndarray) -> np.ndarray:
        """Load only the specified rows of the embedding dataset.

        Sorts indices before reading for chunk efficiency, then restores
        original order.

        Parameters
        ----------
        indices : np.ndarray
            1-D integer array of row indices (0-based, within committed range).

        Returns
        -------
        np.ndarray
            Shape ``(len(indices), D)`` float32.
        """
        if not self.lazy and self._ram_data is not None:
            if self._ram_data.embeddings is None:
                raise RuntimeError("No embeddings loaded in memory.")
            return self._ram_data.embeddings[indices]

        if self._file is None:
            raise RuntimeError("Checkpoint file is not open.")
        if "embeddings" not in self._file:
            raise RuntimeError("No embeddings dataset in this checkpoint.")

        n_emb = self.n_committed
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            dim = int(self._file.attrs.get("embedding_dim", 0))
            return np.empty((0, dim), dtype=np.float32)

        if indices.max() >= n_emb:
            raise IndexError(f"Index {indices.max()} out of committed range [0, {n_emb - 1}].")

        sort_order = np.argsort(indices)
        sorted_idx = indices[sort_order]
        inv_order = np.empty_like(sort_order)
        inv_order[sort_order] = np.arange(len(sort_order))

        raw = np.array(cast(h5py.Dataset, self._file["embeddings"])[sorted_idx], dtype=np.float32)
        return raw[inv_order]

    def load_metadata_and_paths(self) -> tuple[list[Any], list[dict[str, Any]]]:
        """Load all committed paths and metadata dicts.

        Paths are resolved to absolute when _storage_root is available (from file or inference).

        Returns
        -------
        (paths, metadata_dicts)
            ``paths`` is a list of ``str`` (single-channel) or ``List[str]``
            (multi-channel).  ``metadata_dicts`` is a list of dicts.
        """
        if not self.lazy and self._ram_data is not None:
            return list(self._ram_data.img_path), list(self._ram_data.metadata)

        n = self.n_committed
        if self._file is None or n == 0:
            return [], []

        f = self._file
        loaded_paths: list[Any]
        if self.is_multichannel and "img_path_channels" in f:
            ds = cast(h5py.Dataset, f["img_path_channels"])
            raw = ds[:n]  # (n, C) array of strings/bytes
            loaded_paths = []
            for row in raw:
                decoded = [decode(v) for v in row]
                non_empty = [s for s in decoded if s]
                loaded_paths.append(
                    non_empty if len(non_empty) > 1 else (non_empty[0] if non_empty else "")
                )
        else:
            loaded_paths = cast(
                list[Any],
                decode_list(cast(h5py.Dataset, f["img_path"])[:n]) if "img_path" in f else [""] * n,
            )

        loaded_metadata: list[dict[str, Any]] = []
        if "metadata" in f:
            loaded_metadata = read_metadata_group(cast(h5py.Group, f["metadata"]), n)
        else:
            loaded_metadata = [{} for _ in range(n)]

        root = self._storage_root or f.attrs.get(_STORAGE_ROOT_ATTR)
        if root:
            if isinstance(root, bytes):
                root = root.decode("utf-8")
            tmp_res = PhenoMeResults(img_path=list(loaded_paths))
            tmp_res.rebase_paths(str(root), old_data_dir="relative")
            loaded_paths = cast(list[Any], tmp_res.img_path)

        return loaded_paths, loaded_metadata

    def load_properties_all(self) -> list[dict[str, Any]]:
        """Load all committed property dicts.

        Returns
        -------
        list of dict
            Length n_committed_props.  Empty list when no properties stored.
        """
        if not self.lazy and self._ram_data is not None:
            return self._ram_data.properties[: self._n_committed_props_ram]
        n = self.n_committed_props
        if self._file is None or n == 0 or "properties" not in self._file:
            return []
        return read_properties_group(cast(h5py.Group, self._file["properties"]), n)

    def load_committed_properties(self) -> list[dict[str, Any]]:
        """Load only the committed properties (avoids loading embeddings/paths).

        Used when merging partial property results.
        """
        return self.load_properties_all()

    # ------------------------------------------------------------------
    # Full read-back
    # ------------------------------------------------------------------

    def load_committed_results(self) -> PhenoMeResults:
        """Read all committed data and return a :class:`PhenoMeResults`.

        Paths are resolved to absolute when _storage_root is available.

        Returns
        -------
        PhenoMeResults
            Embeddings are loaded eagerly into the ``embeddings`` field.
            Returns an empty ``PhenoMeResults`` when no data is committed.
        """
        if not self.lazy and self._ram_data is not None:
            return self._ram_data

        if self._file is None:
            return PhenoMeResults()

        n = self.n_committed
        n_prop = self.n_committed_props
        if n == 0 and n_prop == 0:
            return PhenoMeResults()

        f = self._file
        paths, metadata = self.load_metadata_and_paths()

        embeddings: np.ndarray | None = None
        if "embeddings" in f and n > 0:
            embeddings = np.array(cast(h5py.Dataset, f["embeddings"])[:n], dtype=np.float32)

        properties: list[dict[str, Any]] = []
        if "properties" in f and n_prop > 0:
            properties = read_properties_group(cast(h5py.Group, f["properties"]), n_prop)
        # Pad to match n
        if len(properties) < n:
            properties.extend([{} for _ in range(n - len(properties))])

        return PhenoMeResults(
            img_path=paths,
            metadata=metadata,
            properties=properties,
            embeddings=embeddings,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_into_ram(self) -> None:
        """Load all committed data into RAM and close the HDF5 file."""
        if self._file is None:
            self._open_existing()
        assert self._file is not None

        # Read parameters before assigning self._ram_data to avoid bypassing I/O
        self._processing_params_ram = self.get_processing_params()

        self._ram_data = self.load_committed_results()
        self._n_committed_props_ram = int(self._file.attrs.get("n_committed_props", 0))
        self._is_multichannel_ram = bool(self._file.attrs.get("is_multichannel", False))
        self._file.close()
        self._file = None

    def _ensure_open(self, mode: str = "a") -> h5py.File:
        """Ensure the HDF5 file is open. Returns the file handle.

        If lazy=False, the caller is responsible for closing it if it was
        opened by this call.
        """
        if self._file is not None:
            return self._file
        try:
            self._file = h5py.File(self.path, mode)
            return self._file
        except (BlockingIOError, OSError) as e:
            if getattr(e, "errno", None) == 11 or isinstance(e, BlockingIOError):
                raise ConcurrentCheckpointAccessError(self.path, cause=e) from e
            raise

    def close(self) -> None:
        """Flush any remaining buffers and close the HDF5 file."""
        if self._buf_paths:
            self.commit_embeddings()
        if self._buf_props:
            self.commit_properties()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> CheckpointManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is None:
            self.close()
        else:
            self._buf_embeddings.clear()
            self._buf_paths.clear()
            self._buf_channels.clear()
            self._buf_meta.clear()
            self._buf_props.clear()
            if self._file is not None:
                self._file.close()
                self._file = None
        return False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(path: str) -> str:
        """Detect whether *path* is an HDF5 file.

        Returns ``'hdf5'`` or ``'unknown'``.
        """
        try:
            with open(path, "rb") as fh:
                magic = fh.read(4)
        except OSError:
            return "unknown"
        return "hdf5" if magic == b"\x89HDF" else "unknown"

    # ------------------------------------------------------------------
    # Static write / read for complete result sets
    # ------------------------------------------------------------------

    @staticmethod
    def write_results_to_hdf5(
        results: Any,  # PhenoMeResults or legacy dict
        path: str,
        compression: str = "gzip",
        compression_level: int = _GZIP_LEVEL,
        processing_params: dict[str, Any] | None = None,
    ) -> None:
        """Atomically write a full results object to an HDF5 file.

        Uses a temporary file + ``os.replace`` for crash safety.

        Parameters
        ----------
        results : PhenoMeResults or dict
            Source data.  Accepted dict keys: ``'embeddings'``, ``'img_path'``,
            ``'metadata'``, ``'properties'``.
        path : str
            Output file path.
        compression, compression_level
            Passed to h5py for the embeddings dataset.
        processing_params : dict or None
            Written to the ``/config`` group when provided.
        """
        # Normalise: accept both PhenoMeResults and legacy dict
        if hasattr(results, "img_path"):
            img_paths = results.img_path
            metadata = results.metadata
            props = results.properties
            embeddings = results.embeddings
        else:
            img_paths = results.get("img_path", [])
            metadata = results.get("metadata", [])
            props = results.get("properties", [])
            embeddings = results.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == 0:
                embeddings = None

        if len(img_paths) == 0:
            raise ValueError("Cannot save results without image paths.")

        n = len(img_paths)

        if isinstance(embeddings, list) and embeddings:
            embeddings = np.stack(embeddings)

        # Relativize paths for storage (security and portability)
        def _primary(p: Any) -> Any:
            return p[0] if isinstance(p, list) else p

        primaries = [_primary(p) for p in img_paths if _primary(p)]
        storage_root: str | None = None
        all_absolute = primaries and all(os.path.isabs(str(p)) for p in primaries)
        if all_absolute:
            try:
                storage_root = os.path.commonpath(
                    [os.path.dirname(os.path.abspath(str(p))) for p in primaries]
                )

                def _rel_infer(p: Any) -> Any:
                    if isinstance(p, list):
                        return [
                            _normalize_path_for_storage(
                                os.path.relpath(os.path.abspath(str(ch)), storage_root)
                            )
                            for ch in p
                        ]
                    return _normalize_path_for_storage(
                        os.path.relpath(os.path.abspath(str(p)), storage_root)
                    )

                img_paths = [_rel_infer(p) for p in img_paths]
            except (ValueError, OSError) as e:
                raise ValueError("Cannot infer a common data root from image paths.") from e
        else:
            # Paths already relative; normalize separators
            def _norm_only(p: Any) -> Any:
                if isinstance(p, list):
                    return [_normalize_path_for_storage(str(ch)) for ch in p]
                return _normalize_path_for_storage(str(p))

            img_paths = [_norm_only(p) for p in img_paths]

        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".h5.tmp", prefix=".ckpt_")
        os.close(fd)

        try:
            with h5py.File(tmp_path, "w") as f:
                # Embeddings
                if isinstance(embeddings, np.ndarray) and len(embeddings) > 0:
                    dim = embeddings.shape[1]
                    f.create_dataset(
                        "embeddings",
                        data=embeddings.astype(np.float32),
                        maxshape=(None, dim),
                        chunks=(_CHUNK_ROWS, dim),
                        compression=compression,
                        compression_opts=compression_level,
                    )
                    f.attrs["embedding_dim"] = dim
                else:
                    f.attrs["embedding_dim"] = 0

                # Primary paths
                primary_paths = [p[0] if isinstance(p, list) else str(p) for p in img_paths]
                f.create_dataset(
                    "img_path",
                    data=primary_paths,
                    dtype=_VL_STR,
                    maxshape=(None,),
                )

                # Multi-channel paths
                is_multi = any(isinstance(p, list) for p in img_paths)
                if is_multi:
                    max_c = max(len(p) if isinstance(p, list) else 1 for p in img_paths)
                    rows = []
                    for p in img_paths:
                        if isinstance(p, list):
                            rows.append(list(p) + [""] * (max_c - len(p)))
                        else:
                            rows.append([str(p)] + [""] * (max_c - 1))
                    write_2d_vlen(f, "img_path_channels", rows)
                f.attrs["is_multichannel"] = is_multi

                # Metadata (columnar)
                if metadata:
                    write_metadata_group(f, "metadata", metadata, n)

                # Properties (columnar)
                if props:
                    write_properties_group(f, "properties", props, n)
                    f.attrs["n_committed_props"] = n
                else:
                    f.attrs["n_committed_props"] = 0

                # Config (always create for portability)
                cfg = f.create_group("config")
                if processing_params:
                    for key in CheckpointManager._PARAM_KEYS:
                        if key in CheckpointManager._CONFIG_EXCLUDED_KEYS:
                            continue
                        val = processing_params.get(key)
                        if key == "channels":
                            if val is not None:
                                cfg.create_dataset("channels", data=np.array(val, dtype=np.int32))
                        else:
                            cfg.attrs[key] = "none" if val is None else val

                # Root attributes
                f.attrs["version"] = CheckpointManager.FORMAT_VERSION
                f.attrs["n_committed"] = n
                if storage_root:
                    f.attrs[_STORAGE_ROOT_ATTR] = storage_root
                f.flush()

            os.replace(tmp_path, path)

        except BaseException:
            if os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            raise

    @staticmethod
    def load_hdf5_results(path: str) -> PhenoMeResults:
        """Load and validate an HDF5 results/checkpoint file.

        Paths are resolved to absolute when _storage_root is stored in the file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        PhenoMeResults

        Raises
        ------
        ValueError
            If the file format is invalid or corrupted.
        """
        with h5py.File(path, "r") as f:
            version = str(f.attrs.get("version", "0.0"))
            if not _is_checkpoint_version_supported(version):
                raise ValueError(
                    f"Checkpoint version '{version}' is not supported "
                    f"(supported: {sorted(CHECKPOINT_SUPPORTED_VERSIONS)}). "
                    "Re-run processing to create a fresh checkpoint."
                )

            n = int(f.attrs.get("n_committed", 0))
            n_prop = int(f.attrs.get("n_committed_props", 0))
            is_multi = bool(f.attrs.get("is_multichannel", False))

            if n == 0 and n_prop == 0:
                return PhenoMeResults()

            # Paths
            paths: list[Any]
            if is_multi and "img_path_channels" in f:
                ds = cast(h5py.Dataset, f["img_path_channels"])
                actual_n = min(n, ds.shape[0])
                raw = ds[:actual_n]
                paths = []
                for row in raw:
                    decoded = [decode(v) for v in row]
                    non_empty = [s for s in decoded if s]
                    paths.append(
                        non_empty if len(non_empty) > 1 else (non_empty[0] if non_empty else "")
                    )
            elif "img_path" in f:
                ds = cast(h5py.Dataset, f["img_path"])
                actual_n = min(n, ds.shape[0])
                paths = decode_list(ds[:actual_n])
            else:
                paths = []

            n = len(paths)

            storage_root = f.attrs.get(_STORAGE_ROOT_ATTR)
            if storage_root:
                if isinstance(storage_root, bytes):
                    storage_root = storage_root.decode("utf-8")
                tmp_res = PhenoMeResults(img_path=list(paths))
                tmp_res.rebase_paths(str(storage_root), old_data_dir="relative")
                paths = tmp_res.img_path

            # Metadata
            metadata: list[dict[str, Any]] = []
            if "metadata" in f:
                metadata = read_metadata_group(cast(h5py.Group, f["metadata"]), n)
            else:
                metadata = [{} for _ in range(n)]

            # Embeddings
            embeddings: np.ndarray | None = None
            if "embeddings" in f and n > 0:
                ds = cast(h5py.Dataset, f["embeddings"])
                actual_emb = ds.shape[0]
                n_to_load = min(n, actual_emb)
                embeddings = np.array(ds[:n_to_load], dtype=np.float32)

                stored_dim = int(f.attrs.get("embedding_dim", 0))
                if stored_dim > 0 and ds.shape[1] != stored_dim:
                    raise ValueError("embedding_dim attribute does not match dataset shape.")

            # Properties
            properties: list[dict[str, Any]] = []
            if "properties" in f and n_prop > 0:
                actual_prop = min(n_prop, n)
                properties = read_properties_group(cast(h5py.Group, f["properties"]), actual_prop)
            if len(properties) < n:
                properties.extend([{} for _ in range(n - len(properties))])

            return PhenoMeResults(
                img_path=paths,
                metadata=metadata,
                properties=properties,
                embeddings=embeddings,
            )

    # ------------------------------------------------------------------
    # Private — file creation / opening
    # ------------------------------------------------------------------

    def _create_new(
        self,
        embedding_dim: int | None,
        processing_params: dict[str, Any] | None = None,
    ) -> None:
        """Create a new HDF5 checkpoint file on disk (empty datasets)."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._file = h5py.File(self.path, "w")

        if embedding_dim is not None:
            self._file.create_dataset(
                "embeddings",
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
                chunks=(_CHUNK_ROWS, embedding_dim),
                compression="gzip",
                compression_opts=_GZIP_LEVEL,
            )
            self._file.attrs["embedding_dim"] = embedding_dim
        else:
            self._file.attrs["embedding_dim"] = 0

        self._file.create_dataset("img_path", shape=(0,), maxshape=(None,), dtype=_VL_STR)

        self._file.attrs["version"] = self.FORMAT_VERSION
        self._file.attrs["n_committed"] = 0
        self._file.attrs["n_committed_props"] = 0
        self._file.attrs["is_multichannel"] = False

        self._file.require_group("config")
        if processing_params is not None:
            self.set_processing_params(processing_params)

        self._file.flush()

    def _open_existing(self) -> None:
        """Open an existing checkpoint for append access and validate layout."""
        try:
            self._file = h5py.File(self.path, "a")
        except (BlockingIOError, OSError) as e:
            if getattr(e, "errno", None) == 11 or isinstance(e, BlockingIOError):
                raise ConcurrentCheckpointAccessError(self.path, cause=e) from e
            raise
        self._validate_and_truncate()
        dim = self._file.attrs.get("embedding_dim", None)
        self._embedding_dim = int(dim) if dim is not None and int(dim) > 0 else None
        stored_root = self._file.attrs.get(_STORAGE_ROOT_ATTR)
        if stored_root is not None:
            self._storage_root = (
                stored_root.decode("utf-8") if isinstance(stored_root, bytes) else str(stored_root)
            )

    def _validate_and_truncate(self) -> None:
        """Truncate datasets to committed row counts and validate version."""
        f = self._file
        assert f is not None
        version = str(f.attrs.get("version", "0.0"))
        if not _is_checkpoint_version_supported(version):
            raise ValueError(
                f"Checkpoint version '{version}' is not supported "
                f"(supported: {sorted(CHECKPOINT_SUPPORTED_VERSIONS)}). "
                "Re-run processing to create a fresh checkpoint."
            )
        n = int(f.attrs.get("n_committed", 0))
        n_prop = int(f.attrs.get("n_committed_props", 0))

        # Truncate datasets to committed lengths
        for ds_name in ("img_path",):
            if ds_name in f:
                ds = cast(h5py.Dataset, f[ds_name])
                if ds.shape[0] > n:
                    ds.resize(n, axis=0)

        if "img_path_channels" in f:
            ds = cast(h5py.Dataset, f["img_path_channels"])
            if ds.shape[0] > n:
                ds.resize(n, axis=0)

        if "embeddings" in f:
            ds = cast(h5py.Dataset, f["embeddings"])
            if ds.shape[0] > n:
                ds.resize(n, axis=0)

        if "metadata" in f:
            truncate_group_datasets(cast(h5py.Group, f["metadata"]), n)

        if "properties" in f:
            truncate_group_datasets(cast(h5py.Group, f["properties"]), n_prop)

        f.flush()

    def _update_ram_from_buffers(self, embeddings: bool = False, properties: bool = False) -> None:
        """Merge current buffers into self._ram_data."""
        if self._ram_data is None:
            self._ram_data = PhenoMeResults()

        if embeddings:
            # Merge paths, metadata and embeddings
            new_paths: list[Any] = []
            for i, p in enumerate(self._buf_paths):
                ch = self._buf_channels[i]
                if ch is not None:
                    # Multi-channel paths are stored as List[str] in buf_channels
                    new_paths.append(ch)
                else:
                    new_paths.append(p)

            # Rebase paths if storage root was inferred
            if self._storage_root:
                tmp_res = PhenoMeResults(img_path=new_paths)
                tmp_res.rebase_paths(self._storage_root, old_data_dir="relative")
                new_paths = tmp_res.img_path

            self._ram_data.img_path.extend(new_paths)
            self._ram_data.metadata.extend(self._buf_meta)

            if self._buf_embeddings:
                # Concatenate all buffered batches
                all_new_emb = np.concatenate(self._buf_embeddings, axis=0)
                if self._ram_data.embeddings is None:
                    self._ram_data.embeddings = all_new_emb
                else:
                    self._ram_data.embeddings = np.concatenate(
                        [self._ram_data.embeddings, all_new_emb], axis=0
                    )

            # Pad properties to match n_images
            n_img = len(self._ram_data.img_path)
            if len(self._ram_data.properties) < n_img:
                self._ram_data.properties.extend(
                    [{} for _ in range(n_img - len(self._ram_data.properties))]
                )

        if properties:
            # Properties are appended from current n_committed_props
            start = self._n_committed_props_ram
            end = start + len(self._buf_props)
            # Ensure ram_data has enough space (should already be padded by embeddings commit)
            if len(self._ram_data.properties) < end:
                self._ram_data.properties.extend(
                    [{} for _ in range(end - len(self._ram_data.properties))]
                )
            self._ram_data.properties[start:end] = self._buf_props
            self._n_committed_props_ram = end
