"""Main phenotyping analysis pipeline."""

import contextlib
import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ._logging import get_logger
from .core import (
    PhenoMeResults,
    build_export_dataframe,
    get_all_metadata_keys,
    metadata_to_stable_key,
    validate_results,
)
from .core.dataframe_contract import IMAGE_INDEX
from .io import CheckpointManager, FileDiscovery
from .io._path_utils import _resolve_results_hdf5_path
from .io.checkpoint_alignment import (
    _build_unique_path_lookup,
    _lookup_unique_path_index,
    build_metadata_key_index,
    filter_items_not_in_checkpoint,
    get_already_committed_metadata_keys,
    get_already_committed_paths,
    merge_metadata_from_current_run,
    rebase_paths_from_current_run,
)
from .metadata.base import MetadataBase
from .mixins import (
    PhenoMeAnalysis,
    PhenoMeBatchCorrection,
    PhenoMeDistances,
    PhenoMeProperties,
    PhenoMeVisualization,
)
from .mixins.dataset import PhenoMeDataset, collate_fn
from .mixins.embedding_extractor import EmbeddingExtractor
from .utils import TransformBuilder
from .utils.device import get_default_device, set_default_device, set_determinism
from .utils.model_wrapper import ModelWrapper
from .utils.path_utils import path_repr as _path_repr

logger = get_logger(__name__)


def _effective_num_workers(num_workers: int) -> int:
    """Use num_workers=0 in Jupyter to avoid multiprocessing QueueFeederThread crashes."""
    if num_workers <= 0:
        return num_workers
    import sys

    if "ipykernel" in sys.modules:
        logger.debug("Jupyter detected: using num_workers=0 to avoid multiprocessing issues.")
        return 0
    return num_workers


def _normalize_row_indices(
    indices: int | list[int] | np.ndarray | None,
) -> np.ndarray | None:
    """Convert a row index selector to a 1-D int64 array, or None for all rows."""
    if indices is None:
        return None
    if isinstance(indices, (int, np.integer)):
        return np.array([int(indices)], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64).ravel()


class PhenoMe(
    PhenoMeProperties,
    PhenoMeAnalysis,
    PhenoMeDistances,
    PhenoMeVisualization,
    PhenoMeBatchCorrection,
):
    """@section Overview
    @order 0

    Main class for phenotyping analysis using deep learning embeddings.

    Provides: compute_properties, property_stats_by_group; compute_clustering,
    detect_outliers, find_prototypes; compute_reference_distances; correct_batches;
    plot_pca, plot_tsne, plot_umap, and related methods.

    **Embedding lifecycle:**

    - **Eager**: When no checkpoint is used, embeddings live in ``results.embeddings`` (np.ndarray).
    - **Lazy**: When a checkpoint is active (``self._db``), ``results.embeddings`` is ``None``;
      embeddings are read on demand via ``get_embeddings()`` from the HDF5 file.
    - **Temporal**: Rows from ``process_temporal_images()`` use ``self._temporal_embeddings``
      while ``_db`` is open; ``get_embeddings()`` merges them with checkpoint rows by index.
      After ``checkpoint_context()`` exits with temporal data present, the merged array is
      copied to ``results.embeddings`` and the temporal buffer is cleared.

    Args:
        device: Optional torch.device for GPU-accelerated analysis operations.
            Recommended for faster computation. If None, operations run on CPU.
        seed: Optional random seed for reproducibility (default: None).
            When set, used for all random operations (determinism, clustering,
            t-SNE, UMAP, mutual information, random sampling in reports).
            If None, no seeds are set anywhere.
        use_gpu_for_dr: If True and device is CUDA, uses TorchDR for GPU-accelerated
            dimensionality reduction (PCA, t-SNE, UMAP). If False (default), always
            use sklearn/umap-learn on CPU.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        seed: int | None = None,
        use_gpu_for_dr: bool = False,
    ):
        if device is not None:
            set_default_device(device)
        self.device = get_default_device()

        # Normalisation parameters (ImageNet)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Typed results container — embeddings field is None when absent or lazy.
        self.results: PhenoMeResults = PhenoMeResults()

        # In-progress embedding buffer used during extraction.
        # Embeddings accumulate here and are moved to results.embeddings on finalisation.
        # This buffer never lives inside PhenoMeResults.
        self._emb_buffer: list[np.ndarray] = []
        self._metadata_config: MetadataBase | None = None

        self.preprocessing_fn: Callable[[np.ndarray], np.ndarray] | None = None
        self._processing_params: dict[str, Any] | None = None

        # PyTorch transforms on images before the model (resize, pad, normalize, custom Compose, etc.);
        # shared by the embedding dataloader and visualization (e.g. show_image).
        self.image_transforms: Any | None = None

        # When a checkpoint path is active, this holds the open CheckpointManager.
        # results.embeddings is None (lazy sentinel) and embeddings are loaded on demand.
        self._db: CheckpointManager | None = None
        self._db_indices: np.ndarray | None = None
        # Non-lazy: per-row internal dicts (aligned with results.properties), not in PhenoMeResults
        self._ram_internal: list[dict[str, Any]] | None = None

        # Temporal (in-memory only) embeddings appended via process_temporal_images().
        # When _db is open, temporal rows live here; results indices >= _temporal_start_idx.
        self._temporal_embeddings: np.ndarray | None = None
        self._temporal_start_idx: int = 0

        # DataFrame describing the dataset used for processing (from find_files).
        # Stores image and mask filename columns for downstream internal use.
        self._file_df: pd.DataFrame | None = None
        self._image_path_col: str | None = None
        self._channel_path_cols: list[str] | None = None
        self._mask_path_col: str | None = None
        # Optional semantic labels for image channels (order = channel axis / file_path list).
        self._channel_names: list[str] | None = None

        self.seed: int | None = seed
        self.use_gpu_for_dr: bool = use_gpu_for_dr
        if seed is not None:
            set_determinism(seed)

        self._file_discovery = FileDiscovery()
        self._transform_builder = TransformBuilder(mean=self.mean, std=self.std)

        self._batch_correction_applied: bool = False
        self._batch_correction_last_info: dict[str, Any] | None = None

    def __repr__(self) -> str:
        n_images = self.results.n_images
        emb_dim = self.results.embedding_dim
        n_props = len(self.results.property_keys)
        mode = "lazy" if self._db is not None else "eager"
        parts = [
            f"n_images={n_images}",
            f"embedding_dim={emb_dim}",
            f"properties={n_props}",
            f"mode='{mode}'",
            f"device={self.device}",
        ]
        return f"PhenoMe({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, verbose: bool = False, clear_file_df: bool = False) -> None:
        """Reset all stored data to a clean state.

        Closes any open HDF5 database handle before clearing results.
        By default preserves _file_df so compute_properties and inspect_data
        continue to work after process_images. Set clear_file_df=True for a
        full reset (e.g. when switching to a completely new dataset).
        """
        if self._db is not None:
            try:
                self._db.close()
            except OSError as e:
                logger.debug("Error closing checkpoint on reset: %s", e)
            self._db = None
        self._db_indices = None
        self._ram_internal = None
        self._emb_buffer = []
        self._temporal_embeddings = None
        self._temporal_start_idx = 0
        self.results = PhenoMeResults()
        self.reset_properties(verbose=False)
        self.preprocessing_fn = None
        self._processing_params = None
        self.image_transforms = None
        self._metadata_config = None
        if clear_file_df:
            self._file_df = None
            self._image_path_col = None
            self._channel_path_cols = None
            self._mask_path_col = None
            self._channel_names = None
        self._batch_correction_applied = False
        self._batch_correction_last_info = None

        if verbose:
            logger.info("Reset: All stored data cleared.")

    def __del__(self) -> None:
        """Close HDF5 database handle on garbage collection.

        Robust to partial initialization: if ``__init__`` failed before
        assigning ``_db`` (or the Python runtime is tearing down), the
        ``getattr`` default avoids ``AttributeError`` noise at shutdown.
        """
        db = getattr(self, "_db", None)
        if db is not None:
            # ``logger`` may itself be torn down during interpreter exit, so we
            # suppress any follow-up errors to keep __del__ from raising.
            with contextlib.suppress(Exception):
                try:
                    db.close()
                except OSError as e:
                    logger.debug("Error closing checkpoint in __del__: %s", e)
            self._db = None

    # ------------------------------------------------------------------
    # Embedding access (lazy or eager)
    # ------------------------------------------------------------------

    @property
    def has_embeddings(self) -> bool:
        """Return True if embedding data is available (lazy or eager).

        A properties-only checkpoint has ``n_committed > 0`` but no embedding
        dataset; we additionally require ``embedding_dim`` to be a positive int.
        """
        if self._db is not None:
            db_dim = self._db.embedding_dim or 0
            if db_dim > 0 and self._db.n_committed > 0:
                return True
            return bool(
                self._temporal_embeddings is not None and len(self._temporal_embeddings) > 0
            )
        return self.results.has_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality, or 0 if unavailable."""
        if self._db is not None:
            dim = self._db.embedding_dim or 0
            if dim > 0:
                return dim
            if self._temporal_embeddings is not None and self._temporal_embeddings.ndim == 2:
                return int(self._temporal_embeddings.shape[1])
            return 0
        return self.results.embedding_dim

    def get_embeddings(
        self,
        indices: int | list[int] | np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return embeddings for the given row indices.

        When a checkpoint / HDF5 database is active (``self._db`` is not None),
        only the requested rows are read from disk — the core of the lazy-loading
        strategy.  When temporal embeddings exist (from ``process_temporal_images``),
        rows with index ``>= self._temporal_start_idx`` are read from
        ``self._temporal_embeddings``; earlier rows come from the checkpoint.

        When no database is active, returns ``results.embeddings`` (optionally sliced).
        If that array holds only checkpoint rows (length ``_temporal_start_idx``) and
        temporal embeddings are still in memory, they are appended to form a full
        ``(n_images, D)`` array. If ``results.embeddings`` already has ``n_images`` rows
        (e.g. materialized on ``checkpoint_context`` exit), temporal data is not appended
        again.

        Args:
            indices: Optional row index or sequence of row indices (0-based).
                A single ``int`` returns shape ``(1, D)``. If None, returns all
                embeddings.

        Returns:
            np.ndarray shape ``(len(indices), D)`` float32, or None if no
            embeddings are available.
        """
        indices_arr = _normalize_row_indices(indices)
        n_total = self.results.n_images
        temporal_emb = self._temporal_embeddings
        has_temporal = (
            temporal_emb is not None and temporal_emb.ndim == 2 and temporal_emb.shape[0] > 0
        )
        n_temporal = int(temporal_emb.shape[0]) if temporal_emb is not None and has_temporal else 0

        if self._db is not None:
            db = self._db
            db_indices = self._db_indices
            n_ckpt = len(db_indices) if db_indices is not None else 0
            if n_ckpt == 0 and n_temporal == 0:
                return None

            def _fetch(indices_to_use: np.ndarray) -> np.ndarray:
                # indices_to_use: (K,) int64; out: (K, D) float32
                dim = self.embedding_dim
                if dim == 0:
                    return np.array([], dtype=np.float32).reshape(0, 0)
                out = np.empty((len(indices_to_use), dim), dtype=np.float32)
                ckpt_mask = indices_to_use < self._temporal_start_idx
                temp_mask = ~ckpt_mask
                if np.any(ckpt_mask) and db_indices is not None and db is not None:
                    ckpt_pos = np.where(ckpt_mask)[0]
                    ckpt_indices = indices_to_use[ckpt_mask]
                    h5_idx = db_indices[np.asarray(ckpt_indices, dtype=np.int64)]
                    out[ckpt_pos] = db.load_embeddings_by_indices(h5_idx)
                if np.any(temp_mask) and has_temporal and temporal_emb is not None:
                    temp_pos = np.where(temp_mask)[0]
                    local_temp = (
                        np.asarray(indices_to_use[temp_mask], dtype=np.int64)
                        - self._temporal_start_idx
                    )
                    out[temp_pos] = temporal_emb[local_temp]
                return out

            if indices_arr is None:
                indices_arr = np.arange(n_total, dtype=np.int64)

            if len(indices_arr) == 0:
                return None

            out = _fetch(indices_arr)
            if out.size == 0:
                return None
            return out

        emb = self.results.embeddings
        temporal_emb = self._temporal_embeddings
        has_temporal = (
            temporal_emb is not None and temporal_emb.ndim == 2 and temporal_emb.shape[0] > 0
        )
        if emb is None or len(emb) == 0:
            if not has_temporal:
                return None
            emb = temporal_emb
        elif has_temporal and temporal_emb is not None:
            # Merge only when results hold checkpoint rows; skip if already materialized
            # (e.g. after checkpoint_context exit wrote the full merged array).
            n_ckpt_rows = int(self._temporal_start_idx)
            if len(emb) == n_ckpt_rows and n_ckpt_rows + len(temporal_emb) == n_total:
                emb = np.concatenate([emb, temporal_emb], axis=0)
        if indices_arr is None:
            return emb
        return emb[indices_arr]

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def set_file_df(
        self,
        file_df: pd.DataFrame,
        channel_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Set the internal file DataFrame used for processing.

        Validates that file_df has a 'file_path' column and optionally 'mask_path'.
        Stores a normalized copy for use by process_images, compute_properties,
        inspect_data, and related methods.

        Args:
            file_df: DataFrame with 'file_path' column (str or list of str per row).
                May include 'mask_path' and metadata columns.
            channel_names: Optional labels for image channels, in channel-axis order
                (same order as list entries in ``file_path`` for multi-file samples).
                When set, multi-channel property columns use these labels as suffixes
                (e.g. ``intensity_entropy_DAPI`` instead of ``intensity_entropy_ch0``).
                Default: None (integer ``_ch{idx}`` suffixes).

        Returns:
            The normalized DataFrame that was stored.
        """
        self._validate_process_images_inputs(file_df)
        self._file_df = file_df.reset_index(drop=True).copy()
        self._init_file_df_metadata(self._file_df)
        self._channel_names = self._normalize_channel_names(channel_names)
        return self._file_df

    def find_files(
        self,
        image_dir: str,
        mask_dir: str | None = None,
        extensions: list[str] | None = None,
        metadata_fn: Callable[[str], dict[str, Any]] | MetadataBase | None = None,
        on_missing_metadata: str = "drop",
        mask_filename_column: str | None = None,
        mask_extensions: list[str] | None = None,
        channel_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Discover image and mask files from directories, extract metadata, and cache internally.

        Recursively scans `image_dir` for image files and optionally aligns them with masks
        from `mask_dir`. Extracts per-file metadata using `metadata_fn` and stores the result
        as the internal file DataFrame used by `process_images()`, `compute_properties()`,
        and `inspect_data()`.

        Args:
            image_dir (str): Root directory or glob pattern for image files. Scanned recursively.
                Example: "path/to/images/" or "path/to/**/*.tif".
            mask_dir (str or None): Root directory for corresponding mask files. If provided,
                adds a 'mask_path' column to the output DataFrame. Default: None (no masks).
            extensions (list of str or None): File extensions to include (e.g., [".tif", ".png", ".jpg"]).
                If None, defaults to common image formats. Default: None.
            metadata_fn (callable, MetadataBase, or None): Function or object to extract metadata
                from file paths. Signature: `fn(file_path: str) -> dict[str, Any]`.
                Examples:
                - `DefaultMetadata()`: Simple metadata.
                - `PathTemplateMetadata(template="...")`: Extract from path pattern.
                - `lambda p: {'batch': p.split('/')[-2]}`: Custom function.
                If None, no metadata extracted. Default: None.
            on_missing_metadata (str): How to handle files with missing metadata. One of:
                - 'drop': Remove rows with incomplete metadata.
                - 'keep': Include rows with NaN metadata.
                Default: 'drop'.
            mask_filename_column (str or None): Metadata column name used to locate mask files.
                If set, masks are located using this column value. Useful for custom mask naming.
                Default: None (standard alignment logic).
            mask_extensions (list of str or None): Extensions to try when exact mask filename
                fails (e.g., [".png", ".tif"]). Default: None (exact filename required).
            channel_names (list of str or None): Optional labels for image channels, in
                channel-axis order (same order as list entries in ``file_path`` for
                multi-file samples). When set, multi-channel property columns use these
                labels as suffixes (e.g. ``intensity_entropy_DAPI`` instead of
                ``intensity_entropy_ch0``). Names must be non-empty, unique
                (case-insensitive), and contain no whitespace. Default: None
                (integer ``_ch{idx}`` suffixes).

        Returns:
            pd.DataFrame: File discovery DataFrame with columns:
                - 'file_path': (str or list of str) Image file path(s).
                - 'mask_path': (str, if mask_dir set) Corresponding mask path.
                - Metadata columns: Custom columns from `metadata_fn` if provided.
                Shape: (N, 2+M) where N is number of images, M is metadata columns.

        Raises:
            FileNotFoundError: If `image_dir` does not exist.
            ValueError: If all files are dropped due to `on_missing_metadata='drop'`,
                or if `channel_names` is invalid.
            RuntimeError: If metadata extraction fails or masks cannot be resolved.

        Example:
            Discover images with metadata:

            >>> pm = PhenoMe()
            >>> df = pm.find_files(
            ...     "data/images/",
            ...     metadata_fn=PathTemplateMetadata(template="batch_{batch}/sample_{id}"),
            ...     extensions=[".tif", ".png"]
            ... )
            >>> print(df.shape)
            (150, 4)  # 150 images, 4 columns: file_path, batch, id, [others]

            With masks:

            >>> df = pm.find_files(
            ...     image_dir="data/images/",
            ...     mask_dir="data/masks/",
            ...     extensions=[".tif"]
            ... )
            >>> print(df.columns)
            Index(['file_path', 'mask_path'], dtype='object')

            Named channels for property suffixes:

            >>> df = pm.find_files(
            ...     "data/images/",
            ...     channel_names=["DAPI", "GFP"],
            ... )

        See Also:
            `set_file_df`: Manually provide a file DataFrame.
            `process_images`: Process discovered images.
        """
        if isinstance(metadata_fn, MetadataBase):
            self._metadata_config = metadata_fn
        else:
            self._metadata_config = None
        df = self._file_discovery._find_files_from_dir(
            data_dir=image_dir,
            extensions=extensions,
            metadata_fn=metadata_fn,
            on_missing_metadata=on_missing_metadata,
            mask_dir=mask_dir,
            mask_filename_column=mask_filename_column,
            mask_extensions=mask_extensions,
        )
        if isinstance(df, pd.DataFrame) and not df.empty and "file_path" in df.columns:
            self._file_df = df.reset_index(drop=True).copy()
            self._init_file_df_metadata(self._file_df)
            self._channel_names = self._normalize_channel_names(channel_names)
        elif isinstance(df, pd.DataFrame) and df.empty:
            logger.warning("find_files returned an empty DataFrame; clearing previous _file_df.")
            self._file_df = None
            self._channel_names = None
        return df

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_images(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int = 32,
        num_workers: int = 4,
        filters: dict[str, list[Any]] | None = None,
        exclude: dict[str, list[Any]] | None = None,
        channel_mode: Literal["split", "combined"] = "split",
        channels: list[int] | None = None,
        preprocessing_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        custom_transformations: Any | None = None,
        append: bool = False,
        resize_size: int | None = 224,
        pad_size: int | None = None,
        checkpoint_path: str | None = None,
        force_rgb: bool = True,
        l2_normalize_channels: bool = True,
        save_every: int = 5,
        lazy_checkpoint: bool = True,
        force_reprocess: bool = False,
    ) -> None:
        """Extract embeddings from images using a pretrained or custom model.

        Reads images from the file DataFrame (set via `find_files()` or `set_file_df()`),
        processes them through the model in batches, and stores embeddings and metadata
        in `self.results`. Supports filtering, channel selection, custom preprocessing,
        and checkpointing for resumable processing.

        **Embedding storage:**
        - Eager (default): Embeddings stored in `results.embeddings` (ndarray).
        - Lazy (with checkpoint): Embeddings stored in HDF5; `results.embeddings` is None
          and embeddings are loaded on-demand via `get_embeddings()`.
        - Temporal (incremental): New images added to existing checkpoint via
          `process_temporal_images()`.

        Args:
            model_wrapper (ModelWrapper): Model instance (e.g., from `load_dinov2_model()`).
                Must implement `_get_embeddings(tensor: torch.Tensor) -> torch.Tensor`.
                For non-PyTorch models (Keras, TensorFlow, etc.), see the
                [Custom Model Wrapper guide](https://AAitorG.github.io/PhenoMe/guides/custom-model-wrapper/).
            batch_size (int): Batch size for processing. Default: 32. Larger batches are
                faster but use more GPU memory.
            num_workers (int): Number of loader processes. In Jupyter, automatically
                reduced to 0 to avoid multiprocessing issues. Default: 4.
            filters (dict or None): Include only rows matching criteria.
                Format: `{column: [value1, value2, ...]}`. Default: None (no filtering).
            exclude (dict or None): Exclude rows matching *any* criterion (OR across
                fields). Same value format as `filters`. Default: None.
            channel_mode (str): How to handle multi-channel images. One of:
                - 'split': Process each channel separately (default).
                - 'combined': Process all channels as RGB or grayscale.
            channels (list of int or None): Specific channel indices to use. If None,
                uses all available. Default: None.
            preprocessing_fn (callable or None): Custom preprocessing function applied
                to each image before the model. Signature: `fn(image: ndarray) -> ndarray`.
                Example: normalization, contrast adjustment, etc. Default: None.
            custom_transformations (transforms object or None): PyTorch transforms to apply
                before the model (e.g., resize, normalize). If None, defaults are used
                based on `resize_size` and `pad_size`. Default: None.
            append (bool): If True, append to existing embeddings in `results.embeddings`.
                If False (default), replace. Default: False.
            resize_size (int or None): Image resize dimension (square). Default: 224.
                Set to None to skip resizing.
            pad_size (int or None): Padding size (square). Default: None (no padding).
            checkpoint_path (str or None): HDF5 file path for lazy storage and resumable
                processing. If file exists, processing resumes from last checkpoint.
                If None, embeddings stored in-memory. Default: None.
            force_rgb (bool): If True, convert grayscale to RGB before model. Default: True.
            l2_normalize_channels (bool): In ``channel_mode='split'``, L2-normalize each
                channel embedding before concatenation so channels contribute equally.
                Set False to preserve raw channel magnitudes (one channel may dominate).
                Ignored in combined mode. Default: True.
            save_every (int): Save checkpoint every N batches (when using checkpoint_path).
                Default: 5.
            lazy_checkpoint (bool): If True and checkpoint_path is set, enable lazy
                loading (embeddings not kept in-memory). If False, load all embeddings
                after processing. Default: True.
            force_reprocess (bool): If True and ``checkpoint_path`` already exists, delete
                that file and process all requested images from scratch (no resume).
                Default: False.

        Returns:
            None. Modifies `self.results` in-place with embeddings, metadata, and paths.

        Raises:
            FileNotFoundError: If file_df not set (call `find_files()` or `set_file_df()` first).
            ValueError: If filters/exclude reference non-existent columns or produce no data.
            RuntimeError: If model inference fails or checkpoint is corrupted.
            IOError: If checkpoint file cannot be created or read.

        Example:
            Process images with DINOv2 model:

            >>> from phenome import PhenoMe, load_dinov2_model
            >>> pm = PhenoMe(device="cuda")
            >>> pm.find_files("images/")
            >>> model = load_dinov2_model("dinov2_vitb14")
            >>> pm.process_images(model, batch_size=64, resize_size=224)
            >>> print(pm.results.embeddings.shape)
            (1000, 768)

            With filtering and checkpoint:

            >>> pm.process_images(
            ...     model,
            ...     filters={"batch": ["batch_1", "batch_2"]},
            ...     checkpoint_path="embeddings.h5",
            ...     lazy_checkpoint=True
            ... )

        See Also:
            `find_files`: Discover and organize image files.
            `process_temporal_images`: Incrementally add images to existing checkpoint.
            `get_embeddings`: Retrieve embeddings (handles lazy loading).
            `compute_properties`: Extract morphological properties from embeddings.
        """

        file_df = self._require_file_df("process_images")
        self._validate_process_images_inputs(file_df)
        filtered_data, cur_params, _final_data_dir = self._prepare_for_process_images(
            file_df,
            filters,
            exclude,
            append,
            channel_mode=channel_mode,
            channels=channels,
            resize_size=resize_size,
            pad_size=pad_size,
            force_rgb=force_rgb,
            l2_normalize_channels=l2_normalize_channels,
            preprocessing_fn=preprocessing_fn,
        )
        all_requested_data = list(filtered_data)

        ckpt, requested_paths, requested_meta_keys, filtered_data, cur_params = (
            self._setup_checkpoint_resume(
                checkpoint_path,
                cur_params,
                filtered_data,
                lazy=lazy_checkpoint,
                force_reprocess=force_reprocess,
            )
        )
        n_pre_existing = ckpt.n_committed if ckpt is not None else 0
        self._processing_params = cur_params

        if not filtered_data:
            if checkpoint_path and ckpt:
                logger.info("All images already processed. Loading from checkpoint.")
                self._setup_lazy_results(
                    ckpt,
                    requested_paths=requested_paths,
                    requested_meta_keys=requested_meta_keys,
                    current_requested_data=all_requested_data,
                )
            else:
                import warnings

                warnings.warn(
                    "No images to process after filtering. Check your filters/exclude arguments.",
                    stacklevel=2,
                )
            return

        cur_t = custom_transformations or self._build_transforms(resize_size, pad_size)
        self.image_transforms = cur_t

        dl = self._create_dataloader(
            filtered_data,
            transform=cur_t,
            cur_params=cur_params,
            preprocessing_fn=preprocessing_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        ckpt_used = self._run_embedding_extraction(
            model_wrapper,
            dl,
            checkpoint_path=checkpoint_path,
            ckpt=ckpt,
            cur_params=cur_params,
            save_every=save_every,
            lazy_checkpoint=lazy_checkpoint,
        )

        self._finalize_processing(
            checkpoint_path,
            ckpt_used,
            requested_paths,
            requested_meta_keys,
            len(filtered_data),
            all_requested_data=all_requested_data,
            n_pre_existing=n_pre_existing,
            n_load_failures=getattr(dl.dataset, "load_failure_count", 0),
        )

    # ------------------------------------------------------------------
    # Temporal images (in-memory only, no checkpoint persistence)
    # ------------------------------------------------------------------

    def process_temporal_images(
        self,
        model_wrapper: ModelWrapper,
        files: str | list[str] | pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4,
        channel_mode: Literal["split", "combined"] | None = None,
        channels: list[int] | None = None,
        preprocessing_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        custom_transformations: Any | None = None,
        resize_size: int | None = None,
        pad_size: int | None = None,
        force_rgb: bool | None = None,
        l2_normalize_channels: bool | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        """Process new images in-memory (temporary) and append to the current session.

        Use this to explore additional images (e.g., from a new condition or replicate)
        without re-running process_images. Temporal images are not persisted to checkpoint;
        visualize or export before calling clear_temporal_data() or reset().

        Requires process_images() to have been run first. Processing parameters
        (channel_mode, channels, resize_size, etc.) are inherited from the base run
        when not specified. Temporal rows get metadata['source'] = 'NEW'.
        """
        if self.results.n_images == 0:
            raise ValueError("No existing data. Run process_images() first.")

        # Resolve to DataFrame (str→find_files; list→DataFrame; already DataFrame→use)
        if isinstance(files, pd.DataFrame):
            file_df = files
        elif isinstance(files, list):
            paths = [str(p) for p in files if p]
            file_df = (
                pd.DataFrame({"file_path": paths}) if paths else pd.DataFrame(columns=["file_path"])
            )
        elif isinstance(files, str):
            if os.path.isfile(files):
                file_df = pd.DataFrame({"file_path": [files]})
            else:
                file_df = self._file_discovery._find_files_from_dir(files, extensions=extensions)
        else:
            raise ValueError(f"files must be str, list, or DataFrame, got: {type(files)}")

        if not file_df.empty:
            combined = (
                pd.concat([self._file_df, file_df], ignore_index=True)
                if self._file_df is not None
                else file_df
            )
            self._file_df = combined.reset_index(drop=True).copy()
            self._init_file_df_metadata(self._file_df)

        fdata = self._prepare_filtered_data(file_df)
        if not fdata:
            return

        p = (
            (self._db.get_processing_params() if self._db else None)
            or self._processing_params
            or {}
        )
        cur_params = self._resolve_processing_params(
            p,
            channel_mode=channel_mode,
            channels=channels,
            resize_size=resize_size,
            pad_size=pad_size,
            force_rgb=force_rgb,
            l2_normalize_channels=l2_normalize_channels,
        )

        tr = PhenoMeResults()
        tb: list[np.ndarray] = []
        rs = cur_params["resize_size"]
        ps = cur_params["pad_size"]
        rs_int: int | None = int(rs) if isinstance(rs, (int, float)) and rs != "none" else None
        ps_int: int | None = int(ps) if isinstance(ps, (int, float)) and ps != "none" else None
        if rs_int is None and isinstance(rs, str) and rs.lower() == "none":
            rs_int = None
        if ps_int is None and isinstance(ps, str) and ps.lower() == "none":
            ps_int = None
        ct = custom_transformations or self._build_transforms(rs_int, ps_int)
        dl = self._create_dataloader(
            fdata,
            transform=ct,
            cur_params=cur_params,
            preprocessing_fn=preprocessing_fn or self.preprocessing_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self._run_embedding_extraction(
            model_wrapper, dl, results=tr, emb_buffer=tb, cur_params=cur_params
        )
        if not tb:
            return

        n_new = len(tb)
        new_emb = np.stack(tb).astype(np.float32)
        if self.embedding_dim > 0 and new_emb.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: {new_emb.shape[1]} vs {self.embedding_dim}"
            )

        self.results.img_path.extend(tr.img_path)
        # Ensure temporal metadata has same keys as base so plots include them (color_by, filters)
        base_keys = get_all_metadata_keys(self.results)
        meta_list = []
        for m in tr.metadata:
            meta = {**(m or {}), "source": "NEW"}
            for k in base_keys:
                if k.lower() == "source":
                    continue
                if not any(ok.lower() == k.lower() for ok in meta):
                    meta[k] = "Temporal"
            meta_list.append(meta)
        self.results.metadata.extend(meta_list)
        self.results.properties.extend([{} for _ in range(n_new)])

        if self._db is not None:
            self._temporal_embeddings = (
                np.concatenate([self._temporal_embeddings, new_emb], axis=0)
                if self._temporal_embeddings is not None
                else new_emb
            )
        else:
            self.results.embeddings = (
                np.concatenate([self.results.embeddings, new_emb], axis=0)
                if self.results.embeddings is not None
                else new_emb
            )

        logger.info("Added %d temporal images (total: %d).", n_new, self.results.n_images)

    def clear_temporal_data(self) -> int:
        """Remove all temporal images (metadata['source'] == 'NEW') from the session.

        Returns:
            int: Number of temporal rows removed.
        """
        keep_mask = [
            not (isinstance(m, dict) and m.get("source") == "NEW") for m in self.results.metadata
        ]
        n_temporal = sum(not k for k in keep_mask)
        if n_temporal == 0:
            logger.info("No temporal data to clear.")
            return 0
        keep_idx = [i for i, k in enumerate(keep_mask) if k]
        self.results.img_path = [self.results.img_path[i] for i in keep_idx]
        self.results.metadata = [self.results.metadata[i] for i in keep_idx]
        self.results.properties = [self.results.properties[i] for i in keep_idx]
        self._temporal_embeddings = None
        # _db_indices unchanged: temporal rows are never in the checkpoint; only base rows
        # (0 .. _temporal_start_idx-1) are stored, and we only remove temporal rows.
        if self._db is None and self.results.embeddings is not None:
            self.results.embeddings = self.results.embeddings[np.array(keep_idx, dtype=np.int64)]
        self._temporal_start_idx = len(keep_idx)
        logger.info("Cleared %d temporal images (remaining: %d).", n_temporal, len(keep_idx))
        return n_temporal

    # ------------------------------------------------------------------
    # Inspect / Info
    # ------------------------------------------------------------------

    def inspect_data(self) -> pd.DataFrame:
        """Inspect image and mask dimensions, shapes, and data ranges. Delegates to FileDiscovery.

        Uses the internally stored file_df (set via set_file_df or find_files).
        Raises if file_df is not available.
        """
        file_df = self._require_file_df("inspect_data")
        return self._file_discovery.inspect_data(file_df)

    # ------------------------------------------------------------------
    # Info / accessors
    # ------------------------------------------------------------------

    def get_image_info(self, idx: int, distance_results: pd.DataFrame | None = None) -> dict:
        """Return metadata, properties, and optional distance for image idx.

        Args:
            idx: Image index (0 to n_images-1).
            distance_results: Optional DataFrame from compute_reference_distances.

        Returns:
            dict: Keys image_index, image_name, image_path, metadata keys, property keys,
                distance (if distance_results provided), is_reference (if applicable).
        """
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"idx must be int, got: {type(idx)}")
        idx = int(idx)
        n = self.results.n_images
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range [0, {n - 1}]")

        path = self.results.img_path[idx]
        path_for_name = path[0] if isinstance(path, list) else path
        info: dict[str, Any] = {
            "image_index": idx,
            "image_name": os.path.basename(path_for_name),
            "image_path": path,
        }

        if idx < len(self.results.metadata) and isinstance(self.results.metadata[idx], dict):
            info.update(self.results.metadata[idx])

        if idx < len(self.results.properties) and isinstance(self.results.properties[idx], dict):
            for k, v in self.results.properties[idx].items():
                if v is None:
                    info[k] = None
                elif isinstance(v, bool):
                    info[k] = v
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    info[k] = None if np.isnan(v) else float(v)
                else:
                    info[k] = v

        if distance_results is not None and "distance" in distance_results.columns:
            if IMAGE_INDEX in distance_results.columns:
                matched = distance_results.loc[distance_results[IMAGE_INDEX] == idx]
                if not matched.empty:
                    row = matched.iloc[0]
                    info["distance"] = row["distance"]
                    if "is_reference" in distance_results.columns:
                        info["is_reference"] = row["is_reference"]
            else:
                info["distance"] = distance_results.at[idx, "distance"]
                if "is_reference" in distance_results.columns:
                    info["is_reference"] = distance_results.at[idx, "is_reference"]
        return info

    def get_available_metadata_keys(self) -> list[str]:
        """Return sorted metadata keys."""
        return get_all_metadata_keys(self.results)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_results(
        self,
        path: str | None = None,
        compression: str = "gzip",
    ) -> None:
        """Save results to HDF5 (atomic write or in-place flush).

        When a checkpoint database is already active (``self._db`` is set),
        the data is already on disk; this method flushes any uncommitted
        buffers and logs the existing path.  When no database is active,
        writes ``self.results`` to a new HDF5 file using a temporary-file +
        atomic rename for crash safety.

        For full reproducibility, also call [export_experiment_config](pipeline.md#api-phenome-export_experiment_config)
        to save seed, use_gpu_for_dr, reference_filters, and model name
        (not stored in the checkpoint).

        Args:
            path: Explicit output path.
            compression: HDF5 compression algorithm (used only for new files).
        """
        target_path = _resolve_results_hdf5_path(path)

        parent = os.path.dirname(target_path) or "."
        os.makedirs(parent, exist_ok=True)

        if self._db is not None:
            db_path = os.path.abspath(self._db.path)

            # Sync in-memory properties to DB buffers if missing from disk.
            # This ensures that properties computed in-memory (e.g. without passing
            # checkpoint_path to compute_properties) are included in the saved file.
            if self.results.has_properties:
                n_db = self._db.n_committed_props + self._db.properties_buffered
                if n_db < self.results.n_images:
                    db_indices = self._db_indices
                    if db_indices is not None and (
                        len(db_indices) != self.results.n_images
                        or not np.array_equal(db_indices, np.arange(len(db_indices)))
                    ):
                        raise ValueError(
                            "Cannot append properties to checkpoint after a filtered lazy "
                            "reload (non-contiguous row mapping). Recompute properties with "
                            "checkpoint_path= or export with write_results_to_hdf5()."
                        )
                    to_buffer = self.results.properties[n_db:]
                    self._db.buffer_properties(to_buffer)
                    if self._ram_internal and len(self._ram_internal) >= self.results.n_images:
                        to_buffer_int = self._ram_internal[n_db:]
                        self._db.buffer_internal(to_buffer_int)

            if os.path.abspath(target_path) == db_path:
                if self._db.embeddings_buffered:
                    self._db.commit_embeddings()
                if self._db.properties_buffered:
                    self._db.commit_properties()
                if self._db._file is not None:
                    self._db._file.flush()
                logger.info("Results already on disk at %s (flushed).", db_path)
                return

            if self._db.embeddings_buffered:
                self._db.commit_embeddings()
            if self._db.properties_buffered:
                self._db.commit_properties()

            logger.debug("Copying HDF5 database from %s to %s …", db_path, target_path)
            import shutil

            was_open = self._db._file is not None
            if self._db._file is not None:
                self._db._file.flush()
                self._db._file.close()
                self._db._file = None

            shutil.copy2(db_path, target_path)
            self._db.path = target_path

            if was_open:
                self._db._ensure_open()

            logger.info("Results saved to %s", target_path)
            return

        if self.results.n_images == 0:
            raise ValueError("No results to save (img_path is empty).")

        has_emb = self.results.has_embeddings
        has_props = self.results.has_properties
        if not has_emb and not has_props:
            raise ValueError("No embeddings or properties to save.")

        CheckpointManager.write_results_to_hdf5(
            self.results,
            target_path,
            compression=compression,
            processing_params=self._processing_params,
            internal=self._ram_internal,
        )
        logger.info("Results saved to %s", target_path)

    def export_experiment_config(
        self,
        path: str,
        reference_filters: dict[str, Any] | None = None,
        model_name: str | None = None,
        checkpoint_path: str | None = None,
        **extra: Any,
    ) -> None:
        """Export experiment configuration for reproducibility.

        Writes a JSON file with parameters not stored in the HDF5 checkpoint,
        so that analyses can be fully reproduced. Call after save_results()
        and pass the same reference_filters used for distance analysis.

        Args:
            path: Output path for config.json.
            reference_filters: Optional dict used for compute_reference_distances
                (e.g. {'condition': 'Control'}). Include for full traceability.
            model_name: Optional model identifier (e.g. 'dinov2_vitb14_reg').
            checkpoint_path: Optional checkpoint path used during processing (if not
                provided and a checkpoint is open, uses self._db.path).
            **extra: Additional key-value pairs to include in the config.
        """
        try:
            from . import __version__ as pkg_version
        except ImportError:
            pkg_version = "unknown"

        config: dict[str, Any] = {
            "seed": self.seed,
            "use_gpu_for_dr": self.use_gpu_for_dr,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": pkg_version,
            **extra,
        }
        if reference_filters is not None:
            config["reference_filters"] = reference_filters
        if model_name is not None:
            config["model_name"] = model_name

        if self._processing_params is not None:
            # Make JSON-serializable (e.g. numpy types -> native)
            params_copy: dict[str, Any] = {}
            for k, v in self._processing_params.items():
                if v is not None and hasattr(v, "tolist"):
                    params_copy[k] = v.tolist()
                elif isinstance(v, bytes):
                    params_copy[k] = v.decode("utf-8")
                else:
                    params_copy[k] = v
            config["processing_params"] = params_copy

        if checkpoint_path is not None:
            config["checkpoint_path"] = checkpoint_path
        elif self._db is not None:
            config["checkpoint_path"] = os.path.abspath(self._db.path)

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Experiment config saved to %s", path)

    def export_dataset_table(
        self,
        output_path: str | None = None,
        dist_results: pd.DataFrame | None = None,
        include_embeddings: bool | Literal["separate"] = False,
        export_format: Literal["csv", "parquet", "excel"] = "csv",
    ) -> pd.DataFrame:
        """Export the dataset as a table (CSV, Parquet, or Excel).

        Args:
            output_path: Path to save the export. If None, only returns the DataFrame.
            dist_results: Optional DataFrame from compute_reference_distances.
            include_embeddings: If True, adds embedding columns; if 'separate', saves
                embeddings to a companion .npy file.
            export_format: Output format: 'csv', 'parquet', or 'excel'.

        Returns:
            DataFrame with image_index, image_path, metadata, properties, distances,
            and embeddings (if requested).
        """
        if self.results.n_images == 0:
            raise ValueError("No results to export. Run process_images() first.")

        include_emb_in_df = include_embeddings is True

        if include_emb_in_df and self._db is not None:
            results_for_export = PhenoMeResults(
                img_path=self.results.img_path,
                metadata=self.results.metadata,
                properties=self.results.properties,
                embeddings=self.get_embeddings(),
            )
        else:
            results_for_export = self.results

        df = build_export_dataframe(
            results_for_export,
            dist_results=dist_results,
            include_embeddings=include_emb_in_df,
        )

        if output_path is not None:
            out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            base, ext = os.path.splitext(output_path)
            ext_lower = ext.lower() if ext else ""
            if ext_lower == ".parquet":
                write_format = "parquet"
            elif ext_lower in (".xlsx", ".xls"):
                write_format = "excel"
            elif ext_lower == ".csv":
                write_format = "csv"
            else:
                write_format = export_format
                ext_map = {"csv": ".csv", "parquet": ".parquet", "excel": ".xlsx"}
                output_path = base + ext_map.get(write_format, ".csv")

            if write_format == "csv":
                df.to_csv(output_path, index=False)
            elif write_format == "parquet":
                df.to_parquet(output_path, index=False)
            elif write_format == "excel":
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {export_format}")

            if include_embeddings == "separate":
                embeddings = self.get_embeddings()
                if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                    emb_path = base + "_embeddings.npy"
                    np.save(emb_path, embeddings)
                    logger.info("Embeddings saved to %s", emb_path)

            logger.info(
                "Dataset exported to %s (%d rows, %d columns)",
                output_path,
                len(df),
                len(df.columns),
            )

        return df

    @staticmethod
    def _checkpoint_has_embeddings(ckpt: CheckpointManager) -> bool:
        """Return True if *ckpt* stores at least one embedding row."""
        return bool((ckpt.embedding_dim or 0) > 0 and ckpt.n_committed > 0)

    @staticmethod
    def _checkpoint_has_property_content(ckpt: CheckpointManager) -> bool:
        """Return True if *ckpt* stores real property columns/values.

        ``n_committed_props > 0`` alone is insufficient: ``write_results_to_hdf5``
        can set that counter for a list of empty ``{}`` rows and create an empty
        ``/properties`` group.
        """
        if ckpt.n_committed_props <= 0:
            return False
        f = ckpt._file
        if f is not None and "properties" in f:
            prop_group = f["properties"]
            if len(prop_group.keys()) > 0:
                return True
        props = ckpt.load_properties_all()
        return any(isinstance(p, dict) and p for p in props)

    def _open_results_checkpoint(
        self,
        path: str | None,
        *,
        method_name: str,
        lazy_checkpoint: bool = True,
    ) -> tuple[str, CheckpointManager]:
        """Resolve and open a results HDF5 file without mutating pipeline state.

        Validates ``file_df`` availability and file format, but does **not** close
        ``self._db`` or update ``self._file_df``. Callers must validate content,
        then call :meth:`_adopt_checkpoint_load` before setup.
        """
        filename = _resolve_results_hdf5_path(path)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Results file not found: {filename}")

        fmt = CheckpointManager.detect_format(filename)
        if fmt != "hdf5":
            raise ValueError(f"Unrecognised format for '{filename}'. Expected HDF5.")

        self._require_file_df(method_name)
        ckpt = CheckpointManager(filename, lazy=lazy_checkpoint)
        return filename, ckpt

    def _adopt_checkpoint_load(
        self,
        ckpt: CheckpointManager,
        *,
        method_name: str,
    ) -> tuple[list[dict[str, Any]], set | None, set | None]:
        """Adopt an opened checkpoint: close previous ``_db``, refresh file_df alignment.

        Returns:
            ``(current_requested_data, requested_paths, requested_meta_keys)``.
        """
        if self._db is not None and self._db is not ckpt:
            try:
                self._db.close()
            except OSError as e:
                logger.debug("Error closing checkpoint before load: %s", e)
            self._db = None

        self._processing_params = ckpt.get_processing_params()

        file_df = self._require_file_df(method_name)
        self._file_df = file_df.reset_index(drop=True).copy()
        self._init_file_df_metadata(self._file_df)

        current_requested_data = self._prepare_filtered_data(file_df)
        requested_paths = {_path_repr(d["file_path"]) for d in current_requested_data}

        meta_keys_builder: set = set()
        requested_meta_keys: set | None = None
        for d in current_requested_data:
            meta = d.get("metadata", {})
            try:
                key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
            except ValueError:
                requested_meta_keys = None
                break
            meta_keys_builder.add(key)
        else:
            requested_meta_keys = meta_keys_builder

        return current_requested_data, requested_paths, requested_meta_keys

    def _log_checkpoint_load(self, filename: str, ckpt: CheckpointManager) -> None:
        """Emit the standard post-load summary for a checkpoint."""
        if self.results.n_images > 0:
            sample_path = self.results.primary_path(0)
            if not os.path.exists(sample_path):
                logger.warning(
                    "Image not found at: %s\n"
                    "If you moved the dataset, call find_files at the new location "
                    "first, then load_results / load_embeddings / load_properties.",
                    sample_path,
                )

        if self.results.has_properties:
            self._warn_if_nan_properties()

        n_img = self.results.n_images
        meta_keys = get_all_metadata_keys(self.results)
        n_meta = len(meta_keys)

        props = self.results.properties
        prop_keys = sorted(props[0].keys()) if props and isinstance(props[0], dict) else []
        n_prop = len(prop_keys)

        n_emb = ckpt.n_committed
        emb_dim = ckpt.embedding_dim or 0
        # Props-only cold start closes the handle; n_committed then reads as 0.
        if n_emb > 0 and emb_dim > 0:
            loc = "lazy, on disk" if (ckpt.lazy and ckpt._file is not None) else "in RAM"
            emb_str = f", {emb_dim}-dim embeddings ({loc})"
        else:
            emb_str = ", no embeddings"

        logger.info(
            "Loaded %d images from %s (hdf5)%s.\nContent: %d metadata keys %s, %d properties %s.",
            n_img,
            filename,
            emb_str,
            n_meta,
            f"({', '.join(meta_keys[:5])}{'...' if n_meta > 5 else ''})",
            n_prop,
            f"({', '.join(prop_keys[:5])}{'...' if n_prop > 5 else ''})",
        )

    def _overlay_properties_from_checkpoint(self, path: str | None) -> None:
        """Align and overlay properties from *path* onto existing ``self.results``.

        Does not replace an embeddings ``self._db`` when that handle points at a
        different file. Properties and internal flags are loaded eagerly.
        """
        filename = _resolve_results_hdf5_path(path)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Results file not found: {filename}")

        fmt = CheckpointManager.detect_format(filename)
        if fmt != "hdf5":
            raise ValueError(f"Unrecognised format for '{filename}'. Expected HDF5.")

        active_db = self._db
        same_as_active = active_db is not None and os.path.abspath(
            active_db.path
        ) == os.path.abspath(filename)
        if same_as_active:
            assert active_db is not None
            ckpt = active_db
        else:
            ckpt = CheckpointManager(filename, lazy=True)

        try:
            if not self._checkpoint_has_property_content(ckpt):
                raise ValueError(
                    f"Checkpoint '{filename}' has no property content "
                    f"(n_committed_props={ckpt.n_committed_props}). Pass a properties "
                    f"checkpoint from compute_properties(..., checkpoint_path=...)."
                )

            all_props = ckpt.load_properties_all()
            all_internal = ckpt.load_internal_all()
            ckpt_paths = ckpt.get_committed_paths_list()
            ckpt_metadata = ckpt.get_committed_metadata_list()

            image_paths = list(self.results.img_path)
            image_metadata = (
                list(self.results.metadata)
                if self.results.metadata is not None
                else [{} for _ in range(len(image_paths))]
            )

            filtered_props: list[dict[str, Any]] = []
            filtered_internal: list[dict[str, Any]] = []
            try:
                id_to_idx = build_metadata_key_index(ckpt_metadata, on_duplicate="error")
                for meta in image_metadata:
                    key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
                    idx = id_to_idx.get(key, -1)
                    if 0 <= idx < len(all_props):
                        filtered_props.append(dict(all_props[idx]))
                        filtered_internal.append(
                            dict(all_internal[idx]) if 0 <= idx < len(all_internal) else {}
                        )
                    else:
                        filtered_props.append({})
                        filtered_internal.append({})
            except ValueError:
                path_lookup = _build_unique_path_lookup(ckpt_paths)
                filtered_props = []
                filtered_internal = []
                for p in image_paths:
                    idx = _lookup_unique_path_index(path_lookup, p)
                    if idx is not None and 0 <= idx < len(all_props):
                        filtered_props.append(dict(all_props[idx]))
                        filtered_internal.append(
                            dict(all_internal[idx]) if 0 <= idx < len(all_internal) else {}
                        )
                    else:
                        filtered_props.append({})
                        filtered_internal.append({})

            n = self.results.n_images
            if len(filtered_props) < n:
                filtered_props.extend([{} for _ in range(n - len(filtered_props))])
            if len(filtered_internal) < n:
                filtered_internal.extend([{} for _ in range(n - len(filtered_internal))])

            n_matched = sum(1 for p in filtered_props[:n] if isinstance(p, dict) and p)
            if n > 0 and n_matched == 0:
                raise ValueError(
                    f"No properties from '{filename}' matched the {n} loaded image(s). "
                    f"Check that metadata ids (or paths) align with the embeddings "
                    f"checkpoint / current file_df."
                )
            if n_matched < n:
                logger.warning(
                    "Properties from %s matched %d/%d images; unmatched rows get empty dicts.",
                    filename,
                    n_matched,
                    n,
                )

            self.results.properties = filtered_props[:n]
            self._ram_internal = filtered_internal[:n]

            if self.results.has_properties:
                self._warn_if_nan_properties()

            prop_keys = (
                sorted(next(p.keys() for p in filtered_props[:n] if isinstance(p, dict) and p))
                if n_matched
                else []
            )
            logger.info(
                "Loaded properties from %s onto %d images (%d rows matched, %d property keys%s).",
                filename,
                n,
                n_matched,
                len(prop_keys),
                f" ({', '.join(prop_keys[:5])}{'...' if len(prop_keys) > 5 else ''})"
                if prop_keys
                else "",
            )
        finally:
            if not same_as_active:
                try:
                    ckpt.close()
                except OSError as e:
                    logger.debug("Error closing properties checkpoint after overlay: %s", e)

    def load_results(
        self,
        path: str | None = None,
        lazy_checkpoint: bool = True,
    ) -> None:
        """Load and use an existing results/checkpoint file.

        Metadata and properties are loaded immediately. Embeddings are either
        loaded into RAM (lazy_checkpoint=False) or accessed on-demand from
        disk (lazy_checkpoint=True).

        Uses the internally stored file_df (from [find_files](pipeline.md#api-phenome-find_files) or
        [set_file_df](pipeline.md#api-phenome-set_file_df)) to resolve paths so the checkpoint works on this
        machine. Call find_files or set_file_df first.

        For separate embeddings and properties checkpoints, prefer
        [load_embeddings](pipeline.md#api-phenome-load_embeddings) then
        [load_properties](pipeline.md#api-phenome-load_properties).

        Args:
            path: Path to .h5 or .hdf5 file.
            lazy_checkpoint: If True (default), keep the checkpoint file open.
                If False, load all data into RAM and close the file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not recognised as HDF5.
            RuntimeError: If file_df is not available (call find_files or set_file_df first).
            ConcurrentCheckpointAccessError: If the checkpoint is already open in
                another notebook or process (only one instance can access it at a time).
        """
        filename, ckpt = self._open_results_checkpoint(
            path,
            method_name="load_results",
            lazy_checkpoint=lazy_checkpoint,
        )
        try:
            current_requested_data, requested_paths, requested_meta_keys = (
                self._adopt_checkpoint_load(ckpt, method_name="load_results")
            )
            self._setup_lazy_results(
                ckpt,
                requested_paths=requested_paths,
                requested_meta_keys=requested_meta_keys,
                current_requested_data=current_requested_data,
            )
            self._log_checkpoint_load(filename, ckpt)
        except Exception:
            if self._db is not ckpt:
                with contextlib.suppress(OSError):
                    ckpt.close()
            raise

    def load_embeddings(
        self,
        path: str | None = None,
        lazy_checkpoint: bool = True,
    ) -> None:
        """Load an embeddings checkpoint into the pipeline.

        Same path/file_df rebase flow as [load_results](pipeline.md#api-phenome-load_results),
        but requires the file to contain embeddings. Sets ``self._db`` for lazy
        ``get_embeddings()`` access when ``lazy_checkpoint=True``.

        Typical two-file workflow::

            pheno.find_files(data_dir)
            pheno.load_embeddings("run_embeddings.h5")
            pheno.load_properties("run_properties.h5")

        Args:
            path: Path to .h5 or .hdf5 embeddings checkpoint.
            lazy_checkpoint: If True (default), keep the checkpoint file open.
                If False, load all data into RAM and close the file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not recognised as HDF5, or the
                checkpoint has no embeddings.
            RuntimeError: If file_df is not available (call find_files or set_file_df first).
            ConcurrentCheckpointAccessError: If the checkpoint is already open in
                another notebook or process (only one instance can access it at a time).
        """
        filename, ckpt = self._open_results_checkpoint(
            path,
            method_name="load_embeddings",
            lazy_checkpoint=lazy_checkpoint,
        )
        try:
            if not self._checkpoint_has_embeddings(ckpt):
                raise ValueError(
                    f"Checkpoint '{filename}' has no embeddings "
                    f"(embedding_dim={ckpt.embedding_dim or 0}, "
                    f"n_committed={ckpt.n_committed}). "
                    f"Pass an embeddings checkpoint from process_images(..., checkpoint_path=...)."
                )

            current_requested_data, requested_paths, requested_meta_keys = (
                self._adopt_checkpoint_load(ckpt, method_name="load_embeddings")
            )
            self._setup_lazy_results(
                ckpt,
                requested_paths=requested_paths,
                requested_meta_keys=requested_meta_keys,
                current_requested_data=current_requested_data,
            )
            self._log_checkpoint_load(filename, ckpt)
        except Exception:
            if self._db is not ckpt:
                with contextlib.suppress(OSError):
                    ckpt.close()
            raise

    def load_properties(
        self,
        path: str | None = None,
        lazy_checkpoint: bool = True,
    ) -> None:
        """Load a properties checkpoint into the pipeline.

        When results already contain images (e.g. after
        [load_embeddings](pipeline.md#api-phenome-load_embeddings) or
        ``process_images``), properties are aligned by metadata key then path
        and overlaid onto the existing rows without replacing an embeddings
        ``self._db`` from a different file.

        When no results are loaded yet, performs a full checkpoint setup like
        [load_results](pipeline.md#api-phenome-load_results). For a
        properties-only file, the HDF5 handle is closed after load (properties
        are eager); ``self._db`` is kept only if that file also has embeddings.

        Args:
            path: Path to .h5 or .hdf5 properties checkpoint.
            lazy_checkpoint: Used on cold start only. If True (default), keep
                the file open when it also stores embeddings. Ignored when
                overlaying onto existing results.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not recognised as HDF5, or the
                checkpoint has no property content, or overlay matches no rows.
            RuntimeError: On cold start, if file_df is not available
                (call find_files or set_file_df first).
            ConcurrentCheckpointAccessError: If the checkpoint is already open in
                another notebook or process (only one instance can access it at a time).
        """
        if self.results.n_images > 0:
            self._overlay_properties_from_checkpoint(path)
            return

        filename, ckpt = self._open_results_checkpoint(
            path,
            method_name="load_properties",
            lazy_checkpoint=lazy_checkpoint,
        )
        try:
            if not self._checkpoint_has_property_content(ckpt):
                raise ValueError(
                    f"Checkpoint '{filename}' has no property content "
                    f"(n_committed_props={ckpt.n_committed_props}). Pass a properties "
                    f"checkpoint from compute_properties(..., checkpoint_path=...)."
                )

            current_requested_data, requested_paths, requested_meta_keys = (
                self._adopt_checkpoint_load(ckpt, method_name="load_properties")
            )
            self._setup_lazy_results(
                ckpt,
                requested_paths=requested_paths,
                requested_meta_keys=requested_meta_keys,
                current_requested_data=current_requested_data,
            )

            if not self._checkpoint_has_embeddings(ckpt):
                # Properties are already in results; no need to keep a props-only handle.
                try:
                    ckpt.close()
                except OSError as e:
                    logger.debug("Error closing properties-only checkpoint: %s", e)
                self._db = None
                self._db_indices = None

            self._log_checkpoint_load(filename, ckpt)
        except Exception:
            if self._db is not ckpt:
                with contextlib.suppress(OSError):
                    ckpt.close()
            raise

    @contextmanager
    def checkpoint_context(
        self,
        path: str | None = None,
        lazy_checkpoint: bool = True,
    ) -> Generator[Any, None, None]:
        """Context manager that loads a checkpoint and guarantees it is closed on exit.

        More reliable than relying on __del__ for cleanup. Use when you need to
        ensure the HDF5 file handle is released (e.g. before moving or deleting
        the file, or when opening multiple checkpoints in sequence).

        **Embeddings:** While the context is open, lazy embeddings are read from the
        HDF5 file via ``get_embeddings()``. On exit, if temporal embeddings were loaded
        in memory (``process_temporal_images`` while the checkpoint was open), the merged
        full embedding matrix is copied to ``results.embeddings`` and
        ``self._temporal_embeddings`` is cleared. Otherwise ``results.embeddings`` may
        stay ``None``; call ``load_results(..., lazy_checkpoint=True)`` or enter another
        ``checkpoint_context`` before using ``get_embeddings()`` again.

        Args:
            path: Checkpoint HDF5 path. If None, uses the path from the last
                ``load_results`` / ``process_images`` session when available.
            lazy_checkpoint: If True, keep the HDF5 file open for on-demand reads
                inside the context. Default: True.

        Example:
            pheno.find_files("path/to/images")
            with pheno.checkpoint_context("results.h5") as p:
                p.plot_pca(color_by="condition")
            # Checkpoint closed here
        """
        self.load_results(path, lazy_checkpoint=lazy_checkpoint)
        try:
            yield self
        finally:
            if self._db is not None:
                try:
                    if self._temporal_embeddings is not None and len(self._temporal_embeddings) > 0:
                        ckpt_emb = self.get_embeddings()
                        if ckpt_emb is not None and len(ckpt_emb) > 0:
                            self.results.embeddings = ckpt_emb
                            self._temporal_embeddings = None
                    self._db.close()
                except OSError as e:
                    logger.debug("Error closing checkpoint in context exit: %s", e)
                self._db = None
            self._db_indices = None

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        output_path: str = "pheno_report.html",
        title: str = "PhenoMe Analysis Report",
        config: Any | None = None,
        **overrides: Any,
    ) -> str:
        """Generate a comprehensive standalone HTML report from phenotyping results.

        Uses ReportConfig as the primary source of options. Pass config= for full control,
        or use **overrides to tweak individual settings (e.g. include_plots=False).

        Args:
            output_path: Path to save the HTML file.
            title: Report title displayed at the top.
            config: Optional ReportConfig for defaults. If None, uses ReportConfig().
            **overrides: Any ReportConfig field to override (e.g. include_plots=False,
                outlier_threshold=2.5, n_clusters=10).

        Returns:
            Path to the generated HTML file.
        """
        from .report.generator import generate_report as _generate_report

        return _generate_report(
            pipeline=self,
            output_path=output_path,
            title=title,
            config=config,
            **overrides,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_for_process_images(
        self,
        file_df: pd.DataFrame,
        filters: dict[str, list[Any]] | None,
        exclude: dict[str, list[Any]] | None,
        append: bool,
        *,
        channel_mode: str,
        channels: list[int] | None,
        resize_size: int | None,
        pad_size: int | None,
        force_rgb: bool,
        l2_normalize_channels: bool,
        preprocessing_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Prepare state and filtered data for process_images. Returns (filtered_data, cur_params, final_data_dir).
        final_data_dir is inferred from file_df paths for checkpoint internal use."""
        if not append:
            self.reset()
        else:
            if self._db is not None:
                if (
                    self._db_indices is not None
                    and len(self._db_indices) > 0
                    and self._db.n_committed > 0
                ):
                    try:
                        existing = self._db.load_embeddings_by_indices(self._db_indices)
                        if existing is not None and len(existing) > 0:
                            self._emb_buffer = list(existing)
                    except (OSError, KeyError, ValueError) as e:
                        logger.warning(
                            "Could not load existing embeddings from checkpoint for append: %s",
                            e,
                        )
                try:
                    self._db.close()
                except OSError as e:
                    logger.debug("Error closing checkpoint for append: %s", e)
                self._db = None
            if self.results.embeddings is not None:
                self._emb_buffer = list(self.results.embeddings)
                self.results.embeddings = None

        fdf = self._filter_dataframe(file_df, filters, exclude)
        filtered_data = self._prepare_filtered_data(fdf)

        final_data_dir: str | None = None
        if not file_df.empty:
            try:
                paths = file_df["file_path"].tolist()
                flat_paths = []
                for p in paths:
                    if isinstance(p, list):
                        flat_paths.extend([str(x) for x in p if x])
                    elif p:
                        flat_paths.append(str(p))
                if flat_paths:
                    abs_paths = [p for p in flat_paths if os.path.isabs(p)]
                    if abs_paths:
                        common = os.path.commonpath(abs_paths)
                        if common and common != "/":
                            if os.path.isfile(common):
                                common = os.path.dirname(common)
                            final_data_dir = common
                            logger.debug("Inferred data_dir from file_df: %s", final_data_dir)
            except (ValueError, OSError):
                pass

        cur_params = {
            "channel_mode": channel_mode,
            "channels": channels,
            "resize_size": resize_size,
            "pad_size": pad_size,
            "force_rgb": force_rgb,
            "l2_normalize_channels": l2_normalize_channels,
        }
        self.preprocessing_fn = preprocessing_fn
        self._processing_params = cur_params
        return filtered_data, cur_params, final_data_dir

    # ------------------------------------------------------------------
    # Internal file_df helpers
    # ------------------------------------------------------------------

    def _normalize_channel_names(self, channel_names: list[str] | None) -> list[str] | None:
        """Validate and copy optional channel labels for property suffixes.

        Names must be non-empty, unique (case-insensitive), and contain no
        whitespace. Length is checked later in ``compute_properties`` against
        the loaded channel count.
        """
        if channel_names is None:
            return None
        if not isinstance(channel_names, (list, tuple)):
            raise ValueError(
                f"channel_names must be a list of strings or None, got {type(channel_names)}."
            )
        if len(channel_names) == 0:
            raise ValueError("channel_names must be non-empty when provided.")
        normalized: list[str] = []
        seen_lower: set[str] = set()
        for i, name in enumerate(channel_names):
            if not isinstance(name, str):
                raise ValueError(f"channel_names[{i}] must be a str, got {type(name)}.")
            if not name or name != name.strip() or any(c.isspace() for c in name):
                raise ValueError(
                    f"channel_names[{i}]={name!r} must be a non-empty string with no whitespace."
                )
            key = name.lower()
            if key in seen_lower:
                raise ValueError(
                    f"channel_names must be unique (case-insensitive); duplicate {name!r}."
                )
            seen_lower.add(key)
            normalized.append(name)
        return normalized

    def _init_file_df_metadata(self, file_df: pd.DataFrame) -> None:
        """Initialize cached column names for image and mask paths from file_df.

        Expects ``file_df`` to contain at least a ``'file_path'`` column, possibly
        with list-valued entries for multi-channel images. When separate per-channel
        columns are present, they are detected by a simple naming convention.
        Mask filenames are taken from ``'mask_path'`` when available.
        """
        if "file_path" not in file_df.columns:
            raise ValueError("file_df must contain 'file_path' column.")

        self._image_path_col = "file_path"

        channel_cols: list[str] = []
        for col in file_df.columns:
            name = str(col).lower()
            if name.startswith("channel_") and name.endswith("_path"):
                channel_cols.append(col)
        channel_cols.sort()
        self._channel_path_cols = channel_cols or None

        if "mask_path" in file_df.columns:
            self._mask_path_col = "mask_path"
        else:
            self._mask_path_col = None

    def _resolve_processing_params(
        self,
        base: dict[str, Any],
        *,
        channel_mode: Literal["split", "combined"] | None = None,
        channels: list[int] | None = None,
        resize_size: int | None = None,
        pad_size: int | None = None,
        force_rgb: bool | None = None,
        l2_normalize_channels: bool | None = None,
    ) -> dict[str, Any]:
        """Resolve processing params from base dict and overrides.

        Used by process_temporal_images to inherit from the base run when overrides
        are not specified. Bytes values (e.g. from HDF5) are decoded to str.
        """

        def _v(k: str, d: Any) -> Any:
            x = base.get(k)
            return x.decode() if isinstance(x, bytes) else (x if x is not None else d)

        return {
            "channel_mode": channel_mode or _v("channel_mode", "split"),
            "channels": channels if channels is not None else base.get("channels"),
            "resize_size": resize_size if resize_size is not None else _v("resize_size", 224),
            "pad_size": pad_size if pad_size is not None else base.get("pad_size"),
            "force_rgb": force_rgb if force_rgb is not None else _v("force_rgb", True),
            "l2_normalize_channels": (
                l2_normalize_channels
                if l2_normalize_channels is not None
                else _v("l2_normalize_channels", True)
            ),
        }

    def _create_dataloader(
        self,
        filtered_data: list[dict[str, Any]],
        *,
        transform: Any,
        cur_params: dict[str, Any],
        preprocessing_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create DataLoader for embedding extraction."""
        ch_mode = cur_params.get("channel_mode", "split")
        ch_mode_val: Literal["split", "combined"] = cast(
            Literal["split", "combined"],
            ch_mode if ch_mode in ("split", "combined") else "split",
        )
        ch = cur_params.get("channels")
        ch_val: list[int] | None = (
            list(ch)
            if ch is not None and hasattr(ch, "__iter__") and not isinstance(ch, str)
            else (ch if isinstance(ch, list) else None)
        )
        if ch_val is not None and not all(isinstance(x, int) for x in ch_val):
            ch_val = [int(x) for x in ch_val] if ch_val else None
        force_rgb_val = (
            bool(cur_params["force_rgb"]) if cur_params.get("force_rgb") is not None else True
        )
        ds = PhenoMeDataset(
            filtered_data,
            transform=transform,
            channel_mode=ch_mode_val,
            channels=ch_val,
            preprocessing_fn=preprocessing_fn,
            force_rgb=force_rgb_val,
        )
        nw = _effective_num_workers(num_workers)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if nw > 0 else None,
            persistent_workers=False,
        )

    def _run_embedding_extraction(
        self,
        model_wrapper: ModelWrapper,
        dataloader: DataLoader,
        *,
        results: PhenoMeResults | None = None,
        emb_buffer: list[np.ndarray] | None = None,
        checkpoint_path: str | None = None,
        ckpt: CheckpointManager | None = None,
        cur_params: dict[str, Any] | None = None,
        save_every: int = 5,
        lazy_checkpoint: bool = True,
    ) -> CheckpointManager | None:
        """Run embedding extraction via EmbeddingExtractor. Thin wrapper for unit-testing."""
        if results is None:
            results = self.results
        if emb_buffer is None:
            emb_buffer = self._emb_buffer
        do_l2 = True
        if cur_params is not None and cur_params.get("l2_normalize_channels") is not None:
            do_l2 = bool(cur_params["l2_normalize_channels"])
        extractor = EmbeddingExtractor(
            model_wrapper,
            self.device,
            l2_normalize_channels=do_l2,
        )
        return extractor.extract_from_dataloader(
            dataloader=dataloader,
            results=results,
            emb_buffer=emb_buffer,
            checkpoint_path=checkpoint_path,
            ckpt=ckpt,
            cur_params=cur_params,
            save_every=save_every,
            lazy_checkpoint=lazy_checkpoint,
        )

    def _build_transforms(self, resize_size: int | None, pad_size: int | None = None) -> Any:
        """Build torchvision Compose applied to each image before the model forward pass."""
        self.image_transforms = self._transform_builder.build(
            resize_size=resize_size, pad_size=pad_size
        )
        return self.image_transforms

    def _validate_process_images_inputs(self, file_df: pd.DataFrame) -> None:
        """Validate inputs for process_images method."""
        if not isinstance(file_df, pd.DataFrame):
            raise ValueError(f"file_df must be a DataFrame, got: {type(file_df)}")
        if file_df.empty:
            raise ValueError("file_df is empty.")
        if "file_path" not in file_df.columns:
            raise ValueError("file_df must contain 'file_path' column.")
        # Ensure file_path contains non-empty values
        for i, fpath in enumerate(file_df["file_path"]):
            if isinstance(fpath, (list, tuple)):
                if not fpath or not any(p for p in fpath if p):
                    raise ValueError(
                        f"Row {i}: file_path must not be empty or contain only empty strings."
                    )
            elif not fpath or not str(fpath).strip():
                raise ValueError(f"Row {i}: file_path must not be empty.")

    def _filter_dataframe(
        self,
        file_df: pd.DataFrame,
        filters: dict[str, list[Any]] | None,
        exclude: dict[str, list[Any]] | None = None,
    ) -> pd.DataFrame:
        """Filter DataFrame based on metadata filters and exclusions.

        Both filters and exclude use the same format: metadata key -> single value
        or list of values. Include fields are intersected; exclusion fields are
        combined, so a row matching any exclusion is removed after filtering.
        """
        fdf: pd.DataFrame = file_df.copy()
        if filters:
            for key, allowed in filters.items():
                cols = [c for c in fdf.columns if c.lower() == key.lower()]
                if cols:
                    col = cols[0]
                    mask = (
                        fdf[col].isin(allowed)
                        if isinstance(allowed, list)
                        else (fdf[col] == allowed)
                    )
                    fdf = fdf[mask]
        if exclude:
            exclude_mask = pd.Series(False, index=fdf.index)
            for key, excluded in exclude.items():
                cols = [c for c in fdf.columns if c.lower() == key.lower()]
                if cols:
                    col = cols[0]
                    excluded_list = excluded if isinstance(excluded, list) else [excluded]
                    exclude_mask = exclude_mask | fdf[col].isin(excluded_list)
            fdf = fdf[~exclude_mask]
        logger.debug("Processing %d images after filtering.", len(fdf))
        return fdf

    def _prepare_filtered_data(self, fdf: pd.DataFrame) -> list[dict[str, Any]]:
        """Prepare filtered data list from DataFrame."""
        non_fp_cols = [c for c in fdf.columns if str(c).lower() != "file_path"]
        lowered = [str(c).lower() for c in non_fp_cols]
        if len(set(lowered)) < len(lowered):
            seen: dict[str, str] = {}
            for orig, low in zip(non_fp_cols, lowered, strict=False):
                if low in seen:
                    raise ValueError(
                        f"Metadata columns '{seen[low]}' and '{orig}' collide when "
                        f"lowercased to '{low}'. Rename columns so lowercase names are unique."
                    )
                seen[low] = str(orig)
        filtered_data: list[dict[str, Any]] = []
        for _, row in fdf.iterrows():
            meta = {
                str(k).lower(): v for k, v in row.items() if str(k).lower() not in ("file_path",)
            }
            fpath = row["file_path"]
            file_path = fpath if isinstance(fpath, list) else str(fpath)
            filtered_data.append({"file_path": file_path, "metadata": meta})
        return filtered_data

    def _setup_checkpoint_resume(
        self,
        checkpoint_path: str | None,
        cur_params: dict[str, Any],
        filtered_data: list[dict[str, Any]],
        lazy: bool = True,
        force_reprocess: bool = False,
    ) -> tuple[
        CheckpointManager | None,
        set | None,
        set | None,
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        """Setup checkpoint resume and filter already-processed images.

        When an existing checkpoint is found, the ``channels`` parameter stored in it
        is adopted for processing the remaining images (overriding the caller's value).
        This guarantees that newly computed embeddings are compatible with those already
        committed.  All other parameter mismatches still raise ``ValueError``.

        Args:
            checkpoint_path: Path to HDF5 checkpoint.
            cur_params: Current processing parameters.
            filtered_data: List of images to be processed.
            lazy: If True, keep the checkpoint file open for lazy loading.
            force_reprocess: If True and the checkpoint file exists, delete it and return
                with no resume (all images in ``filtered_data`` will be processed).

        Returns:
            Tuple of (ckpt, requested_paths, requested_meta_keys, filtered_data, cur_params).
            ``cur_params`` may differ from the input when the checkpoint overrides ``channels``.

        Raises:
            RuntimeError: If the checkpoint has committed rows but none match the current
                run's metadata keys or paths (unless ``force_reprocess`` removed the file).
        """
        if checkpoint_path is None:
            return None, None, None, filtered_data, cur_params

        requested_paths = {_path_repr(d["file_path"]) for d in filtered_data}
        requested_meta_keys: set | None = None

        if not os.path.isfile(checkpoint_path):
            return None, requested_paths, None, filtered_data, cur_params

        if force_reprocess:
            logger.warning(
                "force_reprocess=True: removing existing checkpoint at %s and starting fresh.",
                checkpoint_path,
            )
            try:
                os.remove(checkpoint_path)
            except OSError as e:
                raise RuntimeError(
                    f"Could not remove checkpoint for force_reprocess: {checkpoint_path}"
                ) from e
            return None, requested_paths, None, filtered_data, cur_params

        ckpt = CheckpointManager(checkpoint_path, lazy=lazy)

        ckpt_params = ckpt.get_processing_params() or {}

        if "channels" in ckpt_params:
            ckpt_channels = ckpt_params["channels"]
            if ckpt_channels != cur_params.get("channels"):
                logger.info(
                    "Adopting channel selection from checkpoint: %s (caller requested: %s).",
                    ckpt_channels,
                    cur_params.get("channels"),
                )
                cur_params = dict(cur_params, channels=ckpt_channels)

        mismatches = ckpt.validate_processing_params(cur_params)
        if mismatches:
            logger.warning("Checkpoint parameter mismatch (%s):", checkpoint_path)
            for k in CheckpointManager._PARAM_KEYS:
                cv = cur_params.get(k)
                sv = ckpt_params.get(k, "<not stored>")
                diff = " <-- DIFF" if sv != cv else ""
                logger.warning("  %s: checkpoint=%s  current=%s%s", k, sv, cv, diff)
            ckpt.close()
            raise ValueError(
                "Checkpoint parameter mismatch. Use a different checkpoint path or "
                "delete the file to start fresh."
            )

        if ckpt.get_processing_params() is None:
            ckpt.set_processing_params(cur_params)

        try:
            requested_meta_keys = {
                metadata_to_stable_key(d.get("metadata", {})) for d in filtered_data
            }
        except ValueError:
            requested_meta_keys = None
            logger.info(
                "Metadata lacks atomic identifiers for checkpoint matching. "
                "Falling back to path-based matching."
            )

        if requested_meta_keys is not None:
            already_meta_keys = get_already_committed_metadata_keys(
                ckpt.get_committed_metadata_list(), ckpt.n_committed
            )
            if already_meta_keys:
                n_before = len(filtered_data)
                filtered_data = filter_items_not_in_checkpoint(
                    filtered_data,
                    already_meta_keys=already_meta_keys,
                )
                n_skipped = n_before - len(filtered_data)
                if ckpt.n_committed > 0 and n_skipped == 0:
                    ckpt.close()
                    raise RuntimeError(
                        "Checkpoint has committed rows but none match current metadata keys. "
                        "Call find_files at the new location and load_results(), or pass "
                        "force_reprocess=True to start fresh."
                    )
                logger.info(
                    "Found %d committed images; skipping %d (already in checkpoint), %d to process.",
                    ckpt.n_committed,
                    n_skipped,
                    len(filtered_data),
                )
        else:
            already_paths_list = ckpt.get_committed_paths_list()
            already_paths = get_already_committed_paths(already_paths_list, ckpt.n_committed)

            if already_paths:
                n_before = len(filtered_data)
                filtered_data = filter_items_not_in_checkpoint(
                    filtered_data,
                    already_paths=already_paths,
                    path_repr_fn=_path_repr,
                )
                n_skipped = n_before - len(filtered_data)
                if ckpt.n_committed > 0 and n_skipped == 0:
                    ckpt.close()
                    raise RuntimeError(
                        "Checkpoint has committed rows but none match your current files. "
                        "Call find_files at the new location and load_results(), or pass "
                        "force_reprocess=True to start fresh."
                    )
                logger.info(
                    "Found %d committed images; skipping %d (already in checkpoint), %d to process.",
                    ckpt.n_committed,
                    n_skipped,
                    len(filtered_data),
                )

        return ckpt, requested_paths, requested_meta_keys, filtered_data, cur_params

    def _finalize_processing(
        self,
        checkpoint_path: str | None,
        ckpt: CheckpointManager | None,
        requested_paths: set | None,
        requested_meta_keys: set | None,
        n_expected: int,
        all_requested_data: list[dict[str, Any]] | None = None,
        *,
        n_pre_existing: int = 0,
        n_load_failures: int = 0,
    ) -> None:
        """Finalize processing and handle checkpoint commit."""
        use_ckpt = checkpoint_path is not None

        if use_ckpt and ckpt is not None:
            ckpt.commit_embeddings()
            n_new_committed = ckpt.n_committed - n_pre_existing
            if n_load_failures > 0 or n_new_committed < n_expected:
                logger.error(
                    "Checkpoint run incomplete: %d image(s) failed to load, "
                    "%d/%d new rows committed.",
                    n_load_failures,
                    n_new_committed,
                    n_expected,
                )
                raise RuntimeError(
                    f"Checkpoint processing incomplete: {n_new_committed} of {n_expected} "
                    f"requested images were committed"
                    + (f" ({n_load_failures} failed to load)" if n_load_failures > 0 else "")
                    + "."
                )
            self._setup_lazy_results(
                ckpt,
                requested_paths=requested_paths,
                requested_meta_keys=requested_meta_keys,
                current_requested_data=all_requested_data,
            )
        else:
            n_actual = self.results.n_images
            if n_actual < n_expected:
                logger.warning(
                    "Some images failed to load: %d processed, %d expected.",
                    n_actual,
                    n_expected,
                )
            self._finalize_results()

    def _setup_lazy_results(
        self,
        ckpt: CheckpointManager,
        requested_paths: set | None = None,
        requested_meta_keys: set | None = None,
        current_requested_data: list[dict[str, Any]] | None = None,
    ) -> None:
        """Set up HDF5-backed lazy results after a checkpoint write.

        Loads paths, metadata, and properties into ``self.results``.
        Embeddings remain on disk; ``results.embeddings`` stays None.
        ``self._db`` holds the open file handle for lazy reads.

        When ``current_requested_data`` is provided (e.g. from the current run's
        file_df), checkpoint metadata is merged with it so that extra columns
        in the new file_df are not lost when the checkpoint had fewer metadata
        keys.

        Args:
            ckpt: Open CheckpointManager.
            requested_paths: Optional set of paths to filter for.
            requested_meta_keys: Optional set of stable metadata keys to filter for.
            current_requested_data: Optional list of {file_path, metadata} from the
                current run's file_df; used to enrich checkpoint metadata with new keys.
        """
        if self._db is not None and self._db is not ckpt:
            try:
                self._db.close()
            except OSError as e:
                logger.debug("Error closing previous checkpoint: %s", e)
        self._db = ckpt

        paths, metadata = ckpt.load_metadata_and_paths()
        if current_requested_data:
            metadata = merge_metadata_from_current_run(
                paths, metadata, current_requested_data, _path_repr
            )
        properties = ckpt.load_properties_all()
        internal = ckpt.load_internal_all()

        h5_indices = np.arange(len(paths), dtype=np.int64)

        if (requested_paths is not None or requested_meta_keys is not None) and paths:
            if requested_meta_keys is not None:
                indices = []
                key_failures = 0
                for i, meta in enumerate(metadata):
                    try:
                        key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
                        if key in requested_meta_keys:
                            indices.append(i)
                    except ValueError:
                        key_failures += 1
                if key_failures:
                    logger.warning(
                        "%d checkpoint row(s) could not be keyed for metadata filter; "
                        "those rows were excluded.",
                        key_failures,
                    )
            else:
                if current_requested_data:
                    current_path_lookup = _build_unique_path_lookup(
                        [d.get("file_path", "") for d in current_requested_data]
                    )
                    indices = [
                        i
                        for i, p in enumerate(paths)
                        if _lookup_unique_path_index(current_path_lookup, p) is not None
                    ]
                else:
                    indices = [
                        i
                        for i, p in enumerate(paths)
                        if _path_repr(p) in (requested_paths or set())
                    ]

            if len(indices) != len(paths):
                paths = [paths[i] for i in indices]
                metadata = [metadata[i] for i in indices]
                h5_indices = h5_indices[indices]
                if properties:
                    properties = [properties[i] for i in indices if i < len(properties)]
                if internal:
                    internal = [internal[i] for i in indices if i < len(internal)]

        # When file_df was provided, replace checkpoint paths with file_df paths
        # so results.img_path points to valid files on this machine.
        if current_requested_data:
            paths = rebase_paths_from_current_run(
                paths, metadata, current_requested_data, _path_repr
            )

        self._db_indices = h5_indices
        self._temporal_start_idx = len(paths)
        self._temporal_embeddings = None

        if len(properties) < len(paths):
            properties = properties + [{} for _ in range(len(paths) - len(properties))]
        if internal and len(internal) < len(paths):
            internal = internal + [{} for _ in range(len(paths) - len(internal))]
        self._ram_internal = internal

        self.results = PhenoMeResults(
            img_path=paths,
            metadata=metadata,
            properties=properties,
            embeddings=None,  # lazy: access via get_embeddings()
        )
        validate_results(self.results)
        logger.info(
            "Pipeline ready: %d images available (lazy loading from %s).",
            len(paths),
            ckpt.path,
        )

    def _finalize_results(self) -> None:
        """Finalise: move _emb_buffer into results.embeddings as a stacked array."""
        if self._emb_buffer:
            self.results.embeddings = np.stack(self._emb_buffer)
            self._emb_buffer = []

    def _get_embedding_data(
        self, method_name: str = "embedding"
    ) -> tuple[np.ndarray | None, str | None]:
        """Get all embeddings, loading lazily from HDF5 when a database is active."""
        emb = self.get_embeddings()
        if emb is None or len(emb) == 0:
            logger.warning(
                "No embedding data available for %s. Run process_images() first.",
                method_name,
            )
            return None, None
        return emb, "Embeddings"

    def _require_file_df(self, method_name: str = "method") -> pd.DataFrame:
        """Return the internally stored file_df or raise if unavailable.

        Used by methods that need access to the dataset description without
        exposing file_df as a public parameter.
        """
        if self._file_df is None or self._file_df.empty:
            logger.warning(
                "Run find_files() first before calling %s.",
                method_name,
            )
            raise RuntimeError(
                "file_df not available. Run find_files() first before calling this method."
            )
        return self._file_df
