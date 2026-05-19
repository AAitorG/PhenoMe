"""
Batch / plate correction for embeddings or scalar properties (after process_images).

Default method is control-based **sphering** per batch. Checkpoint-backed runs apply
corrections in chunks without loading all embeddings into RAM.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from .._logging import get_logger
from ..core.batch_correction import (
    apply_batch_correction_rows,
    build_property_fetch_fn,
    compute_batch_stats,
    row_batch_ids_from_metadata,
)

logger = get_logger(__name__)

MethodName = Literal["sphering", "zscore"]
SourceName = Literal["embeddings", "properties"]


class PhenoMeBatchCorrection:
    """
    @section Analysis
    @order 7

    Mixin: ``correct_batches()`` for plate / batch effects.

    Expects ``self.results``, ``self.get_embeddings``, ``self._db``,
    ``self._db_indices``, ``self._temporal_start_idx``, ``self._temporal_embeddings``,
    ``self.has_embeddings``, ``self.get_available_property_keys`` (for properties).
    """

    results: Any
    _db: Any
    _db_indices: np.ndarray | None
    _temporal_start_idx: int
    _temporal_embeddings: np.ndarray | None
    _property_norm_cache: dict[str, Any] | None

    @property
    def has_embeddings(self) -> bool:
        """Provided by the concrete pipeline; stub for type checkers."""
        raise NotImplementedError

    def get_embeddings(
        self,
        indices: int | list[int] | np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Provided by the concrete pipeline; stub for type checkers."""
        raise NotImplementedError

    def save_results(self, path: str | None = None, compression: str = "gzip") -> None:
        """Provided by the concrete pipeline; stub for type checkers."""
        raise NotImplementedError

    def correct_batches(
        self,
        batch_metadata_key: str,
        method: MethodName = "sphering",
        source: SourceName = "embeddings",
        property_keys: list[str] | None = None,
        control_filters: dict[str, Any] | None = None,
        inplace: bool = True,
        checkpoint_path: str | None = None,
        chunk_size: int = 4096,
        force: bool = False,
        ridge_multiplier: float = 1e-3,
        min_controls: int = 2,
    ) -> dict[str, Any]:
        """Correct plate-to-plate variation using control wells per batch.

        Call after :meth:`process_images` (and optionally after ``compute_properties`` when
        ``source='properties'``). For lazy checkpoints, embeddings are read and rewritten
        in chunks of *chunk_size* rows.

        Args:
            batch_metadata_key: Metadata column for batch / plate id (case-insensitive).
            method: ``\"sphering\"`` (default) or ``\"zscore\"``.
            source: ``\"embeddings\"`` or ``\"properties\"``.
            property_keys: Property names to correct when ``source='properties'``.
                If None, uses all keys from :meth:`get_available_property_keys`.
            control_filters: Same semantics as ``filter_indices`` filters (required).
            inplace: If True, updates ``results.embeddings`` or ``results.properties`` in place.
            checkpoint_path: If set, persist batch-correction metadata and/or export results.
                - If ``None`` (default): Corrections still apply (RAM or an already-open lazy
                  checkpoint), but nothing new is written for persistence—no ``/batch_correction``
                  stats on disk and no new HDF5 via :meth:`save_results`.
                - If ``str``: With an open checkpoint, it is cloned to this path first; stats are
                  saved under ``/batch_correction`` when a DB is active; in eager mode (no DB),
                  results are saved to this path after correction.
            chunk_size: Rows per chunk for embedding I/O and property scatter.
            force: If True, allow re-applying correction after a previous run in this session.
            ridge_multiplier: Covariance ridge strength for sphering (see :func:`compute_batch_stats`).
            min_controls: Minimum control wells required per batch.

        Returns:
            Dict with batch stats summary (``batches``, ``controls_per_batch``, ``method``, …).
        """
        if not force and getattr(self, "_batch_correction_applied", False):
            raise RuntimeError(
                "Batch correction was already applied in this session. "
                "Pass force=True to re-run (not recommended on already-corrected data)."
            )

        if not inplace:
            raise NotImplementedError("inplace=False is not supported; use inplace=True.")

        if checkpoint_path is not None and self._db is not None:
            logger.info("Cloning checkpoint to %s before correction...", checkpoint_path)
            self._db = self._db.clone_to_new_file(checkpoint_path)

        n = int(self.results.n_images)
        if n == 0:
            raise ValueError("No rows in results; run process_images() first.")

        metadata = list(self.results.metadata)
        while len(metadata) < n:
            metadata.append({})

        if source == "embeddings":
            return self._correct_batches_embeddings(
                metadata=metadata,
                n=n,
                batch_metadata_key=batch_metadata_key,
                method=method,
                control_filters=control_filters,
                checkpoint_path=checkpoint_path,
                chunk_size=chunk_size,
                ridge_multiplier=ridge_multiplier,
                min_controls=min_controls,
            )

        return self._correct_batches_properties(
            metadata=metadata,
            n=n,
            batch_metadata_key=batch_metadata_key,
            method=method,
            control_filters=control_filters,
            property_keys=property_keys,
            chunk_size=chunk_size,
            ridge_multiplier=ridge_multiplier,
            min_controls=min_controls,
            checkpoint_path=checkpoint_path,
        )

    def _correct_batches_embeddings(
        self,
        *,
        metadata: list[dict[str, Any]],
        n: int,
        batch_metadata_key: str,
        method: MethodName,
        control_filters: dict[str, Any] | None,
        checkpoint_path: str | None,
        chunk_size: int,
        ridge_multiplier: float,
        min_controls: int,
    ) -> dict[str, Any]:
        if not self.has_embeddings:
            raise ValueError("No embeddings available. Run process_images() first.")

        def fetch_rows(idx: np.ndarray) -> np.ndarray:
            arr = self.get_embeddings(indices=idx)
            if arr is None:
                raise RuntimeError("get_embeddings returned None during batch correction.")
            return arr

        stats, info = compute_batch_stats(
            metadata,
            batch_metadata_key,
            cast(dict[str, Any], control_filters),
            fetch_rows,
            method=method,
            chunk_size=chunk_size,
            ridge_multiplier=ridge_multiplier,
            min_controls=min_controls,
            results_for_filter=self.results,
        )
        row_batch_ids = row_batch_ids_from_metadata(metadata, batch_metadata_key)

        db = self._db
        if db is not None:
            db_indices = self._db_indices
            if db_indices is None:
                raise RuntimeError("Lazy checkpoint is active but _db_indices is None.")
            n_ckpt_logical = int(self._temporal_start_idx)
            for start in range(0, n_ckpt_logical, chunk_size):
                logical = np.arange(start, min(start + chunk_size, n_ckpt_logical), dtype=np.int64)
                emb = fetch_rows(logical)
                bids = [row_batch_ids[int(i)] for i in logical.tolist()]
                corrected = apply_batch_correction_rows(emb, bids, stats, method)
                h5_idx = db_indices[logical]
                db.write_embeddings_rows(h5_idx, corrected)

            temp_emb = self._temporal_embeddings
            if temp_emb is not None and temp_emb.ndim == 2 and temp_emb.shape[0] > 0:
                t0 = int(self._temporal_start_idx)
                for start in range(t0, n, chunk_size):
                    logical = np.arange(start, min(start + chunk_size, n), dtype=np.int64)
                    emb = fetch_rows(logical)
                    bids = [row_batch_ids[int(i)] for i in logical.tolist()]
                    corrected = apply_batch_correction_rows(emb, bids, stats, method)
                    local = (logical - t0).astype(np.int64)
                    self._temporal_embeddings[local] = corrected

            if checkpoint_path is not None:
                db.save_batch_correction_state(
                    stats,
                    method=method,
                    batch_metadata_key=batch_metadata_key,
                    source="embeddings",
                )
                logger.info("Wrote batch correction stats to checkpoint %s", db.path)
        else:
            emb_all = fetch_rows(np.arange(n, dtype=np.int64))
            corrected = apply_batch_correction_rows(emb_all, row_batch_ids, stats, method)
            self.results.embeddings = corrected

        if checkpoint_path is not None and self._db is None:
            self.save_results(checkpoint_path)
            logger.info("Saved corrected results to %s", checkpoint_path)

        self._batch_correction_applied = True
        self._batch_correction_last_info = {**info, "source": "embeddings"}
        self._property_norm_cache = None
        logger.info(
            "Batch correction (%s) applied to embeddings (%d batches).",
            method,
            len(stats),
        )
        return self._batch_correction_last_info

    def _correct_batches_properties(
        self,
        *,
        metadata: list[dict[str, Any]],
        n: int,
        batch_metadata_key: str,
        method: MethodName,
        control_filters: dict[str, Any] | None,
        property_keys: list[str] | None,
        chunk_size: int,
        ridge_multiplier: float,
        min_controls: int,
        checkpoint_path: str | None,
    ) -> dict[str, Any]:
        keys = property_keys
        if keys is None:
            keys = list(self.get_available_property_keys())  # type: ignore[attr-defined]
        if not keys:
            raise ValueError("No property keys to correct; run compute_properties() first.")

        props = self.results.properties
        if len(props) < n:
            raise ValueError("results.properties length does not match n_images.")

        fetch_rows = build_property_fetch_fn(props, keys)
        stats, info = compute_batch_stats(
            metadata,
            batch_metadata_key,
            cast(dict[str, Any], control_filters),
            fetch_rows,
            method=method,
            chunk_size=chunk_size,
            ridge_multiplier=ridge_multiplier,
            min_controls=min_controls,
            results_for_filter=self.results,
        )
        row_batch_ids = row_batch_ids_from_metadata(metadata, batch_metadata_key)

        for start in range(0, n, chunk_size):
            logical = np.arange(start, min(start + chunk_size, n), dtype=np.int64)
            block = fetch_rows(logical)
            bids = [row_batch_ids[int(i)] for i in logical.tolist()]
            corrected = apply_batch_correction_rows(block, bids, stats, method)
            for r, global_i in enumerate(logical.tolist()):
                row = props[global_i]
                if not isinstance(row, dict):
                    continue
                for j, k in enumerate(keys):
                    row[k] = float(corrected[r, j])

        if checkpoint_path is not None and self._db is not None:
            self._db.save_batch_correction_state(
                stats,
                method=method,
                batch_metadata_key=batch_metadata_key,
                source="properties",
            )
            logger.info("Wrote batch correction stats to checkpoint %s", self._db.path)

        if checkpoint_path is not None and self._db is None:
            self.save_results(checkpoint_path)
            logger.info("Saved corrected results to %s", checkpoint_path)

        self._batch_correction_applied = True
        self._batch_correction_last_info = {**info, "source": "properties", "property_keys": keys}
        self._property_norm_cache = None
        logger.info(
            "Batch correction (%s) applied to %d properties (%d batches).",
            method,
            len(keys),
            len(stats),
        )
        return self._batch_correction_last_info
