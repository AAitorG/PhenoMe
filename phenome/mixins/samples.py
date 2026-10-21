"""Experimental embedding for data that is not a 2D image."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch

from ..experimental.samples import SampleSession, bind_data_io, run_process_data
from ..utils.model_wrapper import ModelWrapper
from ..utils.progress import ProgressCallback


class PhenoMeSamples:
    """Experimental entry for non-image files.

    ``set_data_io`` stores the loader and the show function. ``process_data``
    writes embeddings of shape ``(N, D)``. Properties stay on
    ``compute_properties``.
    """

    def set_data_io(
        self,
        load_fn: Callable[[str], Any],
        show_fn: Callable[[Any], Any],
    ) -> None:
        """Store how to read a file and how to draw one sample. Experimental.

        Call this before ``process_data``, ``compute_properties``, and
        ``plot_data_by_index``. Plots, the explorer, and the report use the
        stored functions.

        Args:
            load_fn: ``load_fn(path) -> sample``. ``path`` is one file path.
            show_fn: ``show_fn(sample) -> array`` of shape ``(H, W)`` or
                ``(H, W, C)``.

        Returns:
            None.

        Raises:
            RuntimeError: If this session already ran ``process_images``.
            TypeError: If either argument is missing or not callable.
        """
        bind_data_io(cast(SampleSession, self), load_fn, show_fn)

    def process_data(
        self,
        model_wrapper: ModelWrapper,
        *,
        collate_fn: Callable[[Sequence[Any]], torch.Tensor] | None = None,
        batch_size: int = 32,
        checkpoint_path: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Extract embeddings from non-image files. Experimental.

        The image loader, channel split, resize, and ImageNet normalization
        are not used. The loader from ``set_data_io`` returns one sample.
        ``collate_fn`` turns a list of samples into the tensor
        ``model_wrapper`` receives. When ``collate_fn`` is omitted,
        equal-shaped arrays are stacked. Each array or tensor from the loader
        is copied before the next file is read, so a reused buffer does not
        overwrite earlier rows. The model must return ``(B, D)``.
        A 5D batch is passed through as produced.

        Call ``find_files`` or ``set_file_df`` first, then ``set_data_io``.
        Pass ``extensions`` so discovery keeps the files you intend to load.
        Compute properties afterwards with ``compute_properties``.

        Args:
            model_wrapper: ``ModelWrapper`` subclass. ``_get_embeddings``
                receives the collated batch and returns ``(B, D)``.
            collate_fn: ``collate_fn(samples) -> Tensor``. Default stacks
                equal-shaped arrays. Required when samples differ in length
                or are not arrays.
            batch_size: Rows per forward pass. Default 32.
            checkpoint_path: HDF5 path for the embedding dataset. The file
                must not already exist. Image config fields are stored empty
                and ``sample_mode`` is true. Default None (embeddings stay
                in memory).
            progress_callback: Optional ``(current, total, desc) -> None``.
                When set, the ``tqdm`` bar is disabled.

        Returns:
            None. Writes ``results.embeddings`` with shape ``(N, D)``. With
            ``checkpoint_path``, embeddings stay on disk and
            ``get_embeddings()`` reads them.

        Raises:
            RuntimeError: If ``set_data_io`` was not called, the file catalog
                is missing, or this session already ran ``process_images``.
            TypeError: If ``model_wrapper`` is not a ``ModelWrapper``, or a
                row path is not a single file.
            ValueError: If ``batch_size`` is below 1 or the model output is
                not ``(B, D)``.
            FileExistsError: If ``checkpoint_path`` already exists.

        Example:
            >>> def load_volume(path):
            ...     return np.load(path)
            >>> pm.set_data_io(load_volume, orthogonal_projections)
            >>> pm.process_data(wrapper, batch_size=4)
        """
        run_process_data(
            cast(SampleSession, self),
            model_wrapper,
            collate_fn=collate_fn,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            progress_callback=progress_callback,
        )
