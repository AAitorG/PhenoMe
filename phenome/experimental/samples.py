"""Checks and batching for experimental non-image runs.

The image dataset, channel split, and ``extract_batch`` are not used here.
A batch tensor is passed to ``ModelWrapper.extract_embeddings`` as produced.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .._logging import get_logger
from ..core.pipeline_results import PhenoMeResults
from ..io import CheckpointManager
from ..utils.model_wrapper import ModelWrapper
from ..utils.progress import ProgressCallback, report_progress

logger = get_logger(__name__)

LoadFn = Callable[[str], Any]
ShowFn = Callable[[Any], np.ndarray]
PropertyFn = Callable[[Any], dict[str, Any]]
CollateFn = Callable[[Sequence[Any]], torch.Tensor]

_SAMPLE_PARAMS: dict[str, Any] = {
    "channel_mode": "combined",
    "channels": None,
    "resize_size": None,
    "pad_size": None,
    "force_rgb": False,
    "l2_normalize_channels": False,
    "sample_mode": True,
}


class SampleSession(Protocol):
    """Session attributes the experimental data path reads and writes."""

    results: PhenoMeResults
    _emb_buffer: list[np.ndarray]
    _image_mode: bool
    _sample_mode: bool
    _processing_params: dict[str, Any] | None
    _sample_load_fn: Callable[[str], Any] | None
    _sample_show_fn: Callable[[Any], Any] | None

    def _require_file_df(self, method_name: str = ...) -> pd.DataFrame:
        """Return the file catalog."""
        ...

    def _prepare_filtered_data(self, fdf: pd.DataFrame) -> list[dict[str, Any]]:
        """Turn catalog rows into path and metadata dicts."""
        ...

    def _record_step(
        self,
        method: str,
        kwargs: dict[str, Any] | None = ...,
        extras: dict[str, Any] | None = ...,
    ) -> None:
        """Append one run-log step."""
        ...

    def _setup_lazy_results(
        self,
        ckpt: CheckpointManager,
        requested_paths: set[Any] | None = ...,
        requested_meta_keys: set[Any] | None = ...,
        current_requested_data: list[dict[str, Any]] | None = ...,
        prior_properties: dict[str, dict[str, Any]] | None = ...,
    ) -> None:
        """Point the session at an open checkpoint."""
        ...


def stack_samples(samples: Sequence[Any]) -> torch.Tensor:
    """Stack equal-shaped arrays or tensors on a new batch axis.

    Args:
        samples: One loaded sample per row. Each entry must be a ``numpy.ndarray``
            or ``torch.Tensor``, and every entry must share one shape.

    Returns:
        torch.Tensor: Shape ``(B, *sample_shape)`` on CPU.

    Raises:
        TypeError: If a sample is not an array or tensor.
        ValueError: If the batch is empty or the shapes differ.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    arrays: list[torch.Tensor] = []
    for sample in samples:
        if isinstance(sample, torch.Tensor):
            arrays.append(sample.detach().cpu())
        elif isinstance(sample, np.ndarray):
            arrays.append(torch.from_numpy(np.ascontiguousarray(sample)))
        else:
            raise TypeError(
                "Default collate stacks tensors and arrays of equal shape. "
                "Pass collate_fn for other sample types or unequal lengths. "
                f"Got {type(sample).__name__}."
            )
    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) != 1:
        shown = ", ".join(str(shape) for shape in sorted(shapes))
        raise ValueError(f"Samples have shapes {shown}. Pass collate_fn to batch unequal lengths.")
    return torch.stack(arrays, dim=0)


def _own_sample(sample: Any) -> Any:
    """Copy an array or tensor before the next file is loaded.

    Some loaders reuse one buffer. ``process_data`` would otherwise keep that
    same object for every row, and every embedding would match the last file.

    Args:
        sample: Value returned by ``load_fn``.

    Returns:
        A detached CPU clone for a tensor, a new array for an ``ndarray``,
        or the same object when the sample is not an array or tensor.
    """
    if isinstance(sample, torch.Tensor):
        return sample.detach().clone().cpu()
    if isinstance(sample, np.ndarray):
        return sample.copy()
    return sample


def coerce_property_dict(
    raw: Any,
    expected_keys: tuple[str, ...] | None,
) -> tuple[dict[str, float | int], tuple[str, ...]]:
    """Require a flat scalar dict and the same keys on every row.

    Args:
        raw: Value returned by the user's property function.
        expected_keys: Keys from the first row, or None for that first row.

    Returns:
        Tuple of the coerced dict and the key order to require on later rows.

    Raises:
        TypeError: If ``raw`` is not a dict or a value is not a scalar number.
        ValueError: If the keys differ from ``expected_keys``.
    """
    if not isinstance(raw, dict):
        raise TypeError(
            f"property_fn must return a dict of scalar numbers, got {type(raw).__name__}."
        )
    coerced: dict[str, float | int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise TypeError(f"Property names must be strings, got {type(key).__name__}.")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Property {key!r} is a bool. Use 0 or 1.")
        if isinstance(value, (np.floating, float)):
            coerced[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            coerced[key] = int(value)
        else:
            raise TypeError(f"Property {key!r} must be a float or int, got {type(value).__name__}.")
    keys = tuple(coerced)
    if expected_keys is None:
        return coerced, keys
    if set(keys) != set(expected_keys):
        raise ValueError(
            "property_fn must return the same keys for every sample. "
            f"Expected {list(expected_keys)}, got {list(keys)}."
        )
    ordered = {key: coerced[key] for key in expected_keys}
    return ordered, expected_keys


def validate_show_array(raw: Any) -> np.ndarray:
    """Require a 2D or 3D display array from ``show_fn``.

    Args:
        raw: Value returned by the user's show function.

    Returns:
        np.ndarray: The same data, with shape ``(H, W)`` or ``(H, W, C)``.

    Raises:
        TypeError: If ``raw`` is not array-like.
        ValueError: If the shape is not 2D or 3D with a channel axis.
    """
    array = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] >= 1:
        return array
    raise ValueError(
        f"show_fn must return an array of shape (H, W) or (H, W, C), got shape {array.shape}."
    )


def load_sample_display(load_fn: LoadFn, show_fn: ShowFn, path: Any) -> np.ndarray:
    """Load one catalog row and turn it into the array viewers draw.

    Args:
        load_fn: ``load_fn(path) -> sample`` stored on the session.
        show_fn: ``show_fn(sample) -> array`` stored on the session.
        path: Path string from ``results.img_path``.

    Returns:
        np.ndarray: Display image of shape ``(H, W)`` or ``(H, W, C)``.

    Raises:
        TypeError: If ``path`` is not a single file path.
        ValueError: If ``show_fn`` does not return a display array.
    """
    if isinstance(path, (list, tuple)):
        raise TypeError("set_data_io stores one file path per row.")
    sample = load_fn(str(path))
    return validate_show_array(show_fn(sample))


def embed_batch(
    model_wrapper: ModelWrapper,
    samples: Sequence[Any],
    collate_fn: CollateFn | None,
) -> np.ndarray:
    """Run one collated batch through the wrapper.

    The tensor is not inspected for channel-split layout. A 5D batch stays 5D.

    Args:
        model_wrapper: Wrapper whose ``_get_embeddings`` returns ``(B, D)``.
        samples: Loaded samples in this batch.
        collate_fn: User collate, or None to stack equal-shaped arrays.

    Returns:
        np.ndarray: Float32 embeddings of shape ``(B, D)``.

    Raises:
        TypeError: If collate does not return a tensor.
        ValueError: If the embedding row count does not match the batch.
    """
    tensor = stack_samples(samples) if collate_fn is None else collate_fn(samples)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"collate_fn must return a torch.Tensor, got {type(tensor).__name__}.")
    tensor = tensor.to(model_wrapper.device)
    embeddings = model_wrapper.extract_embeddings(tensor).float().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(samples):
        raise ValueError(
            "Model embeddings must have shape (B, D) with one row per sample. "
            f"Got {embeddings.shape} for a batch of {len(samples)}."
        )
    return embeddings


def bind_data_io(
    session: SampleSession,
    load_fn: Callable[[str], Any],
    show_fn: Callable[[Any], Any],
) -> None:
    """Store the loader and show function for a non-image session.

    Args:
        session: PhenoMe session.
        load_fn: ``load_fn(path) -> sample``.
        show_fn: ``show_fn(sample) -> array`` of shape ``(H, W)`` or ``(H, W, C)``.

    Raises:
        RuntimeError: If this session already ran ``process_images``.
        TypeError: If either callable is missing.
    """
    if session._image_mode:
        raise RuntimeError(
            "This session already ran process_images(). "
            "Start a new PhenoMe() to call set_data_io()."
        )
    _require_callable("load_fn", load_fn)
    _require_callable("show_fn", show_fn)
    session._sample_load_fn = load_fn
    session._sample_show_fn = show_fn
    session._sample_mode = True
    session._record_step("set_data_io", {"experimental": True})


def run_process_data(
    session: SampleSession,
    model_wrapper: ModelWrapper,
    *,
    collate_fn: CollateFn | None = None,
    batch_size: int = 32,
    checkpoint_path: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Load each catalog row and store embeddings of shape ``(N, D)``.

    Each array or tensor from the loader is copied before the next file is
    read, and only the current batch is kept. Properties are left for
    ``compute_properties``.

    Args:
        session: PhenoMe session that already has a file catalog and ``set_data_io``.
        model_wrapper: Wrapper that maps the collated batch to ``(B, D)``.
        collate_fn: Optional batch function. None stacks equal-shaped arrays.
        batch_size: Rows per forward pass.
        checkpoint_path: New HDF5 path, or None to keep embeddings in memory.
        progress_callback: Optional ``(current, total, desc) -> None``.

    Returns:
        None. Writes ``results.embeddings``. With ``checkpoint_path``, embeddings
        stay on disk and ``get_embeddings()`` reads them.

    Raises:
        RuntimeError: If ``set_data_io`` was not called, or this session already
            ran ``process_images``.
        TypeError: If ``model_wrapper`` is not a ``ModelWrapper``, or a row path
            is not a single file.
        ValueError: If ``batch_size`` is below 1 or the model output is not ``(B, D)``.
        FileExistsError: If ``checkpoint_path`` already exists.
    """
    if session._image_mode:
        raise RuntimeError(
            "This session already ran process_images(). "
            "Start a new PhenoMe() to call process_data()."
        )
    load_fn = session._sample_load_fn
    show_fn = session._sample_show_fn
    if load_fn is None or show_fn is None or not session._sample_mode:
        raise RuntimeError("Call set_data_io() before process_data().")
    file_df = session._require_file_df("process_data")
    _require_callable("model_wrapper", model_wrapper, wrapper=True)
    if collate_fn is not None and not callable(collate_fn):
        raise TypeError("collate_fn must be a callable.")
    if not isinstance(model_wrapper, ModelWrapper):
        raise TypeError("model_wrapper must be a ModelWrapper.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        raise FileExistsError(
            f"Checkpoint already exists at {checkpoint_path}. "
            "process_data does not resume an existing file."
        )

    items = session._prepare_filtered_data(file_df)
    logger.info(
        "Experimental data run: embeddings use the loader from set_data_io(). "
        "The image pipeline is not used."
    )
    session._record_step(
        "process_data",
        {
            "batch_size": batch_size,
            "checkpoint_path": checkpoint_path,
            "experimental": True,
        },
        extras={"n_files": len(items)},
    )

    parts: list[np.ndarray] = []
    total = len(items)
    desc = "Processing samples"
    disable_tqdm = progress_callback is not None
    n_batches = max(1, (total + batch_size - 1) // batch_size) if total else 1
    checked_show = False
    for batch_index, start in enumerate(
        tqdm(
            range(0, total, batch_size),
            desc=desc,
            disable=disable_tqdm or total == 0,
        ),
        start=1,
    ):
        batch: list[Any] = []
        for item in items[start : start + batch_size]:
            path = item["file_path"]
            if not isinstance(path, str):
                raise TypeError(
                    f"process_data expects one file path per row. Got {type(path).__name__}."
                )
            sample = _own_sample(load_fn(path))
            if not checked_show:
                validate_show_array(show_fn(sample))
                checked_show = True
            batch.append(sample)
        parts.append(embed_batch(model_wrapper, batch, collate_fn))
        report_progress(progress_callback, batch_index, n_batches, desc)

    embeddings = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)
    session._processing_params = dict(_SAMPLE_PARAMS)
    if checkpoint_path is None:
        session.results.embeddings = embeddings
        session._emb_buffer = []
        return

    ckpt = CheckpointManager(
        checkpoint_path,
        embedding_dim=int(embeddings.shape[1]) if embeddings.size else 0,
        processing_params=dict(_SAMPLE_PARAMS),
        lazy=True,
    )
    if embeddings.size:
        paths = [item["file_path"] for item in items]
        metas = [dict(item.get("metadata") or {}) for item in items]
        ckpt.buffer_embeddings(embeddings, paths, metas)
        ckpt.commit_embeddings()
    session._setup_lazy_results(ckpt, current_requested_data=items)


def compute_loaded_properties(
    load_fn: LoadFn,
    paths: Sequence[Any],
    functions: Sequence[Callable[..., dict[str, Any]]],
) -> list[dict[str, float | int]]:
    """Load each path and merge property dicts from ``fn(sample, None)``.

    Args:
        load_fn: ``load_fn(path) -> sample`` from ``set_data_io``.
        paths: Catalog paths, one string per row.
        functions: Callables with the usual ``(sample, mask)`` signature.
            ``mask`` is ``None``.

    Returns:
        One flat scalar dict per row. Every row has the same keys.

    Raises:
        TypeError: If a path is not a single file or a function does not return a dict.
        ValueError: If the keys differ across rows.
    """
    rows: list[dict[str, float | int]] = []
    expected: tuple[str, ...] | None = None
    for path in paths:
        if not isinstance(path, str):
            raise TypeError(
                f"compute_properties expects one file path per row. Got {type(path).__name__}."
            )
        sample = load_fn(path)
        merged: dict[str, Any] = {}
        for fn in functions:
            out = fn(sample, None)
            if not isinstance(out, dict):
                raise TypeError(
                    "A property function must return a dict of scalar numbers, "
                    f"got {type(out).__name__}."
                )
            merged.update(out)
        coerced, expected = coerce_property_dict(merged, expected)
        rows.append(coerced)
    return rows


def _require_callable(name: str, fn: Any, *, wrapper: bool = False) -> None:
    """Raise TypeError when a required experimental argument was omitted.

    Args:
        name: Argument name used in the error.
        fn: Value the caller passed.
        wrapper: When True, a ``ModelWrapper`` instance is accepted even
            though the instance itself may not be the callable.

    Raises:
        TypeError: If ``fn`` is None, or not callable and not a wrapper.
    """
    if fn is None or (not wrapper and not callable(fn)):
        raise TypeError(f"{name} is required.")
