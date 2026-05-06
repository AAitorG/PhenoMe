"""Internal Implementation: Helper functions for pipeline mixins.

Consolidates utility functions shared across analysis, properties,
and visualization mixins to avoid fragmentation and code duplication.
These helpers focus on data conversion and common mathematical operations.
"""

import numpy as np


def numpy_for_torch(x: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, writable array for :func:`torch.from_numpy`.

    Embeddings and other pipeline arrays may be read-only views (e.g. HDF5, slices).
    Row subsets ``data[indices]`` can also be read-only views. PyTorch requires
    a writable buffer when sharing with NumPy.
    """
    x = np.ascontiguousarray(x)
    if not x.flags.writeable:
        x = x.copy()
    return x


def index_array_for_torch(a: object) -> np.ndarray:
    """Writable integer array for ``tensor[indices]`` and related NumPy index ops.

    PyTorch may call :func:`torch.from_numpy` on advanced-index arrays. Pandas
    :meth:`~pandas.Index.to_numpy` often returns a read-only view, which triggers
    the same non-writable warning.
    """
    return np.array(a, dtype=np.intp, copy=True, order="C")


def map_to_full(values: np.ndarray, valid_indices: list[int], n_total: int) -> np.ndarray:
    """Map values for valid_indices into a full-length NaN array.

    Args:
        values: np.ndarray shape (len(valid_indices),).
        valid_indices: List[int]. Global indices where values apply.
        n_total: Total number of images.

    Returns:
        np.ndarray shape (n_total,), dtype float32. NaN where not in valid_indices.
    """
    full = np.full(n_total, np.nan, dtype=np.float32)
    full[np.asarray(valid_indices, dtype=np.int64)] = values.astype(np.float32)
    return full


# Supported metrics for ordering rows in the embedding-property summary.
ORDER_METRICS: tuple[str, ...] = ("mean_abs", "max_abs", "mean", "std")
