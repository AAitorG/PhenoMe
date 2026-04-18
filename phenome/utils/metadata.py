"""
Metadata extraction for the phenotyping pipeline.

Provides path-template and dataframe-based helpers to build metadata functions
compatible with :class:`phenome.core.MetadataFn`. The default column
name for file identifiers is ``filename`` across all helpers.

- :func:`default_metadata_from_path`: Minimal extractor returning ``file_path`` and ``filename``.
- :func:`get_metadata_from_path`: Build extractor from a path template. Adds ``filename`` key.
- :func:`make_dataframe_metadata_fn`: Look up metadata from a DataFrame. Supports
  a single filename column (default ``"filename"``) or a list of columns for
  multi-channel data. Accepts filenames with or without extension; only the last
  dot is treated as the extension.

These functions are thin wrappers over :class:`phenome.metadata` classes.
For OOP usage with configurable columns and auto-generated IDs, use
:class:`DefaultMetadata`, :class:`PathTemplateMetadata`, or :class:`DataFrameMetadata`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..metadata import DataFrameMetadata, DefaultMetadata, PathTemplateMetadata


def default_metadata_from_path(path: str) -> dict[str, Any]:
    """@section Metadata helpers
    @order 35

    Simple, dataset-agnostic metadata extractor used by default.

    Returns 'file_path' (full file path) and 'filename' (basename stem, no extension).
    For richer metadata (e.g., 'drug', 'time', 'plate', 'well'), define a
    dataset-specific function and pass it as the metadata_fn argument to
    PhenoMe.

    Helpers for path templates and dataframe lookup are available in
    phenome.utils.metadata. The default column name for file
    identifiers is ``"filename"`` across all metadata helpers.

    Args:
        path: File path.

    Returns:
        Dictionary with 'file_path', 'filename', and 'id' keys.

    Example:
        >>> meta = default_metadata_from_path('/data/image.tif')
        >>> meta['file_path'], meta['filename']
        ('/data/image.tif', 'image')
    """
    return DefaultMetadata().metadata_fn(path, data_dir=None)


def get_metadata_from_path(template: str) -> Callable[[str], dict[str, Any]]:
    """@section Metadata helpers
    @order 36

    Create metadata extractor function from a path template.

    Features:
    - Use '...' at the start to indicate the template matches a suffix of the path.
    - Use parentheses for capture groups: (field_name).
    - Use '.*' for matching any file extension: (filename).*
    - Supports both Unix (/) and Windows (\\) separators in the template.

    Args:
        template: Path template with capture groups in parentheses.

    Returns:
        Callable[[str], Dict[str, Any]]: Function (path) -> dict of captured group names to values.
            Returns {} if no match. Keys from template placeholders plus ``filename`` (the
            default column name for file identifiers). The returned function has a
            ``group_by`` attribute (last capture group, or ``"filename"``) for
            :meth:`find_files` multi-channel grouping.

    Example:
        >>> extractor = get_metadata_from_path(".../(drug)/(time)/(crop_name).tif")
        >>> meta = extractor("/data/project/DrugA/24h/crop_01.tif")
        >>> meta['drug'], meta['crop_name'], meta['filename']
        ('DrugA', 'crop_01', 'crop_01')

        >>> # Match any extension
        >>> extractor = get_metadata_from_path(".../(plate)/(well).*")
        >>> extractor("/data/P1/A01.tif")['filename']
        'A01'
    """
    meta_src = PathTemplateMetadata(template=template)

    def extractor(path: str) -> dict[str, Any]:
        return meta_src.metadata_fn(path, data_dir=None)

    extractor.group_by = meta_src.group_by  # type: ignore[attr-defined]
    extractor._metadata_source = meta_src  # type: ignore[attr-defined]
    return extractor


def make_dataframe_metadata_fn(
    metadata_df: pd.DataFrame,
    filename_column: str | list[str] = "filename",
) -> Callable[[str], dict[str, Any]]:
    """@section Metadata helpers
    @order 37

    Build metadata function using a dataframe for metadata lookup.

    Supports two modes:

    1. **Single file per sample** (default): Pass a single column name as
       `filename_column`. The dataframe must contain that column with image
       filenames; both stems (e.g. ``'my_image'``) and full names with
       extension (e.g. ``'my_image.tif'``) are accepted. Each row is one image.

    2. **Multiple channels in separate files**: Pass a list of column names
       in channel order, e.g. ``['ch0', 'ch1', 'ch2']``. Each column holds
       the filename (with or without extension) for that channel. Each row is
       one sample; the same row is matched when any of its channel filenames
       is seen. The returned metadata includes ``channel_index`` (0, 1, …).
       :meth:`find_files` groups paths by the first filename column (e.g.
       ``ch0``) into one row per sample with ``file_path`` as a list of paths
       in channel order.

    All other columns are returned with their column names as keys. The
    function adds ``file_path`` (current path) and, in multi-channel mode,
    ``channel_index``. No separate ``img_name`` column is added when the
    filename column already identifies the image.

    Args:
        metadata_df: DataFrame with at least the column(s) specified by
            filename_column.
        filename_column: Column name(s) for filenames. Single string (default
            ``"filename"``) for one image per row, or list of column names for
            multi-channel. Accepts filenames with or without extension. Only
            the last dot is treated as the extension; internal dots (e.g. in
            ``'plate.A01.well'``) are preserved.

    Returns:
        Callable[[str], Dict[str, Any]]: Function (path) -> dict. Adds
        ``file_path`` and, in multi-channel mode, ``channel_index``. All
        dataframe columns are included (filename column identifies the image).
        The returned function has a ``group_by`` attribute (first filename column)
        for :meth:`find_files` multi-channel grouping.

    Raises:
        ValueError: If filename_column (or any of its elements) is not in
            metadata_df.

    Example (single file):
        >>> df = pd.DataFrame({
        ...     'filename': ['img1', 'img2'],
        ...     'condition': ['Control', 'Treatment'],
        ...     'time': ['24h', '48h']
        ... })
        >>> metadata_fn = make_dataframe_metadata_fn(df, filename_column='filename')
        >>> meta = metadata_fn('/data/img1.tif')
        >>> sorted(meta.items())  # doctest: +NORMALIZE_WHITESPACE
        [('condition', 'Control'), ('file_path', '/data/img1.tif'), ('filename', 'img1'),
         ('id', 'img1'), ('time', '24h')]

    Example (single file, df with extension - both formats accepted):
        >>> df_ext = pd.DataFrame({'filename': ['img1.tif', 'img2.tif'], 'cond': ['A', 'B']})
        >>> fn_ext = make_dataframe_metadata_fn(df_ext, filename_column='filename')
        >>> fn_ext('/data/img1.tif')['cond']
        'A'

    Example (multi-channel, one column per channel):
        >>> df = pd.DataFrame({
        ...     'ch0': ['sample1_c0', 'sample2_c0'],
        ...     'ch1': ['sample1_c1', 'sample2_c1'],
        ...     'ch2': ['sample1_c2', 'sample2_c2'],
        ...     'condition': ['Control', 'Treatment'],
        ... })
        >>> metadata_fn = make_dataframe_metadata_fn(df, filename_column=['ch0', 'ch1', 'ch2'])
        >>> meta = metadata_fn('/data/sample1_c1.tif')
        >>> meta['condition'], meta['channel_index'], meta['ch0']
        ('Control', 1, 'sample1_c0')
    """
    meta_src = DataFrameMetadata(
        metadata_df=metadata_df,
        filename_columns=filename_column,
    )

    def metadata_fn(path: str) -> dict[str, Any]:
        return meta_src.metadata_fn(path, data_dir=None)

    metadata_fn.group_by = meta_src.group_by  # type: ignore[attr-defined]
    metadata_fn._metadata_source = meta_src  # type: ignore[attr-defined]
    return metadata_fn
