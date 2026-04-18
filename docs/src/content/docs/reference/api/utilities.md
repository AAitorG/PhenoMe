---
title: "Utility functions"
description: Device helpers, transforms, image I/O, checkpoints, file discovery.
---

Auto-generated from `phenome.utils.device`, `phenome.utils.metadata`, `phenome.io`, and `phenome.utils.transforms`.

**See also:** [Model wrappers](model-wrapper.md) · [HDF5 protocol](../DATABASE_PROTOCOL.md) · [Best practices](../guides/best-practices.md)

## Device and reproducibility — `set_default_device`

```python
set_default_device(
    device: torch.device | str | None
) -> None
```

Set the default device for phenotyping operations.

**Args:**

- **`device`**: Target device (torch.device, string like 'cuda:0', or None to reset).

## Device and reproducibility — `get_default_device`

```python
get_default_device()
```

Return a default device for phenotyping operations.

If set_default_device() was called, returns that device.
Otherwise, uses a single GPU (cuda:0) when available, else CPU.
When CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the first visible GPU.

**Returns:**

- **`torch.device`**: cuda:0 if CUDA is available, else cpu.

## Device and reproducibility — `set_determinism`

```python
set_determinism(
    seed: int
) -> None
```

Set random seeds for reproducibility across all frameworks.

**Args:**

- **`seed`**: Random seed value

## Metadata helpers

Functional wrappers over `phenome.metadata` classes. For OOP extractors, see [Metadata classes](metadata.md).

### `default_metadata_from_path`

```python
default_metadata_from_path(
    path: 'str'
) -> dict[str, Any]
```

Simple, dataset-agnostic metadata extractor used by default.

Returns 'file_path' (full file path) and 'filename' (basename stem, no extension).
For richer metadata (e.g., 'drug', 'time', 'plate', 'well'), define a
dataset-specific function and pass it as the metadata_fn argument to
PhenoMe.

Helpers for path templates and dataframe lookup are available in
phenome.utils.metadata. The default column name for file
identifiers is ``"filename"`` across all metadata helpers.

**Args:**

- **`path`**: File path.

**Returns:**

Dictionary with 'file_path', 'filename', and 'id' keys.

**Example:**

```python
>>> meta = default_metadata_from_path('/data/image.tif')
>>> meta['file_path'], meta['filename']
('/data/image.tif', 'image')

```

### `get_metadata_from_path`

```python
get_metadata_from_path(
    template: 'str'
) -> Callable[[str], dict[str, Any]]
```

Create metadata extractor function from a path template.

Features:
- Use '...' at the start to indicate the template matches a suffix of the path.
- Use parentheses for capture groups: (field_name).
- Use '.*' for matching any file extension: (filename).*
- Supports both Unix (/) and Windows (\) separators in the template.

**Args:**

- **`template`**: Path template with capture groups in parentheses.

**Returns:**

Callable[[str], Dict[str, Any]]: Function (path) -> dict of captured group names to values.
Returns &#123;&#125; if no match. Keys from template placeholders plus ``filename`` (the
default column name for file identifiers). The returned function has a
``group_by`` attribute (last capture group, or ``"filename"``) for
:meth:`find_files` multi-channel grouping.

**Example:**

```python
>>> extractor = get_metadata_from_path(".../(drug)/(time)/(crop_name).tif")
>>> meta = extractor("/data/project/DrugA/24h/crop_01.tif")
>>> meta['drug'], meta['crop_name'], meta['filename']
('DrugA', 'crop_01', 'crop_01')

>>> # Match any extension
>>> extractor = get_metadata_from_path(".../(plate)/(well).*")
>>> extractor("/data/P1/A01.tif")['filename']
'A01'

```

### `make_dataframe_metadata_fn`

```python
make_dataframe_metadata_fn(
    metadata_df: 'pd.DataFrame',
    filename_column: 'str | list[str]' = 'filename'
) -> Callable[[str], dict[str, Any]]
```

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

**Args:**

- **`metadata_df`**: DataFrame with at least the column(s) specified by
  filename_column.
- **`filename_column`**: Column name(s) for filenames. Single string (default
  ``"filename"``) for one image per row, or list of column names for
  multi-channel. Accepts filenames with or without extension. Only
  the last dot is treated as the extension; internal dots (e.g. in
  ``'plate.A01.well'``) are preserved.

**Returns:**

Callable[[str], Dict[str, Any]]: Function (path) -> dict. Adds
``file_path`` and, in multi-channel mode, ``channel_index``. All
dataframe columns are included (filename column identifies the image).
The returned function has a ``group_by`` attribute (first filename column)
for :meth:`find_files` multi-channel grouping.

**Raises:**

- **`ValueError`**: If filename_column (or any of its elements) is not in
  metadata_df.
Example (single file):
    >>> df = pd.DataFrame(&#123;
    ...     'filename': ['img1', 'img2'],
    ...     'condition': ['Control', 'Treatment'],
    ...     'time': ['24h', '48h']
    ... &#125;)
    >>> metadata_fn = make_dataframe_metadata_fn(df, filename_column='filename')
    >>> meta = metadata_fn('/data/img1.tif')
    >>> sorted(meta.items())  # doctest: +NORMALIZE_WHITESPACE
    [('condition', 'Control'), ('file_path', '/data/img1.tif'), ('filename', 'img1'),
     ('id', 'img1'), ('time', '24h')]

Example (single file, df with extension - both formats accepted):
    >>> df_ext = pd.DataFrame(&#123;'filename': ['img1.tif', 'img2.tif'], 'cond': ['A', 'B']&#125;)
    >>> fn_ext = make_dataframe_metadata_fn(df_ext, filename_column='filename')
    >>> fn_ext('/data/img1.tif')['cond']
    'A'

Example (multi-channel, one column per channel):
    >>> df = pd.DataFrame(&#123;
    ...     'ch0': ['sample1_c0', 'sample2_c0'],
    ...     'ch1': ['sample1_c1', 'sample2_c1'],
    ...     'ch2': ['sample1_c2', 'sample2_c2'],
    ...     'condition': ['Control', 'Treatment'],
    ... &#125;)
    >>> metadata_fn = make_dataframe_metadata_fn(df, filename_column=['ch0', 'ch1', 'ch2'])
    >>> meta = metadata_fn('/data/sample1_c1.tif')
    >>> meta['condition'], meta['channel_index'], meta['ch0']
    ('Control', 1, 'sample1_c0')

## Image I/O — `read_image`

```python
read_image(
    path: str | list[str]
) -> ndarray
```

Read an image and return as a numpy array.

Supports TIFF (via tifffile), NIfTI (via nibabel), Numpy arrays (.npy, .npz), and
common formats (PNG, JPEG, BMP, etc. via OpenCV).
When ``path`` is a list of paths (e.g. one per channel), each file is read and
concatenated along the channel axis, yielding shape (H, W, C_total).

**Args:**

- **`path`**: Path to the image file, or list of paths (one per channel) for
  multi-channel datasets where each channel is in a separate file.

**Returns:**

- **`np.ndarray`**: Shape (H, W) or (H, W, C), dtype float32. All channels returned.

## Image I/O — `ensure_hwc`

```python
ensure_hwc(
    img: numpy.ndarray
) -> ndarray
```

Ensure image is in (H, W, C) format.

**Args:**

- **`img`**: np.ndarray. Supported shapes: (H, W), (C, H, W), (H, W, C).
  For (C, H, W), transposes to (H, W, C) using smallest dim as C.

**Returns:**

- **`np.ndarray`**: Shape (H, W, C). Single channel gets (H, W, 1).

## `CheckpointManager`

Crash-safe incremental checkpoint backed by a single HDF5 file.

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

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-buffer_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-buffer_embeddings"><code>buffer_embeddings</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.buffer_embeddings(
    self,
    embeddings: 'np.ndarray',
    img_paths: 'list[Any]',
    metadata_dicts: 'list[dict[str, Any]]'
) -> None
```

</div>

<div class="api-body">

Add a batch of embeddings to the in-memory buffer.

Parameters
----------
embeddings : np.ndarray
    Shape ``(B, D)`` float32.
img_paths : list
    Length B.  Each element is a ``str`` (single-channel) or a
    ``List[str]`` (multi-channel).
metadata_dicts : list of dict
    Length B.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-buffer_paths_and_metadata">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-buffer_paths_and_metadata"><code>buffer_paths_and_metadata</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.buffer_paths_and_metadata(
    self,
    img_paths: 'list[Any]',
    metadata_dicts: 'list[dict[str, Any]]'
) -> None
```

</div>

<div class="api-body">

Buffer paths and metadata without embeddings (property-only run).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-buffer_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-buffer_properties"><code>buffer_properties</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.buffer_properties(
    self,
    properties_dicts: 'list[dict[str, Any]]'
) -> None
```

</div>

<div class="api-body">

Add property dicts to the in-memory buffer.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-clear_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-clear_properties"><code>clear_properties</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.clear_properties(
    self
) -> None
```

</div>

<div class="api-body">

Truncate all property datasets to 0 and reset n_committed_props.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-close">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-close"><code>close</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.close(
    self
) -> None
```

</div>

<div class="api-body">

Flush any remaining buffers and close the HDF5 file.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-commit_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-commit_embeddings"><code>commit_embeddings</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.commit_embeddings(
    self
) -> int
```

</div>

<div class="api-body">

Flush embedding + path + metadata buffer to HDF5 atomically.

Returns
-------
int
    Total n_committed after this commit.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-commit_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-commit_properties"><code>commit_properties</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.commit_properties(
    self
) -> int
```

</div>

<div class="api-body">

Flush property buffer to HDF5 atomically.

Returns
-------
int
    Total n_committed_props after this commit.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-detect_format">

<div class="api-method-header">
<span class="api-badge api-badge--staticmethod">Static method</span>
<h4 class="api-method-title" id="api-checkpointmanager-detect_format"><code>detect_format</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.detect_format(
    path: 'str'
) -> str
```

</div>

<div class="api-body">

Detect whether *path* is an HDF5 file.

Returns ``'hdf5'`` or ``'unknown'``.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-embedding_dim">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-checkpointmanager-embedding_dim"><code>embedding_dim</code></h4>
</div>

<p><em>Property on <code>CheckpointManager</code></em></p>

<div class="api-body">

Embedding width *D* from file attrs, or ``None`` when not set.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-embeddings_buffered">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-checkpointmanager-embeddings_buffered"><code>embeddings_buffered</code></h4>
</div>

<p><em>Property on <code>CheckpointManager</code></em></p>

<div class="api-body">

Number of paths/embeddings currently in the write buffer.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-get_committed_metadata_list">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-get_committed_metadata_list"><code>get_committed_metadata_list</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.get_committed_metadata_list(
    self
) -> list[dict[str, Any]]
```

</div>

<div class="api-body">

Return committed metadata dicts in order.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-get_committed_paths">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-get_committed_paths"><code>get_committed_paths</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.get_committed_paths(
    self
) -> set[str]
```

</div>

<div class="api-body">

Return the set of committed primary ``img_path`` values (resolved to absolute when possible).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-get_committed_paths_list">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-get_committed_paths_list"><code>get_committed_paths_list</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.get_committed_paths_list(
    self
) -> list[str]
```

</div>

<div class="api-body">

Return committed primary paths in order (resolved to absolute when storage root available).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-get_processing_params">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-get_processing_params"><code>get_processing_params</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.get_processing_params(
    self
) -> dict[str, Any] | None
```

</div>

<div class="api-body">

Return processing parameters from ``/config`` group, or None.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-is_multichannel">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-checkpointmanager-is_multichannel"><code>is_multichannel</code></h4>
</div>

<p><em>Property on <code>CheckpointManager</code></em></p>

<div class="api-body">

Whether the checkpoint stores separate paths per channel.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-load_committed_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-load_committed_properties"><code>load_committed_properties</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.load_committed_properties(
    self
) -> list[dict[str, Any]]
```

</div>

<div class="api-body">

Load only the committed properties (avoids loading embeddings/paths).

Used when merging partial property results.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-load_committed_results">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-load_committed_results"><code>load_committed_results</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.load_committed_results(
    self
) -> PhenoMeResults
```

</div>

<div class="api-body">

Read all committed data and return a :class:`PhenoMeResults`.

Paths are resolved to absolute when _storage_root is available.

Returns
-------
PhenoMeResults
    Embeddings are loaded eagerly into the ``embeddings`` field.
    Returns an empty ``PhenoMeResults`` when no data is committed.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-load_embeddings_by_indices">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-load_embeddings_by_indices"><code>load_embeddings_by_indices</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.load_embeddings_by_indices(
    self,
    indices: 'np.ndarray'
) -> np.ndarray
```

</div>

<div class="api-body">

Load only the specified rows of the embedding dataset.

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

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-load_hdf5_results">

<div class="api-method-header">
<span class="api-badge api-badge--staticmethod">Static method</span>
<h4 class="api-method-title" id="api-checkpointmanager-load_hdf5_results"><code>load_hdf5_results</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.load_hdf5_results(
    path: 'str'
) -> PhenoMeResults
```

</div>

<div class="api-body">

Load and validate an HDF5 results/checkpoint file.

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

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-load_metadata_and_paths">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-load_metadata_and_paths"><code>load_metadata_and_paths</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.load_metadata_and_paths(
    self
) -> tuple[list[Any], list[dict[str, Any]]]
```

</div>

<div class="api-body">

Load all committed paths and metadata dicts.

Paths are resolved to absolute when _storage_root is available (from file or inference).

Returns
-------
(paths, metadata_dicts)
    ``paths`` is a list of ``str`` (single-channel) or ``List[str]``
    (multi-channel).  ``metadata_dicts`` is a list of dicts.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-load_properties_all">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-load_properties_all"><code>load_properties_all</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.load_properties_all(
    self
) -> list[dict[str, Any]]
```

</div>

<div class="api-body">

Load all committed property dicts.

Returns
-------
list of dict
    Length n_committed_props.  Empty list when no properties stored.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-n_committed">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-checkpointmanager-n_committed"><code>n_committed</code></h4>
</div>

<p><em>Property on <code>CheckpointManager</code></em></p>

<div class="api-body">

Rows committed for paths + embeddings + metadata.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-n_committed_props">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-checkpointmanager-n_committed_props"><code>n_committed_props</code></h4>
</div>

<p><em>Property on <code>CheckpointManager</code></em></p>

<div class="api-body">

Rows committed for properties (may lag n_committed).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-properties_buffered">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-checkpointmanager-properties_buffered"><code>properties_buffered</code></h4>
</div>

<p><em>Property on <code>CheckpointManager</code></em></p>

<div class="api-body">

Number of property dicts in the write buffer.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-set_processing_params">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-set_processing_params"><code>set_processing_params</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.set_processing_params(
    self,
    params: 'dict[str, Any]'
) -> None
```

</div>

<div class="api-body">

Write processing params to ``/config`` group (creates if needed).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-validate_processing_params">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-checkpointmanager-validate_processing_params"><code>validate_processing_params</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.validate_processing_params(
    self,
    current_params: 'dict[str, Any]'
) -> list[tuple[str, Any, Any]]
```

</div>

<div class="api-body">

Compare *current_params* against stored params.

Returns list of ``(key, stored_value, current_value)`` for mismatches.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-checkpointmanager-write_results_to_hdf5">

<div class="api-method-header">
<span class="api-badge api-badge--staticmethod">Static method</span>
<h4 class="api-method-title" id="api-checkpointmanager-write_results_to_hdf5"><code>write_results_to_hdf5</code></h4>
</div>

<div class="api-signature">

```python
CheckpointManager.write_results_to_hdf5(
    results: 'Any',
    path: 'str',
    compression: 'str' = 'gzip',
    compression_level: 'int' = 4,
    processing_params: 'dict[str, Any] | None' = None
) -> None
```

</div>

<div class="api-body">

Atomically write a full results object to an HDF5 file.

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

</div>

</div>

## `FileDiscovery`

Finds image files and extracts metadata from directories.

<div class="api-method" role="region" aria-labelledby="api-filediscovery-inspect_data">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-filediscovery-inspect_data"><code>inspect_data</code></h4>
</div>

<div class="api-signature">

```python
FileDiscovery.inspect_data(
    self,
    file_df: pandas.DataFrame
) -> DataFrame
```

</div>

<div class="api-body">

Inspect image and mask dimensions, shapes, and data ranges.

Reads each image (and mask when present) and reports: shape statistics,
image-mask shape matching, and mask dtype/range. Masks are expected in
the ``mask_path`` column of file_df when present.

**Args:**

- **`file_df`**: pd.DataFrame from find_files(). Must have 'file_path' column.
  When masks exist, includes 'mask_path' column.

**Returns:**

pd.DataFrame with columns: file_path, height, width, channels, min, max,
error, shape_match (True/False/None), mask_height, mask_width, mask_dtype,
mask_min, mask_max, mask_error.

</div>

</div>

## `TransformBuilder`

Builds torchvision transform pipelines for image preprocessing.

Uses ImageNet normalization by default.

**Args:**

- **`mean`**: Normalization mean per channel (default: ImageNet).
- **`std`**: Normalization std per channel (default: ImageNet).

<div class="api-method" role="region" aria-labelledby="api-transformbuilder-build">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-transformbuilder-build"><code>build</code></h4>
</div>

<div class="api-signature">

```python
TransformBuilder.build(
    self,
    resize_size: int | None = None,
    pad_size: int | None = None
) -> Compose
```

</div>

<div class="api-body">

Build torchvision transform pipeline for image preprocessing.

Pipeline steps: ToTensor -> TypeMaxNorm -> [PadToSize] -> [Resize] -> Normalize.
When both pad_size and resize_size are given, padding is applied first, then resize
(even if resize_size is smaller than pad_size).

**Args:**

- **`resize_size`**: Target spatial size (H, W) for final resize. If None, no resize.
- **`pad_size`**: Minimum size for H and W before resize. If None, no padding.

**Returns:**

- **`transforms.Compose`**: Pipeline for (H, W, C) numpy image -> (C, H', W') tensor.

</div>

</div>
