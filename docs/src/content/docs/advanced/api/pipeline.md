---
title: "PhenoMe API Reference"
description: "Main orchestrator — data, properties, analysis, reporting."
editUrl: false
tableOfContents:
  maxHeadingLevel: 3
---

<p><span class="api-tier api-tier--public">Tier: Public API</span></p>

:::note[Auto-generated]
This page is rebuilt from docstrings in [`phenome.pipeline`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/pipeline.py), the mixins under [`phenome.mixins`](https://github.com/AAitorG/PhenoMe/tree/main/phenome/mixins), and [`phenome.core.pipeline_results`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/core/pipeline_results.py).
:::

**See also:** [Getting started](/PhenoMe/getting-started/) · [Concepts](/PhenoMe/concepts/) · [Visualization](/PhenoMe/advanced/api/visualization/) · [Distances](/PhenoMe/advanced/api/distances/) · [Model wrappers](/PhenoMe/advanced/api/model-wrapper/)

## Class `PhenoMe`

Main class for phenotyping analysis using deep learning embeddings.

Provides: compute_properties, property_stats_by_group; compute_clustering,
detect_outliers, find_prototypes; compute_reference_distances; correct_batches;
plot_pca, plot_tsne, plot_umap, and related methods.

**Embedding lifecycle:**

- **Eager**: When no checkpoint is used, embeddings live in ``results.embeddings`` (np.ndarray).
- **Lazy**: When a checkpoint is active (``self._db``), ``results.embeddings`` is ``None``;
  embeddings are read on demand via ``get_embeddings()`` from the HDF5 file.
- **Temporal**: Rows from ``process_temporal_images()`` are in-memory only
  (``self._temporal_embeddings``) when ``_db`` is open; merged with checkpoint data in
  ``get_embeddings()``.

**Args:**

- **`device`**: Optional torch.device for GPU-accelerated analysis operations.
  Recommended for faster computation. If None, operations run on CPU.
- **`seed`**: Optional random seed for reproducibility (default: None).
  When set, used for all random operations (determinism, clustering,
  t-SNE, UMAP, mutual information, random sampling in reports).
  If None, no seeds are set anywhere.
- **`use_gpu_for_dr`**: If True and device is CUDA, uses TorchDR for GPU-accelerated
  dimensionality reduction (PCA, t-SNE, UMAP). If False (default), always
  use sklearn/umap-learn on CPU.

```python
PhenoMe(
    device: torch.device | str | None = None,
    seed: int | None = None,
    use_gpu_for_dr: bool = False
)
```

### Data ingestion and embedding extraction

<div class="api-method" role="region" aria-labelledby="api-phenome-find_files">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-find_files"><code>find_files</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.find_files(
    self,
    image_dir: str,
    mask_dir: str | None = None,
    extensions: list[str] | None = None,
    metadata_fn: collections.abc.Callable[[str], dict[str, typing.Any]] | phenome.metadata.base.MetadataBase | None = None,
    on_missing_metadata: str = 'drop',
    mask_filename_column: str | None = None,
    mask_extensions: list[str] | None = None
) -> DataFrame
```

</div>

<div class="api-body">

Discover image and mask files from directories, extract metadata, and cache internally.

Recursively scans `image_dir` for image files and optionally aligns them with masks
from `mask_dir`. Extracts per-file metadata using `metadata_fn` and stores the result
as the internal file DataFrame used by `process_images()`, `compute_properties()`,
and `inspect_data()`.

**Args:**

- **`image_dir`** (`str`): Root directory or glob pattern for image files. Scanned recursively.
  Example: "path/to/images/" or "path/to/**/*.tif".
- **`mask_dir`** (`str or None`): Root directory for corresponding mask files. If provided,
  adds a 'mask_path' column to the output DataFrame. Default: None (no masks).
- **`extensions`** (`list of str or None`): File extensions to include (e.g., [".tif", ".png", ".jpg"]).
  If None, defaults to common image formats. Default: None.
- **`metadata_fn`** (`callable, MetadataBase, or None`): Function or object to extract metadata
  from file paths. Signature: `fn(file_path: str) -> dict[str, Any]`.
  Examples:
  - `DefaultMetadata()`: Simple metadata.
  - `PathTemplateMetadata(template="...")`: Extract from path pattern.
  - `lambda p: {'batch': p.split('/')[-2]}`: Custom function.
  If None, no metadata extracted. Default: None.
- **`on_missing_metadata`** (`str`): How to handle files with missing metadata. One of:
  - 'drop': Remove rows with incomplete metadata.
  - 'keep': Include rows with NaN metadata.
  Default: 'drop'.
- **`mask_filename_column`** (`str or None`): Metadata column name used to locate mask files.
  If set, masks are located using this column value. Useful for custom mask naming.
  Default: None (standard alignment logic).
- **`mask_extensions`** (`list of str or None`): Extensions to try when exact mask filename
  fails (e.g., [".png", ".tif"]). Default: None (exact filename required).

**Returns:**

  pd.DataFrame: File discovery DataFrame with columns:
  - 'file_path': (str or list of str) Image file path(s).
  - 'mask_path': (str, if mask_dir set) Corresponding mask path.
  - Metadata columns: Custom columns from `metadata_fn` if provided.
  Shape: (N, 2+M) where N is number of images, M is metadata columns.

**Raises:**

- **`FileNotFoundError`**: If `image_dir` does not exist.
- **`ValueError`**: If all files are dropped due to `on_missing_metadata='drop'`.
- **`RuntimeError`**: If metadata extraction fails or masks cannot be resolved.

**Example:**

```python
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

```

**See Also:**

- `set_file_df`: Manually provide a file DataFrame.
- `process_images`: Process discovered images.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-set_file_df">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-set_file_df"><code>set_file_df</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.set_file_df(
    self,
    file_df: pandas.DataFrame
) -> DataFrame
```

</div>

<div class="api-body">

Set the internal file DataFrame used for processing.

Validates that file_df has a 'file_path' column and optionally 'mask_path'.
Stores a normalized copy for use by process_images, compute_properties,
inspect_data, and related methods.

**Args:**

- **`file_df`**: DataFrame with 'file_path' column (str or list of str per row).
  May include 'mask_path' and metadata columns.

**Returns:**

  The normalized DataFrame that was stored.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-process_images">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-process_images"><code>process_images</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.process_images(
    self,
    model_wrapper: phenome.utils.model_wrapper.ModelWrapper,
    batch_size: int = 32,
    num_workers: int = 4,
    filters: dict[str, list[typing.Any]] | None = None,
    exclude: dict[str, list[typing.Any]] | None = None,
    channel_mode: Literal['split', 'combined'] = 'split',
    channels: list[int] | None = None,
    preprocessing_fn: collections.abc.Callable[[numpy.ndarray], numpy.ndarray] | None = None,
    custom_transformations: typing.Any | None = None,
    append: bool = False,
    resize_size: int | None = 224,
    pad_size: int | None = None,
    checkpoint_path: str | None = None,
    force_rgb: bool = True,
    save_every: int = 5,
    lazy_checkpoint: bool = True
) -> None
```

</div>

<div class="api-body">

Extract embeddings from images using a pretrained or custom model.

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

**Args:**

- **`model_wrapper`** (`ModelWrapper`): Model instance (e.g., from `load_dinov2_model()`).
  Must implement `_get_embeddings(tensor: torch.Tensor) -> torch.Tensor`.
  For non-PyTorch models (Keras, TensorFlow, etc.), see the
  [Custom Model Wrapper guide](https://AAitorG.github.io/PhenoMe/guides/custom-model-wrapper/).
- **`batch_size`** (`int`): Batch size for processing. Default: 32. Larger batches are
  faster but use more GPU memory.
- **`num_workers`** (`int`): Number of loader processes. In Jupyter, automatically
  reduced to 0 to avoid multiprocessing issues. Default: 4.
- **`filters`** (`dict or None`): Include only rows matching criteria.
  Format: `{column: [value1, value2, ...]}`. Default: None (no filtering).
- **`exclude`** (`dict or None`): Exclude rows matching criteria. Same format as `filters`.
  Default: None.
- **`channel_mode`** (`str`): How to handle multi-channel images. One of:
  - 'split': Process each channel separately (default).
  - 'combined': Process all channels as RGB or grayscale.
- **`channels`** (`list of int or None`): Specific channel indices to use. If None,
  uses all available. Default: None.
- **`preprocessing_fn`** (`callable or None`): Custom preprocessing function applied
  to each image before the model. Signature: `fn(image: ndarray) -> ndarray`.
  Example: normalization, contrast adjustment, etc. Default: None.
- **`custom_transformations`** (`transforms object or None`): PyTorch transforms to apply
  before the model (e.g., resize, normalize). If None, defaults are used
  based on `resize_size` and `pad_size`. Default: None.
- **`append`** (`bool`): If True, append to existing embeddings in `results.embeddings`.
  If False (default), replace. Default: False.
- **`resize_size`** (`int or None`): Image resize dimension (square). Default: 224.
  Set to None to skip resizing.
- **`pad_size`** (`int or None`): Padding size (square). Default: None (no padding).
- **`checkpoint_path`** (`str or None`): HDF5 file path for lazy storage and resumable
  processing. If file exists, processing resumes from last checkpoint.
  If None, embeddings stored in-memory. Default: None.
- **`force_rgb`** (`bool`): If True, convert grayscale to RGB before model. Default: True.
- **`save_every`** (`int`): Save checkpoint every N batches (when using checkpoint_path).
  Default: 5.
- **`lazy_checkpoint`** (`bool`): If True and checkpoint_path is set, enable lazy
  loading (embeddings not kept in-memory). If False, load all embeddings
  after processing. Default: True.

**Returns:**

  None. Modifies `self.results` in-place with embeddings, metadata, and paths.

**Raises:**

- **`FileNotFoundError`**: If file_df not set (call `find_files()` or `set_file_df()` first).
- **`ValueError`**: If filters/exclude reference non-existent columns or produce no data.
- **`RuntimeError`**: If model inference fails or checkpoint is corrupted.
- **`IOError`**: If checkpoint file cannot be created or read.

**Example:**

```python
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

```

**See Also:**

- `find_files`: Discover and organize image files.
- `process_temporal_images`: Incrementally add images to existing checkpoint.
- `get_embeddings`: Retrieve embeddings (handles lazy loading).
- `compute_properties`: Extract morphological properties from embeddings.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-process_temporal_images">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-process_temporal_images"><code>process_temporal_images</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.process_temporal_images(
    self,
    model_wrapper: phenome.utils.model_wrapper.ModelWrapper,
    files: str | list[str] | pandas.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    channel_mode: Optional[Literal['split', 'combined']] = None,
    channels: list[int] | None = None,
    preprocessing_fn: collections.abc.Callable[[numpy.ndarray], numpy.ndarray] | None = None,
    custom_transformations: typing.Any | None = None,
    resize_size: int | None = None,
    pad_size: int | None = None,
    force_rgb: bool | None = None,
    extensions: list[str] | None = None
) -> None
```

</div>

<div class="api-body">

Process new images in-memory (temporary) and append to the current session.

Use this to explore additional images (e.g., from a new condition or replicate)
without re-running process_images. Temporal images are not persisted to checkpoint;
visualize or export before calling clear_temporal_data() or reset().

Requires process_images() to have been run first. Processing parameters
(channel_mode, channels, resize_size, etc.) are inherited from the base run
when not specified. Temporal rows get metadata['source'] = 'NEW'.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-inspect_data">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-inspect_data"><code>inspect_data</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.inspect_data(
    self
) -> DataFrame
```

</div>

<div class="api-body">

Inspect image and mask dimensions, shapes, and data ranges. Delegates to FileDiscovery.

Uses the internally stored file_df (set via set_file_df or find_files).
Raises if file_df is not available.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-clear_temporal_data">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-clear_temporal_data"><code>clear_temporal_data</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.clear_temporal_data(
    self
) -> int
```

</div>

<div class="api-body">

Remove all temporal images (metadata['source'] == 'NEW') from the session.

**Returns:**

- **`int`**: Number of temporal rows removed.

</div>

</div>


### Accessors and inspection

<div class="api-method" role="region" aria-labelledby="api-phenome-get_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-get_embeddings"><code>get_embeddings</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.get_embeddings(
    self,
    indices: Union[list[int], ForwardRef('np.ndarray'), NoneType] = None
) -> numpy.ndarray | None
```

</div>

<div class="api-body">

Return embeddings for the given row indices.

When a checkpoint / HDF5 database is active (``self._db`` is not None),
only the requested rows are read from disk — the core of the lazy-loading
strategy.  When temporal embeddings exist (from process_temporal_images),
they are merged with checkpoint data. When no database is active, slices
``results.embeddings``.

**Args:**

- **`indices`**: Optional integer array/list of row indices (0-based).
  If None, returns all embeddings.

**Returns:**

  np.ndarray shape ``(len(indices), D)`` float32, or None if no
  embeddings are available.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-get_image_info">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-get_image_info"><code>get_image_info</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.get_image_info(
    self,
    idx: int,
    distance_results: pandas.DataFrame | None = None
) -> dict
```

</div>

<div class="api-body">

Return metadata, properties, and optional distance for image idx.

**Args:**

- **`idx`**: Image index (0 to n_images-1).
- **`distance_results`**: Optional DataFrame from compute_reference_distances.

**Returns:**

- **`dict`**: Keys idx, img_name, img_path, metadata keys, property keys,
  distance (if distance_results provided), is_reference (if applicable).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-get_available_metadata_keys">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-get_available_metadata_keys"><code>get_available_metadata_keys</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.get_available_metadata_keys(
    self
) -> list
```

</div>

<div class="api-body">

Return sorted metadata keys.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeproperties-get_available_property_keys">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeproperties-get_available_property_keys"><code>get_available_property_keys</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeProperties.get_available_property_keys(
    self
) -> list
```

</div>

<div class="api-body">

Return sorted list of property keys stored in results.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-has_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenome-has_embeddings"><code>has_embeddings</code></h4>
</div>

<p><em>Property on <code>PhenoMe</code></em></p>

<div class="api-body">

Return True if embedding data is available (lazy or eager).

A properties-only checkpoint has ``n_committed > 0`` but no embedding
dataset; we additionally require ``embedding_dim`` to be a positive int.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-embedding_dim">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenome-embedding_dim"><code>embedding_dim</code></h4>
</div>

<p><em>Property on <code>PhenoMe</code></em></p>

<div class="api-body">

Return the embedding dimensionality, or 0 if unavailable.

</div>

</div>


### Data lifecycle and export

<div class="api-method" role="region" aria-labelledby="api-phenome-reset">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-reset"><code>reset</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.reset(
    self,
    verbose: bool = False,
    clear_file_df: bool = False
) -> None
```

</div>

<div class="api-body">

Reset all stored data to a clean state.

Closes any open HDF5 database handle before clearing results.
By default preserves _file_df so compute_properties and inspect_data
continue to work after process_images. Set clear_file_df=True for a
full reset (e.g. when switching to a completely new dataset).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeproperties-reset_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeproperties-reset_properties"><code>reset_properties</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeProperties.reset_properties(
    self,
    verbose: bool = False
) -> None
```

</div>

<div class="api-body">

Reset only the computed properties.

**Args:**

- **`verbose`**: If True, logs a confirmation message.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeproperties-transfer_metadata_to_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeproperties-transfer_metadata_to_properties"><code>transfer_metadata_to_properties</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeProperties.transfer_metadata_to_properties(
    self,
    columns: str | list[str]
) -> None
```

</div>

<div class="api-body">

Transfer specified metadata columns to properties.

**Args:**

- **`columns`**: A single metadata column name or a list of names to transfer.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-save_results">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-save_results"><code>save_results</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.save_results(
    self,
    path: str | None = None,
    compression: str = 'gzip'
) -> None
```

</div>

<div class="api-body">

Save results to HDF5 (atomic write or in-place flush).

When a checkpoint database is already active (``self._db`` is set),
the data is already on disk; this method flushes any uncommitted
buffers and logs the existing path.  When no database is active,
writes ``self.results`` to a new HDF5 file using a temporary-file +
atomic rename for crash safety.

For full reproducibility, also call [export_experiment_config](/PhenoMe/advanced/api/pipeline/#api-phenome-export_experiment_config)
to save seed, use_gpu_for_dr, reference_filters, and model name
(not stored in the checkpoint).

**Args:**

- **`path`**: Explicit output path.
- **`compression`**: HDF5 compression algorithm (used only for new files).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-load_results">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-load_results"><code>load_results</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.load_results(
    self,
    path: str | None = None,
    lazy_checkpoint: bool = True
) -> None
```

</div>

<div class="api-body">

Load and use an existing results/checkpoint file.

Metadata and properties are loaded immediately. Embeddings are either
loaded into RAM (lazy_checkpoint=False) or accessed on-demand from
disk (lazy_checkpoint=True).

Uses the internally stored file_df (from [find_files](/PhenoMe/advanced/api/pipeline/#api-phenome-find_files) or
[set_file_df](/PhenoMe/advanced/api/pipeline/#api-phenome-set_file_df)) to resolve paths so the checkpoint works on this
machine. Call find_files or set_file_df first.

**Args:**

- **`path`**: Path to .h5 or .hdf5 file.
- **`lazy_checkpoint`**: If True (default), keep the checkpoint file open.
  If False, load all data into RAM and close the file.

**Raises:**

- **`FileNotFoundError`**: If file does not exist.
- **`ValueError`**: If file format is not recognised as HDF5.
- **`RuntimeError`**: If file_df is not available (call find_files or set_file_df first).
- **`ConcurrentCheckpointAccessError`**: If the checkpoint is already open in
  another notebook or process (only one instance can access it at a time).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-checkpoint_context">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-checkpoint_context"><code>checkpoint_context</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.checkpoint_context(
    self,
    path: str | None = None,
    lazy_checkpoint: bool = True
) -> Generator
```

</div>

<div class="api-body">

Context manager that loads a checkpoint and guarantees it is closed on exit.

More reliable than relying on __del__ for cleanup. Use when you need to
ensure the HDF5 file handle is released (e.g. before moving or deleting
the file, or when opening multiple checkpoints in sequence).

**Embeddings:** While the context is open, lazy embeddings are read from the
HDF5 file via ``get_embeddings()``. After exit, ``self._db`` is closed and
``results.embeddings`` remains ``None``; call ``load_results(..., lazy_checkpoint=True)``
again (or another ``checkpoint_context``) before using ``get_embeddings()``.

**Example:**

```python
pheno.find_files("path/to/images")
with pheno.checkpoint_context("results.h5") as p:
    p.plot_pca(color_by="condition")
# Checkpoint closed here
```

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-export_experiment_config">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-export_experiment_config"><code>export_experiment_config</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.export_experiment_config(
    self,
    path: str,
    reference_filters: dict[str, typing.Any] | None = None,
    model_name: str | None = None,
    checkpoint_path: str | None = None,
    **extra: Any
) -> None
```

</div>

<div class="api-body">

Export experiment configuration for reproducibility.

Writes a JSON file with parameters not stored in the HDF5 checkpoint,
so that analyses can be fully reproduced. Call after save_results()
and pass the same reference_filters used for distance analysis.

**Args:**

- **`path`**: Output path for config.json.
- **`reference_filters`**: Optional dict used for compute_reference_distances
  (e.g. &#123;'condition': 'Control'&#125;). Include for full traceability.
- **`model_name`**: Optional model identifier (e.g. 'dinov2_vitb14_reg').
- **`checkpoint_path`**: Optional checkpoint path used during processing (if not
  provided and a checkpoint is open, uses self._db.path).
  **extra: Additional key-value pairs to include in the config.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenome-export_dataset_table">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-export_dataset_table"><code>export_dataset_table</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.export_dataset_table(
    self,
    output_path: str | None = None,
    dist_results: pandas.DataFrame | None = None,
    include_embeddings: Union[bool, Literal['separate']] = False,
    export_format: Literal['csv', 'parquet', 'excel'] = 'csv'
) -> DataFrame
```

</div>

<div class="api-body">

Export the dataset as a table (CSV, Parquet, or Excel).

**Args:**

- **`output_path`**: Path to save the export. If None, only returns the DataFrame.
- **`dist_results`**: Optional DataFrame from compute_reference_distances.
- **`include_embeddings`**: If True, adds embedding columns; if 'separate', saves
  embeddings to a companion .npy file.
- **`export_format`**: Output format: 'csv', 'parquet', or 'excel'.

**Returns:**

  DataFrame with image_index, image_path, metadata, properties, distances,
  and embeddings (if requested).

</div>

</div>


### Properties

<div class="api-method" role="region" aria-labelledby="api-phenomeproperties-compute_properties">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeproperties-compute_properties"><code>compute_properties</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeProperties.compute_properties(
    self,
    metadata_config: typing.Any | None = None,
    property_preset: str | None = None,
    additional_property_functions: dict[str, collections.abc.Callable | list[collections.abc.Callable]] | None = None,
    checkpoint_path: str | None = None,
    save_every: int = 50,
    n_jobs: int = 1,
    lazy_checkpoint: bool = True,
    force_update: bool = False
) -> DataFrame
```

</div>

<div class="api-body">

Compute per-image properties using presets and/or custom functions.

Each function receives a 2D image slice and/or mask slice and returns
a dict[str, float]. When multiple channels exist, properties are
suffixed with ``_ch{idx}``.

Use ``property_preset`` for built-in property sets and
``additional_property_functions`` to add custom functions on top
(optional).

This method now relies solely on ``file_df`` (typically produced by
[PhenoMe.find_files](/PhenoMe/advanced/api/pipeline/#api-phenome-find_files)) for resolving image and mask
paths. Legacy ``image_dir`` / ``mask_dir`` parameters are no longer
supported.

**Args:**

- **`metadata_config`**: Optional MetadataBase instance (e.g. from
  `phenome.metadata`). When it has ``mask_dir`` and
  ``mask_filename_column``, uses explicit mask lookup.
- **`property_preset`**: Preset name. Use ``"none"`` or ``None`` for no
  preset. Valid: ``"none"``, ``"basic"``, ``"regionprops"``,
  ``"intensity"``, ``"full"``, ``"full_extended"``.
- **`additional_property_functions`**: Extra property functions to add on
  top of preset. Dict of {requirement: fn_or_list} where
  requirement is ``"image"``, ``"mask"``, ``"both"``, or
  ``"any"``. Merged with preset when both are used; used alone
  when ``property_preset`` is ``"none"`` or ``None``.
- **`checkpoint_path`**: HDF5 checkpoint for incremental persistence.
- **`save_every`**: Commit frequency (images) when checkpointing.
- **`n_jobs`**: Number of parallel jobs for property computation.
  ``1`` (default) = sequential. Use ``-1`` to use all available
  CPU cores.
- **`lazy_checkpoint`**: If True (default), keep the checkpoint file open.
  If False, load all data into RAM and close the file.
- **`force_update`**: If True and ``checkpoint_path`` points to an existing file,
  clears stored properties in that file, writes current
  ``_processing_params`` into the checkpoint config, and recomputes
  properties for every image (no resume). Use this after changing
  ``property_preset`` or custom functions, or after log warnings about
  missing checkpoint property keys. If ``checkpoint_path``
  is set but the file does not exist yet, only an info log is emitted;
  computation proceeds as for a new checkpoint.

**Returns:**

  DataFrame with all properties and metadata columns. Also populates
  ``results.properties`` as ``List[dict]`` (one dict per image).

**Raises:**

- **`ValueError`**: If no images processed or invalid preset/requirement
  keys are used.
- **`TypeError`**: If ``additional_property_functions`` values are not
  callable.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeproperties-property_stats_by_group">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeproperties-property_stats_by_group"><code>property_stats_by_group</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeProperties.property_stats_by_group(
    self,
    group_by: list[str] | None = None,
    properties: list[str] | None = None,
    print_table: bool = True,
    print_properties: list[str] | None = None,
    group_column_width_max: int = 25,
    content_col_width_max: int = 25
) -> DataFrame
```

</div>

<div class="api-body">

Group properties, compute per-group statistics, and optionally print a table.

Uses `_build_properties_dataframe` from computed
``results.properties`` and metadata (same source as
[compute_properties](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-compute_properties)).

**Args:**

- **`group_by`**: Metadata columns to group by. If None, auto-selects first 2.
- **`properties`**: Property columns to include in aggregation. If None, auto-detects numeric.
- **`print_table`**: If True, log a formatted mean±std table.
- **`print_properties`**: Subset of properties to show in the table. If None, shows all.
- **`group_column_width_max`**: Maximum width for each grouping column when printing.
- **`content_col_width_max`**: Maximum width for each property statistic column when printing.

**Returns:**

  Aggregated DataFrame with mean/std/min/max per property per group.

**Raises:**

- **`TypeError`**: If group_by/properties are invalid types.
- **`ValueError`**: If group_by keys are not in DataFrame columns.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeproperties-top_properties_different_from_reference">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeproperties-top_properties_different_from_reference"><code>top_properties_different_from_reference</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeProperties.top_properties_different_from_reference(
    self,
    df: pandas.DataFrame,
    reference_group: dict[str, typing.Any],
    k: int = 5,
    properties: list[str] | None = None,
    metric: Literal['cohens_d', 'mean_diff'] = 'cohens_d',
    print_output: bool = True,
    group_column_width_max: int = 25,
    top_property_col_width_max: int = 25,
    top_effect_col_width_min: int = 12
) -> DataFrame
```

</div>

<div class="api-body">

For each non-reference group, return the top k properties that most differentiate it from the reference.

Uses Cohen's d (effect size) by default to rank properties by how different each group
is from the reference. Both higher and lower values count as "different" (uses |d|).

Cohen's d formula (pooled): d = (mean_group - mean_ref) / s_pooled, where
s_pooled = sqrt([(n_ref-1)*std_ref² + (n_other-1)*std_other²] / (n_ref + n_other - 2)).
Requires sample std (ddof=1) from property_stats_by_group.

**Args:**

- **`df`**: Aggregated DataFrame from property_stats_by_group().
- **`reference_group`**: Dict mapping grouping column names to values (e.g. &#123;"drug": "Control", "time": "60_min"&#125;).
- **`k`**: Number of top properties per group.
- **`properties`**: Property names to consider. If None, uses all in DataFrame.
- **`metric`**: 'cohens_d' (effect size) or 'mean_diff' (absolute mean difference).
- **`print_output`**: If True, pretty-print the results.
- **`group_column_width_max`**: Maximum width for each grouping column in the printed table.
- **`top_property_col_width_max`**: Max width for the property name column in the printed table.
- **`top_effect_col_width_min`**: Min width for Cohen's d and mean diff columns when printing.

**Returns:**

  DataFrame with columns: grouping cols, property, effect_size (Cohen's d when
  ``metric='cohens_d'``), mean_diff, ref_mean, group_mean, rank. One row per
  (group, property) for top-k only.

</div>

</div>


### Advanced analysis

<div class="api-method" role="region" aria-labelledby="api-phenomebatchcorrection-correct_batches">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomebatchcorrection-correct_batches"><code>correct_batches</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeBatchCorrection.correct_batches(
    self,
    batch_metadata_key: 'str',
    method: 'MethodName' = 'sphering',
    source: 'SourceName' = 'embeddings',
    property_keys: 'list[str] | None' = None,
    control_filters: 'dict[str, Any] | None' = None,
    inplace: 'bool' = True,
    checkpoint_path: 'str | None' = None,
    chunk_size: 'int' = 4096,
    force: 'bool' = False,
    ridge_multiplier: 'float' = 0.001,
    min_controls: 'int' = 2
) -> dict[str, Any]
```

</div>

<div class="api-body">

Correct plate-to-plate variation using control wells per batch.

Call after :meth:`process_images` (and optionally after ``compute_properties`` when
``source='properties'``). For lazy checkpoints, embeddings are read and rewritten
in chunks of *chunk_size* rows.

**Args:**

- **`batch_metadata_key`**: Metadata column for batch / plate id (case-insensitive).
- **`method`**: ``"sphering"`` (default) or ``"zscore"``.
- **`source`**: ``"embeddings"`` or ``"properties"``.
- **`property_keys`**: Property names to correct when ``source='properties'``.
  If None, uses all keys from :meth:`get_available_property_keys`.
- **`control_filters`**: Same semantics as ``filter_indices`` filters (required).
- **`inplace`**: If True, updates ``results.embeddings`` or ``results.properties`` in place.
- **`checkpoint_path`**: If set, persist batch-correction metadata and/or export results.
  - If ``None`` (default): Corrections still apply (RAM or an already-open lazy
  checkpoint), but nothing new is written for persistence—no ``/batch_correction``
  stats on disk and no new HDF5 via :meth:`save_results`.
  - If ``str``: With an open checkpoint, it is cloned to this path first; stats are
  saved under ``/batch_correction`` when a DB is active; in eager mode (no DB),
  results are saved to this path after correction.
- **`chunk_size`**: Rows per chunk for embedding I/O and property scatter.
- **`force`**: If True, allow re-applying correction after a previous run in this session.
- **`ridge_multiplier`**: Covariance ridge strength for sphering (see :func:`compute_batch_stats`).
- **`min_controls`**: Minimum control wells required per batch.

**Returns:**

  Dict with batch stats summary (``batches``, ``controls_per_batch``, ``method``, …).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-compute_clustering">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-compute_clustering"><code>compute_clustering</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.compute_clustering(
    self,
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    n_clusters: int = 5,
    clustering_method: Literal['kmeans', 'dbscan', 'gmm'] = 'kmeans',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any | list[typing.Any]] | None = None,
    exclude: dict[str, typing.Any | list[typing.Any]] | None = None,
    random_state: int | None = None,
    normalize: bool = True,
    reduce_dim: int | None = 100,
    reduce_method: Literal['pca', 'tsne', 'umap'] = 'pca',
    return_silhouette: bool = False,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int | None = None
) -> numpy.ndarray | tuple[numpy.ndarray, float | None]
```

</div>

<div class="api-body">

Perform clustering and store labels in ``metadata[i]['cluster']`` for each image.

**Args:**

- **`source`**: ``'embeddings'``, ``'properties'``, or ``'combined'``.
- **`n_clusters`**: Number of clusters (k-means, GMM). Ignored for ``clustering_method='dbscan'``.
- **`clustering_method`**: ``'kmeans'`` (default), ``'dbscan'``, or ``'gmm'``.
- **`property_keys`**: Property subset when *source='properties'*.
- **`filters`**: Optional metadata filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`random_state`**: Seed for k-means/GMM. If None, uses pipeline seed when available.
- **`normalize`**: Whether to normalize data before clustering (default: True).
  For embeddings, uses L2 normalization. For properties, uses StandardScaler.
- **`reduce_dim`**: If set, apply dimensionality reduction before clustering
  (default: 100). Only applied when data has more features than *reduce_dim*.
  Set to None to disable dimensionality reduction.
- **`reduce_method`**: Method for dimensionality reduction when *reduce_dim* is set:
  ``'pca'`` (fast, linear), ``'tsne'`` (non-linear, slower), ``'umap'``
  (non-linear, preserves structure). Default: ``'pca'``.
- **`return_silhouette`**: If True, compute and return the silhouette score (default: False).
  Returns ``(labels, score)``; score is None if it could not be computed.
  Silhouette requires at least 2 clusters and 2 samples per cluster.
- **`dbscan_eps`**: Maximum distance between two samples for DBSCAN (only when
  clustering_method='dbscan'). If None, uses 0.5.
- **`dbscan_min_samples`**: Minimum samples in a neighborhood for DBSCAN (only when
  clustering_method='dbscan'). If None, uses max(2, n_samples // 20).

**Returns:**

  If return_silhouette=False: np.ndarray of shape (n_total,) with cluster labels.
  If return_silhouette=True: Tuple of (labels, silhouette_score). Score is None
  if it could not be computed. Labels: 0..K-1; NaN for excluded/DBSCAN noise.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-detect_outliers">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-detect_outliers"><code>detect_outliers</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.detect_outliers(
    self,
    method: Literal['z-score', 'iqr'] = 'z-score',
    threshold: float = 3.0,
    group_by: str | list[str] | None = None,
    filters: dict[str, typing.Any] | None = None,
    exclude: dict[str, typing.Any] | None = None,
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    normalize: bool = True,
    drop_outliers: bool = False,
    plot: bool = True
) -> DataFrame
```

</div>

<div class="api-body">

Detect outliers based on distance to centroid.

**Args:**

- **`method`**: ``'z-score'`` or ``'iqr'``.
- **`threshold`**: Threshold multiplier.
- **`group_by`**: Property/metadata key(s) for grouping. Can be a single string
  or a list of strings. If a list is provided, groups are formed by
  combining values from all specified columns (e.g., "drug1-10uM").
  If *None*, all images are treated as one group.
- **`filters`**: Optional metadata filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`source`**: Feature space for outlier detection:
  - ``'embeddings'``: model embeddings
  - ``'properties'``: scalar phenotypic properties
  - ``'combined'``: normalized concatenation of embeddings and properties.
- **`property_keys`**: Property subset when *source* is ``'properties'`` or ``'combined'``.
- **`normalize`**: Whether to normalize data before outlier detection (default: True).
  For embeddings, uses L2 normalization. For properties, uses StandardScaler.
- **`drop_outliers`**: If True, remove detected outliers from self.results (default: False).
- **`plot`**: If True (default), shows one matplotlib figure per group: subplots for
  detected outliers and a figure title naming the group (``Group: …``).
  Images are shown without pipeline ``image_transforms`` (same as
  ``apply_transforms=False`` for display).

**Returns:**

  pd.DataFrame: A DataFrame containing detected outliers.
  Columns include:
  - ``idx``: Global image index.
  - ``file_path``: Path to the image file.
  - Additional columns for each grouping key (if *group_by* was set).
  - ``threshold``: The threshold value used for the image's group.
  - ``distance_to_centroid``: Distance to the group centroid.
  If no outliers are detected, returns an empty DataFrame.
  The index is a range index from 0 to N-1.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-find_prototypes">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-find_prototypes"><code>find_prototypes</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.find_prototypes(
    self,
    cluster_col: str | list[str] | None = 'cluster',
    n_prototypes: int = 5,
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any] | None = None,
    exclude: dict[str, typing.Any] | None = None,
    metric: str = 'euclidean',
    normalize: bool = True,
    plot: bool = True
) -> DataFrame
```

</div>

<div class="api-body">

Find images closest to each group centroid.

**Args:**

- **`cluster_col`**: Property/metadata key(s) for grouping. Can be a single string
  or a list of strings. If a list is provided, groups are formed by
  combining values from all specified columns (e.g., "drug1-10uM").
  If *None*, all images are treated as one group.
- **`n_prototypes`**: How many prototypes per group.
- **`source`**: ``'embeddings'``, ``'properties'``, or ``'combined'``.
- **`property_keys`**: Property subset when *source='properties'*.
- **`filters`**: Optional metadata filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`metric`**: ``'euclidean'`` or ``'cosine'``.
- **`normalize`**: Whether to normalize data before finding prototypes (default: True).
  For embeddings, uses L2 normalization. For properties, uses StandardScaler.
- **`plot`**: If True (default), shows one matplotlib figure per group: subplots for
  that group's prototypes and a figure title naming the group (``Group: …``).
  Images are shown without pipeline ``image_transforms`` (same as
  ``apply_transforms=False`` for display).

**Returns:**

  pd.DataFrame: DataFrame with one row per prototype. Columns include grouping
  keys (split into individual columns if multiple), ``prototype_index``,
  and ``image_path``.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-analyze_group_enrichment">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-analyze_group_enrichment"><code>analyze_group_enrichment</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.analyze_group_enrichment(
    self,
    group_by: str | list[str] = 'cluster',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any] | None = None,
    exclude: dict[str, typing.Any] | None = None,
    plot: bool = True,
    return_fig: bool = False,
    top_k: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    correct_multiple_testing: bool = True
) -> pandas.DataFrame | tuple[pandas.DataFrame, Any]
```

</div>

<div class="api-body">

Compute z-score enrichment of properties per group (e.g. cluster labels).

For each group, z-scores are computed against a **leave-group-out**
population (all samples *except* the current group).  When
*correct_multiple_testing* is True (default), Benjamini-Hochberg FDR
correction is applied across all (group, property) pairs and a
``Significant`` column is added to the output DataFrame.

**Args:**

- **`group_by`**: Property/metadata key(s) holding group labels (e.g. ``"cluster"``).
  Can be a single string or a list of strings for composite grouping.
- **`property_keys`**: Properties to analyse (all numeric if *None*).
- **`filters`**: Optional metadata filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`plot`**: If True (default), show an interactive Plotly faceted bar chart (one row per
  group). If False, log a plain-text summary via the package logger instead
  (unless *return_fig* requests a figure).
- **`return_fig`**: If True, return a tuple ``(df, fig)`` where ``fig`` is the Plotly figure.
  When *return_fig* is True, ``fig.show()`` is not called; use
  ``plot=True`` with ``return_fig=False`` for the default interactive display.
- **`top_k`**: Max properties per group in the figure and in the text summary (``None`` = all).
- **`figsize`**: Figure size ``(width, height)`` in inches for the Plotly layout.
- **`title`**: Optional figure title.
- **`correct_multiple_testing`** (`bool`): If True (default), apply Benjamini-Hochberg
  FDR correction across all (group, property) z-scores and add a
  ``Significant`` column (alpha = 0.05).

**Returns:**

  pd.DataFrame or tuple[pd.DataFrame, Any]:
  - If *return_fig* is False (default): pd.DataFrame with columns:
  [group_by columns], Property, Score, Mean_Group, Mean_Pop, AbsScore,
  and optionally p_value / Significant when *correct_multiple_testing* is True.
  - If *return_fig* is True: A tuple (enrichment_df, fig).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-compute_component_correlation">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-compute_component_correlation"><code>compute_component_correlation</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.compute_component_correlation(
    self,
    method: Literal['pca', 'tsne', 'umap'] = 'pca',
    n_components: int = 2,
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any] | None = None,
    exclude: dict[str, typing.Any] | None = None,
    top_k: int | None = None,
    normalize: bool = True,
    correlation_method: Literal['pearson', 'spearman', 'distance_correlation', 'mutual_info'] = 'pearson',
    plot: bool = True,
    return_fig: bool = False,
    figsize: tuple[int, int] = (10, 6)
) -> pandas.DataFrame | tuple[pandas.DataFrame, Any]
```

</div>

<div class="api-body">

Correlate dim-reduction components with phenotypic properties.

**Args:**

- **`method`**: Dimensionality reduction method.
- **`n_components`**: Number of components to compute.
- **`source`**: ``'embeddings'``, ``'properties'``, or ``'combined'``.
- **`property_keys`**: Property subset when *source* is ``'properties'`` or ``'combined'``.
- **`filters`**: Optional metadata filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`top_k`**: Number of top correlations to return in summary and per facet in the plot.
- **`normalize`**: Whether to normalize data before computing correlations (default: True).
  - Source data (embeddings/properties): Normalized before dimensionality reduction
  (L2 normalization for embeddings, StandardScaler for properties)
  - Properties: Normalized using StandardScaler before correlation computation
- **`correlation_method`**: Correlation method to use. Options:
  - ``'pearson'``: Pearson correlation coefficient (default, linear relationships)
  - ``'spearman'``: Spearman rank correlation (monotonic relationships, robust to outliers)
  - ``'distance_correlation'``: Distance correlation (detects non-linear relationships,
  requires ``dcor`` library: ``pip install dcor``)
  - ``'mutual_info'``: MI-derived correlation coefficient (detects any dependency,
  mapped to [0, 1] via the Gaussian bivariate transform, **not** standard NMI)
- **`plot`**: If True (default), show an interactive Plotly faceted bar chart (one row per
  component). If False, log a plain-text summary via the package logger instead
  (unless *return_fig* requests a figure).
- **`return_fig`**: If True, return a tuple of ``(correlation_df, figure)``.
  When ``return_fig`` is True, ``fig.show()`` is not called.
- **`figsize`**: Figure size ``(width, height)`` in inches for the Plotly layout.

**Returns:**

  pd.DataFrame | tuple[pd.DataFrame, Any]:
  - If ``return_fig`` is False (default): Returns the correlation DataFrame
  with a ``Properties`` column and an integer index.
  - If ``return_fig`` is True: Returns a tuple of ``(correlation_df, figure)``.

**Raises:**

- **`ValueError`**: If dimensionality reduction fails, no components are found,
  no numeric properties are available, or correlation computation fails.

**Examples:**

```python
Get correlations directly::

    corr = pheno.compute_component_correlation(plot=False)

Get correlations and save the figure::

    corr, fig = pheno.compute_component_correlation(return_fig=True)
    fig.write_html("component_corr.html")
```

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-compute_embedding_property_correlations">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-compute_embedding_property_correlations"><code>compute_embedding_property_correlations</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.compute_embedding_property_correlations(
    self,
    property_keys: list[str] | None = None,
    normalize: bool = True,
    method: Literal['pearson', 'spearman', 'distance_correlation', 'mutual_info'] = 'pearson',
    n_jobs: int = 1
) -> dict
```

</div>

<div class="api-body">

Compute correlations between embedding dimensions and phenotypic scalar properties.

This is the time-consuming step that computes raw correlations for each property.

**Args:**

- **`property_keys`**: Property names (all if *None*).
- **`normalize`**: Whether to normalize data before computing correlations (default: True).
  - Embeddings: L2 normalization (unit norm)
  - Properties: StandardScaler normalization (zero mean, unit variance)
- **`method`**: Correlation method to use. Options:
  - ``'pearson'``: Pearson correlation coefficient (default, linear relationships)
  - ``'spearman'``: Spearman rank correlation (monotonic relationships, robust to outliers)
  - ``'distance_correlation'``: Distance correlation (detects non-linear relationships,
  requires ``dcor`` library: ``pip install dcor``)
  - ``'mutual_info'``: MI-derived correlation coefficient (detects any dependency,
  mapped to [0, 1] via the Gaussian bivariate transform, **not** standard NMI)

**Returns:**

  Dict mapping property names to correlation arrays (n_dims,)

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-summarize_embedding_property_correlations">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-summarize_embedding_property_correlations"><code>summarize_embedding_property_correlations</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.summarize_embedding_property_correlations(
    self,
    correlations: dict[str, numpy.ndarray],
    order_by: str = 'mean_abs',
    top_k: int | None = 20,
    plot: bool = True,
    return_fig: bool = False,
    figsize: tuple[int, int] = (10, 8),
    title: str = 'Property Correlations with Embeddings',
    embedding_shape: tuple[int, int] | None = None,
    n_properties: int | None = None,
    correlation_method: str = 'pearson'
) -> pandas.DataFrame | tuple[pandas.DataFrame, Any | None]
```

</div>

<div class="api-body">

Summarize embedding-property correlations, optionally plot, and/or return a Plotly figure.

Computes per-property ``mean_abs``, ``std`` (across dimensions), ``max_abs``, ``min_abs``,
and ``mean``, orders rows by ``order_by``, and either shows a
horizontal violin plot of the distribution of |r| across dimensions per property or logs
a plain-text table when ``plot`` is *False*.

**Args:**

- **`correlations`**: Output from :meth:`compute_embedding_property_correlations`
  (dict mapping property names to correlation arrays).
- **`order_by`**: Metric used to sort properties (descending):
  ``mean_abs`` | ``max_abs`` | ``mean`` | ``std``.
- **`top_k`**: Number of top properties shown in the plot and listed in ``top_properties``;
  full sorted table is always in ``summary`` (*None* = all).
- **`plot`**: If *True* (default), show a Plotly violin plot of |r| per dimension for the top
  ``top_k`` properties by ``order_by`` (*None* = all). Highest metric at the **top**
  of the y-axis.
- **`return_fig`**: If *True*, return both the summary DataFrame and the
  Plotly figure.
- **`figsize`**: Figure size in inches, converted to pixels for Plotly layout.
- **`title`**: Chart title.
- **`embedding_shape`**: Shape of embeddings (n_samples, n_dims) for metadata display.
- **`n_properties`**: Number of properties processed for metadata display.
- **`correlation_method`**: Method used for correlation computation for metadata display.

**Returns:**

  Sorted DataFrame with ``property``, ``mean_abs``, ``std``, ``max_abs``, ``min_abs``,
  and ``mean``. If ``return_fig`` is *True*, returns ``(summary, figure)``.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeanalysis-compute_multivariate_interpretability">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeanalysis-compute_multivariate_interpretability"><code>compute_multivariate_interpretability</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeAnalysis.compute_multivariate_interpretability(
    self,
    method: Literal['pca', 'tsne', 'umap'] = 'tsne',
    component: int = 1,
    model_type: Literal['lasso', 'random_forest'] = 'lasso',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any] | None = None,
    exclude: dict[str, typing.Any] | None = None,
    normalize: bool = True,
    cv: int = 5,
    rf_n_estimators: int = 100,
    seed: int | None = None,
    plot: bool = True,
    return_fig: bool = False,
    top_k: int = 10,
    figsize: tuple[int, int] = (10, 8)
) -> pandas.DataFrame | tuple[pandas.DataFrame, Any]
```

</div>

<div class="api-body">

Explain a dimensionality reduction component using LASSO or Random Forest.

Calculates which phenotypic properties (features) best explain the variability
seen in a deep learning embedding dimension (the target, usually t-SNE 1 or 2).

**Args:**

- **`method`**: Dimensionality reduction method ('pca', 'tsne', or 'umap').
- **`component`**: Which component to explain (1, 2, ...).
- **`model_type`**: The regression model to use ('lasso' or 'random_forest').
  'lasso' uses L1 regularization for linear, sparse explanations.
  'random_forest' captures non-linear relationships.
- **`property_keys`**: Subset of properties to use as features.
- **`filters`**: Optional metadata filters.
- **`exclude`**: Optional metadata exclusions.
- **`normalize`**: Whether to normalize features before regression (default: True).
  Uses StandardScaler for properties to ensure comparable coefficients.
- **`cv`**: Number of cross-validation folds (only for 'lasso').
- **`rf_n_estimators`**: Number of trees (only for 'random_forest').
- **`seed`**: Random seed for reproducibility. If None, uses the pipeline's ``seed`` when set.
- **`plot`**: If True (default), show an interactive Plotly bar chart of top drivers.
  If False, log a plain-text summary via the package logger instead.
- **`return_fig`**: If True, returns a tuple of ``(df, fig)``.
  When ``return_fig`` is True, ``fig.show()`` is not called; use
  ``plot=True`` with ``return_fig=False`` for interactive display.
- **`top_k`**: Number of top drivers to show in the plot or text summary.
- **`figsize`**: Figure size ``(width, height)`` passed through to Plotly layout; each
  value is multiplied by 100 to set width and height in layout pixels.

**Returns:**

  pd.DataFrame | tuple[pd.DataFrame, Any]:
  - If ``return_fig`` is False (default): A DataFrame of ranked drivers.
  Columns are ``feature`` and ``weight``. Metadata is in ``df.attrs``.
  - If ``return_fig`` is True: A tuple of ``(df, fig)``.
  The DataFrame contains:
  - ``feature``: The phenotypic property name.
  - ``weight``: LASSO coefficient or Random Forest Gini importance.
  Metadata in ``df.attrs`` includes:
  - ``r2``: Explainability Score (R^2) for the model.
  - ``n_samples``: Number of samples used for the model.
  - ``n_features``: Total number of features considered.
  - ``method``: The dimensionality reduction method used.
  - ``model_type``: The regression model type used.
  - ``target_component``: The component name explained.

</div>

</div>


### Reporting

<div class="api-method" role="region" aria-labelledby="api-phenome-generate_report">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenome-generate_report"><code>generate_report</code></h4>
</div>

<div class="api-signature">

```python
PhenoMe.generate_report(
    self,
    output_path: str = 'pheno_report.html',
    title: str = 'PhenoMe Analysis Report',
    config: typing.Any | None = None,
    **overrides: Any
) -> str
```

</div>

<div class="api-body">

Generate a comprehensive standalone HTML report from phenotyping results.

Uses ReportConfig as the primary source of options. Pass config= for full control,
or use **overrides to tweak individual settings (e.g. include_plots=False).

**Args:**

- **`output_path`**: Path to save the HTML file.
- **`title`**: Report title displayed at the top.
- **`config`**: Optional ReportConfig for defaults. If None, uses ReportConfig().
  **overrides: Any ReportConfig field to override (e.g. include_plots=False,
  outlier_threshold=2.5, n_clusters=10).

**Returns:**

  Path to the generated HTML file.

</div>

</div>


### Other

<div class="api-method" role="region" aria-labelledby="api-_imagedisplaymixin-image_preview_png_bytes">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_imagedisplaymixin-image_preview_png_bytes"><code>image_preview_png_bytes</code></h4>
</div>

<div class="api-signature">

```python
_ImageDisplayMixin.image_preview_png_bytes(
    self,
    idx: int,
    distance_results: pandas.DataFrame | None = None,
    channels: int | list | None = None,
    title_fields: list[str] | None = None,
    show_extra_info: bool = False,
    apply_transforms: bool = True,
    downsample: int | None = 720
) -> tuple
```

</div>

<div class="api-body">

Rasterize the same view as ``plot_image_by_index`` to PNG bytes.

Used by the interactive explorer with ``ipywidgets.Image`` because matplotlib
``display()`` / ``plt.show()`` from Plotly click callbacks does not reliably
target a nested ``Output`` widget (output goes to the cell or nowhere).

When ``show_extra_info`` is True, also emits the same details as
``plot_image_by_index`` via logging and returns their text for UI display.

**Returns:**

  ``(png_bytes, details_text_or_none)`` — ``details_text_or_none`` is the
  formatted extra-info block when ``show_extra_info`` is True and details exist.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-_drplotsmixin-plot_centroids">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_drplotsmixin-plot_centroids"><code>plot_centroids</code></h4>
</div>

<div class="api-signature">

```python
_DRPlotsMixin.plot_centroids(
    self,
    group_by: str | list[str],
    method: Literal['pca', 'tsne', 'umap'] = 'pca',
    n_components: int = 2,
    filters: dict[str, typing.Any | list[typing.Any]] | None = None,
    exclude: dict[str, typing.Any | list[typing.Any]] | None = None,
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    show_points: bool = True,
    show_centroids: bool = True,
    figsize: tuple[int, int] = (10, 8),
    return_fig: bool = False,
    normalize: bool = True,
    **dr_kwargs: Any
) -> Any
```

</div>

<div class="api-body">

Plot centroids of groups in reduced embedding space.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-_imagedisplaymixin-plot_image_by_index">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_imagedisplaymixin-plot_image_by_index"><code>plot_image_by_index</code></h4>
</div>

<div class="api-signature">

```python
_ImageDisplayMixin.plot_image_by_index(
    self,
    idx: int,
    distance_results: pandas.DataFrame | None = None,
    channels: int | list | None = None,
    figsize: tuple = (6, 6),
    title_fields: list[str] | None = None,
    show_extra_info: bool = True,
    apply_transforms: bool = False,
    downsample: int | None = 720,
    ax: Any = None,
    return_fig: bool = False,
    show_mask_overlay: bool = False
) -> Any
```

</div>

<div class="api-body">

Plot a specific image by its index.

**Args:**

- **`idx`**: Image index in results
- **`distance_results`**: Optional distance computation results DataFrame
- **`channels`**: specific channels to plot
- **`figsize`**: Figure size
- **`title_fields`**: Optional list of field names to display in title
- **`show_extra_info`**: If True, prints detailed information after plotting
- **`apply_transforms`**: If True, apply pipeline image_transforms
- **`downsample`**: If not None, approximate desired size (in pixels) for the
  longest image edge when downsampling. The final image size may
  differ slightly due to integer stepping. If None, no downsampling.
- **`ax`**: Optional matplotlib axes to plot on. If provided, a new figure is not created.
- **`return_fig`**: If True, returns the matplotlib figure object.
- **`show_mask_overlay`**: If True, draws the segmentation mask as a semi-transparent
  yellow overlay with a crisp contour. Requires masks to have been discovered
  via ``mask_dir`` in ``find_files``. Useful for verifying that masks load
  correctly and spatially align with their images.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-_drplotsmixin-plot_pca">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_drplotsmixin-plot_pca"><code>plot_pca</code></h4>
</div>

<div class="api-signature">

```python
_DRPlotsMixin.plot_pca(
    self,
    n_components: int = 2,
    color_by: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any | list[typing.Any]] | None = None,
    exclude: dict[str, typing.Any | list[typing.Any]] | None = None,
    hover_features: list[str] | None = None,
    return_fig: bool = False,
    render_mode: Literal['auto', 'svg', 'webgl'] = 'webgl',
    normalize: bool = True,
    sample_size: int | None = None,
    **dr_kwargs: Any
) -> Any
```

</div>

<div class="api-body">

Plot PCA of embeddings, properties, or combined features (Plotly, WebGL by default).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-_drplotsmixin-plot_tsne">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_drplotsmixin-plot_tsne"><code>plot_tsne</code></h4>
</div>

<div class="api-signature">

```python
_DRPlotsMixin.plot_tsne(
    self,
    n_components: int = 2,
    perplexity: float = 30.0,
    color_by: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any | list[typing.Any]] | None = None,
    exclude: dict[str, typing.Any | list[typing.Any]] | None = None,
    hover_features: list[str] | None = None,
    return_fig: bool = False,
    render_mode: Literal['auto', 'svg', 'webgl'] = 'webgl',
    normalize: bool = True,
    sample_size: int | None = None,
    **dr_kwargs: Any
) -> Any
```

</div>

<div class="api-body">

Plot t-SNE of embeddings, properties, or combined features (Plotly, WebGL by default).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-_drplotsmixin-plot_umap">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_drplotsmixin-plot_umap"><code>plot_umap</code></h4>
</div>

<div class="api-signature">

```python
_DRPlotsMixin.plot_umap(
    self,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    color_by: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: list[str] | None = None,
    filters: dict[str, typing.Any | list[typing.Any]] | None = None,
    exclude: dict[str, typing.Any | list[typing.Any]] | None = None,
    hover_features: list[str] | None = None,
    return_fig: bool = False,
    render_mode: Literal['auto', 'svg', 'webgl'] = 'webgl',
    normalize: bool = True,
    sample_size: int | None = None,
    **dr_kwargs: Any
) -> Any
```

</div>

<div class="api-body">

Plot UMAP of embeddings, properties, or combined features (Plotly, WebGL by default).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-_distanceplotsmixin-print_distance_summary">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_distanceplotsmixin-print_distance_summary"><code>print_distance_summary</code></h4>
</div>

<div class="api-signature">

```python
_DistancePlotsMixin.print_distance_summary(
    self,
    distance_results: pandas.DataFrame,
    group_by: str | None = None,
    dist_range: tuple = (0, 100)
) -> None
```

</div>

<div class="api-body">

Print distance summary statistics without plotting. Safe to use when enable_plots=False.

**Args:**

- **`distance_results`**: DataFrame containing 'distance' column
- **`group_by`**: Metadata key to group by
- **`dist_range`**: Tuple of (min, max) distances to include

</div>

</div>



## Class `PhenoMeResults`

Typed container for per-image phenotyping data.

Replaces the plain ``Dict[str, Any]`` that was previously used as
``PhenoMe.results``.  It exposes a fully dict-compatible
interface (``__getitem__``, ``__setitem__``, ``get``, ``__contains__``,
``__iter__``) so that all existing mixin code keeps working without any
changes.  On top of that it provides typed attributes and convenience
properties for cleaner client code.

**Embeddings lifecycle**
``embeddings`` tracks the eager copy held by the pipeline:

* **Not yet computed** — ``embeddings is None``, ``pipeline._db is None``
* **Lazy-backed (HDF5)** — ``embeddings is None``, ``pipeline._db is not None``
  (rows are fetched from disk on demand)
* **Eagerly computed** — ``embeddings`` is an ``np.ndarray`` of shape ``(N, D)``

The former ``[]``-sentinel that mixed "uninitialised" with "lazy" with
"being filled during extraction" has been removed.  In-progress embeddings
during extraction live in ``pipeline._emb_buffer`` (a plain ``List[np.ndarray]``)
and never touch this container until they are finalised.

**Parameters:**

- **`img_path`** (`list of str or list of list[str]`):
  Per-image file path(s).  Single-channel images use a plain ``str``;
  multi-channel images use a ``List[str]`` with one path per channel.
- **`metadata`** (`list of dict`):
  Per-image metadata dicts (e.g. ``{"drug": "DMSO", "time": "24h"}``).
- **`properties`** (`list of dict`):
  Per-image computed scalar properties
  (e.g. ``{"intensity_mean": 0.42, "area": 1024}``).
- **`embeddings`** (`np.ndarray or None`):
  Shape ``(N, D)`` float32 array, or ``None`` when not available /
  when embeddings are stored lazily in an HDF5 file.

### Accessors and inspection

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-has_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenomeresults-has_embeddings"><code>has_embeddings</code></h4>
</div>

<p><em>Property on <code>PhenoMeResults</code></em></p>

<div class="api-body">

True if embeddings are stored eagerly (not lazy or absent).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-embedding_dim">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenomeresults-embedding_dim"><code>embedding_dim</code></h4>
</div>

<p><em>Property on <code>PhenoMeResults</code></em></p>

<div class="api-body">

Embedding dimensionality, or 0 if no embeddings are stored.

</div>

</div>


### Other

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-clear">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-clear"><code>clear</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.clear(
    self
) -> None
```

</div>

<div class="api-body">

Reset all fields to empty state.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-get">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-get"><code>get</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.get(
    self,
    key: 'str',
    default: 'Any' = None
) -> Any
```

</div>

<div class="api-body">

Dict-style .get() with default.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-has_properties">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenomeresults-has_properties"><code>has_properties</code></h4>
</div>

<p><em>Property on <code>PhenoMeResults</code></em></p>

<div class="api-body">

True if at least one image has computed properties.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-image_name">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-image_name"><code>image_name</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.image_name(
    self,
    idx: 'int'
) -> str
```

</div>

<div class="api-body">

Return the basename of the primary path for image *idx*.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-items">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-items"><code>items</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.items(
    self
) -> Iterator[tuple[str, Any]]
```

</div>

<div class="api-body">

Yield ``(key, value)`` pairs for each valid key.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-keys">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-keys"><code>keys</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.keys(
    self
) -> Iterable[str]
```

</div>

<div class="api-body">

Return the valid dict-style keys for this results container.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-metadata_keys">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenomeresults-metadata_keys"><code>metadata_keys</code></h4>
</div>

<p><em>Property on <code>PhenoMeResults</code></em></p>

<div class="api-body">

Sorted unique metadata keys across all images (case-preserving).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-metadata_value">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-metadata_value"><code>metadata_value</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.metadata_value(
    self,
    idx: 'int',
    key: 'str'
) -> Any
```

</div>

<div class="api-body">

Case-insensitive metadata lookup for image *idx*.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-n_images">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenomeresults-n_images"><code>n_images</code></h4>
</div>

<p><em>Property on <code>PhenoMeResults</code></em></p>

<div class="api-body">

Number of images stored.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-primary_path">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-primary_path"><code>primary_path</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.primary_path(
    self,
    idx: 'int'
) -> str
```

</div>

<div class="api-body">

Return the primary (or only) file path for image *idx*.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-property_keys">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-phenomeresults-property_keys"><code>property_keys</code></h4>
</div>

<p><em>Property on <code>PhenoMeResults</code></em></p>

<div class="api-body">

Sorted list of property names from the first non-empty properties dict.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-property_value">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-property_value"><code>property_value</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.property_value(
    self,
    idx: 'int',
    key: 'str'
) -> float | None
```

</div>

<div class="api-body">

Return a scalar property for image *idx*, or None if absent.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-rebase_paths">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-rebase_paths"><code>rebase_paths</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.rebase_paths(
    self,
    data_dir: 'str',
    old_data_dir: 'str | None' = None
) -> None
```

</div>

<div class="api-body">

Update all image paths by replacing *old_data_dir* with *data_dir*.

This is useful when moving a checkpoint and its dataset to a different
machine or directory.

**Parameters:**

- **`data_dir`** (`str`):
  The new base directory where the images are located.
- **`old_data_dir`** (`str, optional`):
  The old base directory to be replaced. If not provided, it is
  automatically detected by finding the common prefix of all
  stored paths.  Special value "relative" indicates that stored
  paths are already relative and just need to be joined with
  *data_dir*.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeresults-values">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeresults-values"><code>values</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeResults.values(
    self
) -> Iterator[Any]
```

</div>

<div class="api-body">

Yield values for each valid key.

</div>

</div>
