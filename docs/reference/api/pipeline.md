# PhenoMe API Reference

The main orchestrator class for processing images with deep learning models and analyzing phenotypic differences via embedding distances. It inherits from [`PhenoMeProperties`](properties.md), `PhenoMeAnalysis`, [`PhenoMeDistances`](distances.md), and [`PhenoMeVisualization`](visualization.md).

**Related:** [Getting Started](../../getting-started.md) · [Core Concepts](../../concepts.md) · [Common Workflows](../../workflows.md) · [Utilities](utilities.md) · [Minimal external checkpoints](../../guides/external-checkpoints.md)

---

#### `PhenoMe`

```python
PhenoMe(device: Optional[torch.device] = None, seed: Optional[int] = None, use_gpu_for_dr: bool = False)
```
Initializes the PhenoMe framework.

**Parameters:**
- **device** (*Optional[torch.device]*) – Device for computation (CPU or CUDA). Defaults to single GPU when available.
- **seed** (*Optional[int]*) – Random seed for reproducibility across random operations (t-SNE, clustering, sampling, etc).
- **use_gpu_for_dr** (*bool*) – If `True` and device is CUDA, uses `torchdr` for GPU-accelerated dimensionality reduction. Defaults to `False`.

---

### Data Processing

#### `set_file_df`

```python
set_file_df(file_df: pd.DataFrame) -> pd.DataFrame
```
Stores a file DataFrame for processing. Call this when you have a pre-built DataFrame (e.g. from a CSV or custom script). For directory-based discovery, use `find_files` instead.

**Parameters:**
- **file_df** (*pd.DataFrame*) – DataFrame with `file_path` column (and optionally `mask_path`).

**Returns:**
- *pd.DataFrame* – The normalized DataFrame that was stored.

---

#### `find_files`

```python
find_files(image_dir: str, mask_dir: Optional[str] = None, extensions: Optional[List[str]] = None, metadata_fn: Optional[Union[Callable, MetadataBase]] = None, on_missing_metadata: str = 'drop', mask_filename_column: Optional[str] = None, mask_extensions: Optional[List[str]] = None) -> pd.DataFrame
```
Discovers image and mask files from directories, extracts metadata, and stores the result for use by the framework. Use this as the primary way to load data from disk.

**Parameters:**
- **image_dir** (*str*) – Root directory or glob pattern for image files.
- **mask_dir** (*Optional[str]*) – Root directory for mask files (adds `mask_path` column).
- **extensions** (*Optional[List[str]]*) – File-extension filter (e.g. `[".tif", ".png"]`).
- **metadata_fn** (*Optional[Union[Callable, MetadataBase]]*) – Per-file metadata extractor function or a `MetadataBase` instance (e.g., `DataFrameMetadata`).
- **on_missing_metadata** (*str*) – `'drop'` or `'keep'` for files with missing metadata.
- **mask_filename_column** (*Optional[str]*) – Metadata column for custom mask filename.
- **mask_extensions** (*Optional[List[str]]*) – Extensions to try when exact mask path fails.

**Returns:**
- *pd.DataFrame* – DataFrame with `file_path`, `mask_path` (when mask_dir set), and metadata. Use this for `process_images` and `compute_properties`. `load_results` uses the file list from `find_files` to resolve paths.

---

#### `process_images`

```python
process_images(model_wrapper: ModelWrapper, batch_size: int = 32, num_workers: int = 4, filters: Optional[Dict[str, List[Any]]] = None, exclude: Optional[Dict[str, List[Any]]] = None, channel_mode: str = 'split', channels: Optional[List[int]] = None, preprocessing_fn: Optional[Callable] = None, custom_transformations: Optional[Any] = None, append: bool = False, resize_size: int = 224, pad_size: Optional[int] = None, checkpoint_path: Optional[str] = None, force_rgb: bool = True, save_every: int = 5, lazy_checkpoint: bool = True)
```
Processes images through the vision model to extract embeddings. Call `find_files` (or `set_file_df`) first.

**Notes:** Automatically calls `reset()` before processing unless `append=True`. Use `checkpoint_path` for crash-safe, incremental processing on large datasets.

**Parameters:**
- **model_wrapper** ([*ModelWrapper*](model-wrapper.md)) – Vision model wrapper for feature extraction.
- **batch_size** (*int*) – Inference batch size.
- **num_workers** (*int*) – DataLoader workers.
- **filters** (*Optional[Dict]*) – Metadata filters to subset images.
- **exclude** (*Optional[Dict]*) – Metadata exclusions (same structure as `filters`). Applied after filters.
- **channel_mode** (*str*) – `'split'` or `'combined'`.
- **channels** (*Optional[List[int]]*) – Channel indices to use (0-based). If None, all channels are used.
- **preprocessing_fn** (*Optional[Callable]*) – Custom image preprocessing function.
- **custom_transformations** (*Optional[Any]*) – Replaces the default transform pipeline (resize, normalize).
- **append** (*bool*) – If `True`, appends results to existing ones.
- **resize_size** (*int*) – Target size for resizing images.
- **pad_size** (*Optional[int]*) – Target size for padding smaller images.
- **checkpoint_path** (*Optional[str]*) – HDF5 path to save/resume progress incrementally.
- **force_rgb** (*bool*) – Replicates 1-channel images to 3 channels.
- **save_every** (*int*) – Checkpoint save frequency (in batches).
- **lazy_checkpoint** (*bool*) – If True (default), keep the checkpoint file open for lazy loading. If False, load all data into RAM and close the file.

---

#### `process_temporal_images`

```python
process_temporal_images(model_wrapper: ModelWrapper, files: Union[str, List[str], pd.DataFrame], batch_size: int = 32, num_workers: int = 4, channel_mode: Optional[Literal['split', 'combined']] = None, channels: Optional[List[int]] = None, preprocessing_fn: Optional[Callable] = None, custom_transformations: Optional[Any] = None, resize_size: Optional[int] = None, pad_size: Optional[int] = None, force_rgb: Optional[bool] = None, extensions: Optional[List[str]] = None) -> None
```
Processes new images **in-memory** (temporary) and appends them to the current session. Use this to explore additional images (e.g., from a new condition or replicate) without re-running `process_images`. Temporal images are **not** persisted to checkpoint; visualize or export before calling `clear_temporal_data()` or `reset()`.

**Prerequisites:** Run `process_images()` first to establish the base dataset.

**Parameters:**
- **model_wrapper** ([*ModelWrapper*](model-wrapper.md)) – Vision model wrapper for feature extraction.
- **files** (*Union[str, List[str], pd.DataFrame]*) – Image source: a single file path, directory path (uses `find_files`), list of paths, or DataFrame from `find_files()`.
- **batch_size** (*int*) – Inference batch size.
- **num_workers** (*int*) – DataLoader workers.
- **channel_mode** (*Optional[Literal]*) – `'split'` or `'combined'`. Defaults to the value from the initial `process_images` run.
- **channels** (*Optional[List[int]]*) – Channel indices. Defaults to the value from the initial run.
- **preprocessing_fn** (*Optional[Callable]*) – Custom image preprocessing.
- **custom_transformations** (*Optional[Any]*) – Override default transforms.
- **resize_size** (*Optional[int]*) – Target size for resizing.
- **pad_size** (*Optional[int]*) – Target size for padding.
- **force_rgb** (*Optional[bool]*) – Force 3-channel output.
- **extensions** (*Optional[List[str]]*) – File extensions when `files` is a directory (e.g., `['.tif', '.png']`).

**Notes:** Temporal rows get `metadata['source'] = 'NEW'` and `"Temporal"` for missing metadata keys so they work with filters and plots. Processing parameters (channel_mode, channels, resize_size, etc.) are inherited from the base run when not specified.

---

#### `clear_temporal_data`

```python
clear_temporal_data() -> int
```
Removes all temporal images (rows with `metadata['source'] == 'NEW'`) from the current session.

**Returns:**
- *int* – Number of temporal rows removed.

---

#### `inspect_data`

```python
inspect_data() -> pd.DataFrame
```
Audits image and mask dimensions, shapes, and data ranges. Call `find_files` (or `set_file_df`) first. When `mask_path` is present, validates image-mask shape matching.

**Returns:**
- *pd.DataFrame* – DataFrame with columns: file_path, height, width, channels,
  min, max, error, shape_match (True/False/None), mask_height, mask_width,
  mask_dtype, mask_min, mask_max, mask_error.

---

### Info / Accessors

#### `get_image_info`

```python
get_image_info(idx: int, distance_results: Optional[dict] = None) -> dict
```
Returns metadata, properties, and optional distance for image at index `idx`.

**Parameters:**
- **idx** (*int*) – Image index (0 to n_images-1).
- **distance_results** (*Optional[dict]*) – Optional output from `compute_reference_distances()` to include distance and is_reference.

**Returns:**
- *dict* – Keys: idx, img_name, img_path, metadata keys, property keys, distance (if provided), is_reference (if applicable).

---

#### `get_available_metadata_keys`

```python
get_available_metadata_keys() -> List[str]
```
Returns sorted list of metadata keys present in the dataset. Useful for debugging filters and understanding available grouping columns.

---

#### `get_embeddings`

```python
get_embeddings(indices: Optional[Union[List[int], np.ndarray]] = None) -> Optional[np.ndarray]
```
Returns embeddings for the given row indices. When using checkpoint/lazy loading, loads from HDF5 on demand. When `indices` is `None`, returns all embeddings. Returns `None` if no embeddings are available.

---

### Results container: PhenoMeResults

`pheno.results` is a `PhenoMeResults` instance. Import: `from phenome import PhenoMeResults`.

**Constructor:** `PhenoMeResults(img_path=None, metadata=None, properties=None, embeddings=None)`

**Attributes:** `img_path`, `metadata`, `properties`, `embeddings` (aligned by row index).

**Properties:** `n_images`, `has_embeddings`, `has_properties`, `property_keys`, `metadata_keys`, `embedding_dim`.

**Dict-like:** `results['metadata']`, `results.keys()`, `results.get('metadata')`. Valid keys: `img_path`, `metadata`, `properties`, `embeddings`.

**Helpers:** `primary_path(idx)`, `image_name(idx)`, `metadata_value(idx, key)`, `property_value(idx, key)`, `rebase_paths(data_dir, old_data_dir=None)`.

Use `get_embeddings()` for lazy-safe access when using checkpoints. See [Core Concepts: Embedding Lifecycle](../../concepts.md#embedding-lifecycle-eager-vs-lazy).

---

### Data Management

#### `reset`

```python
reset(verbose: bool = False, clear_file_df: bool = False)
```
Clears all stored data (embeddings, metadata, properties). If `clear_file_df` is `True`, also removes the loaded file paths (from `find_files`).

---

#### `reset_properties`

```python
reset_properties(verbose: bool = False)
```
Clears only the computed properties.

---

#### `transfer_metadata_to_properties`

```python
transfer_metadata_to_properties(columns: Union[str, List[str]])
```
Transfers specified metadata columns to properties, copying the values so they are available as computed properties.

**Parameters:**
- **columns** (*Union[str, List[str]]*) – A single metadata column name or a list of names to transfer.

---

#### `save_results`

```python
save_results(output_dir: Optional[str] = None, filename: Optional[str] = None, compression: str = "gzip")
```
Saves the complete framework state (embeddings, metadata, properties) to an HDF5 file.

**Parameters:**
- **output_dir** (*Optional[str]*) – Directory to save `phenome_results.h5`.
- **filename** (*Optional[str]*) – Custom filepath.
- **compression** (*str*) – Compression algorithm for datasets (default `"gzip"`).

---

#### `load_results`

```python
load_results(filename: str, lazy_checkpoint: bool = True)
```
Loads framework state from an HDF5 file. Metadata and properties are loaded immediately. Embeddings are either loaded into RAM (`lazy_checkpoint=False`) or accessed on-demand from disk (`lazy_checkpoint=True`). Uses the file list from :meth:`find_files` or :meth:`set_file_df` to resolve paths. Call :meth:`find_files` first before loading.

**Parameters:**
- **filename** (*str*) – Path to the `.h5` checkpoint or result file.
- **lazy_checkpoint** (*bool*) – If `True` (default), keep the checkpoint open for lazy loading. If `False`, load all data into RAM and close the file.

For guaranteed HDF5 handle cleanup when moving, deleting, or opening multiple checkpoints in sequence, use [`checkpoint_context`](#checkpoint_context) instead.

---

#### `checkpoint_context`

```python
checkpoint_context(path: str, lazy_checkpoint: bool = True) -> Generator[PhenoMe, None, None]
```
Context manager that loads a checkpoint and **guarantees the HDF5 file handle is closed on exit**. Prefer over `load_results` alone when you need to ensure the file handle is released (e.g., before moving/deleting the file or opening multiple checkpoints).

**Example:**
```python
pheno.find_files("path/to/images")
with pheno.checkpoint_context("results.h5") as p:
    p.plot_pca(color_by="condition")
# Checkpoint closed here; safe to move or delete results.h5
```

---

#### `export_experiment_config`

```python
export_experiment_config(path: str, reference_filters: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None, checkpoint_path: Optional[str] = None, **extra: Any) -> None
```
Exports experiment configuration to JSON for reproducibility. Saves parameters not stored in the HDF5 checkpoint (seed, use_gpu_for_dr, reference_filters, model name). Call after `save_results()` with the same `reference_filters` used for distance analysis.

**Parameters:**
- **path** (*str*) – Output path for config.json.
- **reference_filters** (*Optional[Dict]*) – Dict used for `compute_reference_distances` (e.g. `{'condition': 'Control'}`).
- **model_name** (*Optional[str]*) – Model identifier (e.g. `'dinov2_vitb14_reg'`).
- **checkpoint_path** (*Optional[str]*) – Checkpoint path used during processing.
- **\*\*extra** – Additional key-value pairs to include in the config.

---

#### `export_dataset_table`

```python
export_dataset_table(output_path: Optional[str] = None, dist_results: Optional[Dict[str, Any]] = None, include_embeddings: Union[bool, Literal["separate"]] = False, export_format: Literal["csv", "parquet", "excel"] = "csv") -> pd.DataFrame
```
Exports dataset metadata, properties, and distances as a table.

**Parameters:**
- **output_path** (*Optional[str]*) – Filepath to save export. If `None`, returns DataFrame only.
- **dist_results** (*Optional[Dict]*) – Optional results from `compute_reference_distances()`.
- **include_embeddings** (*Union[bool, Literal["separate"]]*) – Include embeddings in table (`True`), separate `.npy` (`"separate"`), or not at all (`False`).
- **export_format** (*Literal*) – Export format (`"csv"`, `"parquet"`, `"excel"`).

**Returns:**
- *pd.DataFrame* – Dataset table.

---

### Advanced Analysis

#### `compute_clustering`

```python
compute_clustering(source: Literal['embeddings', 'properties', 'combined'] = 'embeddings', n_clusters: int = 5, clustering_method: Literal['kmeans', 'dbscan', 'gmm'] = 'kmeans', property_keys: Optional[List[str]] = None, filters: Optional[Dict] = None, exclude: Optional[Dict] = None, random_state: Optional[int] = None, normalize: bool = True, reduce_dim: Optional[int] = 100, reduce_method: Literal['pca', 'tsne', 'umap'] = 'pca', return_silhouette: bool = False, dbscan_eps: Optional[float] = None, dbscan_min_samples: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, Optional[float]]]
```
Performs clustering and stores labels as the `'cluster'` metadata.

**Parameters:**
- **source** (*Literal*) – `'embeddings'`, `'properties'`, or `'combined'`.
- **n_clusters** (*int*) – Number of clusters (k-means, GMM). Ignored for `clustering_method='dbscan'`.
- **clustering_method** (*Literal*) – `'kmeans'` (default), `'dbscan'`, or `'gmm'`.
- **property_keys** (*Optional[List[str]]*) – Properties to use (if `source='properties'`).
- **filters** (*Optional[Dict]*) – Subset filters.
- **exclude** (*Optional[Dict]*) – Subset exclusions (same structure as filters).
- **random_state** (*Optional[int]*) – Seed for k-means/GMM.
- **normalize** (*bool*) – Normalize data before clustering.
- **reduce_dim** (*Optional[int]*) – Apply dimensionality reduction before clustering.
- **reduce_method** (*Literal*) – `'pca'`, `'tsne'`, or `'umap'`.
- **return_silhouette** (*bool*) – If True, return silhouette score as second value. Default: False.
- **dbscan_eps** (*Optional[float]*) – Max distance between samples for DBSCAN (only when `clustering_method='dbscan'`). Default 0.5.
- **dbscan_min_samples** (*Optional[int]*) – Min samples in neighborhood for DBSCAN (only when `clustering_method='dbscan'`).

**Returns:**
- *np.ndarray* or *Tuple[np.ndarray, Optional[float]]* – Cluster labels per image (NaN for filtered-out or DBSCAN noise). When `return_silhouette=True`, returns `(labels, silhouette_score)`; score is None if it could not be computed.

---

#### `detect_outliers`

```python
detect_outliers(method: Literal['z-score', 'iqr'] = 'z-score', threshold: float = 3.0, group_by: Optional[str] = None, filters: Optional[Dict] = None, metric: str = 'embeddings', property_keys: Optional[List[str]] = None) -> Dict[str, Any]
```
Detects outliers based on embedding distances or properties.

**Parameters:**
- **method** (*Literal*) – `'z-score'` or `'iqr'`.
- **threshold** (*float*) – Threshold multiplier (std devs or IQR).
- **group_by** (*Optional[str]*) – Metadata key to compute outliers per group.
- **filters** (*Optional[Dict]*) – Subset filters.
- **exclude** (*Optional[Dict]*) – Subset exclusions (same structure as filters).
- **metric** (*str*) – `'embeddings'` or `'properties'`.
- **property_keys** (*Optional[List[str]]*) – Specific properties to use.
- **source** (*Literal*) – `'embeddings'`, `'properties'`, or `'combined'`.

**Returns:**
- *Dict[str, Any]* – Details of detected outliers.

---

#### `compute_component_correlation`

```python
compute_component_correlation(method: Literal['pca', 'tsne', 'umap'] = 'pca', n_components: int = 2, source: str = 'embeddings', property_keys: Optional[List[str]] = None, filters: Optional[Dict] = None, top_k: Optional[int] = None, normalize: bool = True, correlation_method: Literal['pearson', 'spearman', 'distance_correlation', 'mutual_info'] = 'pearson') -> Dict[str, Any]
```
Computes correlation between DR components and scalar properties.

**Parameters:**
- **method** (*Literal*) – DR method (`'pca'`, `'tsne'`, `'umap'`).
- **n_components** (*int*) – Number of components.
- **source** (*str*) – Input data (`'embeddings'` or `'properties'`).
- **filters** (*Optional[Dict]*) – Subset filters.
- **exclude** (*Optional[Dict]*) – Subset exclusions (same structure as filters).
- **correlation_method** (*Literal*) – `'pearson'`, `'spearman'`, `'distance_correlation'`, or `'mutual_info'`.

**Returns:**
- *Dict[str, Any]* – Correlation matrix and summary.

---

#### `find_prototypes`

```python
find_prototypes(cluster_col: Optional[str] = 'cluster', n_prototypes: int = 5, source: str = 'embeddings', property_keys: Optional[List[str]] = None, filters: Optional[Dict] = None, metric: str = 'euclidean') -> Dict[str, List[int]]
```
Finds the most representative images (prototypes) for each cluster or group based on proximity to the centroid.

**Parameters:**
- **cluster_col** (*Optional[str]*) – Column containing group labels.
- **n_prototypes** (*int*) – Prototypes to return per group.
- **filters** (*Optional[Dict]*) – Subset filters.
- **exclude** (*Optional[Dict]*) – Subset exclusions (same structure as filters).
- **metric** (*str*) – `'euclidean'` or `'cosine'`.

**Returns:**
- *Dict[str, List[int]]* – Group names mapped to lists of prototype indices.

---

#### `analyze_cluster_enrichment`

```python
analyze_cluster_enrichment(cluster_col: str = 'cluster', property_keys: Optional[List[str]] = None, filters: Optional[Dict] = None, exclude: Optional[Dict] = None) -> pd.DataFrame
```
Analyzes which properties are statistically enriched (Z-score) in each cluster.

**Parameters:**
- **cluster_col** (*str*) – Column containing cluster labels.
- **property_keys** (*Optional[List[str]]*) – Properties to analyze.
- **filters** (*Optional[Dict]*) – Subset filters.
- **exclude** (*Optional[Dict]*) – Subset exclusions (same structure as filters).

**Returns:**
- *pd.DataFrame* – Enrichment scores per property and cluster.

---

#### `compute_embedding_property_correlations`

```python
compute_embedding_property_correlations(property_keys: Optional[List[str]] = None, normalize: bool = True, method: Literal['pearson', 'spearman', 'distance_correlation', 'mutual_info'] = 'pearson') -> Dict[str, Any]
```
Computes global correlation between embedding dimensions and extracted properties.

**Parameters:**
- **property_keys** (*Optional[List[str]]*) – Subset of properties.
- **normalize** (*bool*) – Normalizes embeddings (L2) and properties (StandardScaler) before correlation.
- **method** (*Literal*) – Correlation measure (`'pearson'`, `'spearman'`, `'distance_correlation'`, `'mutual_info'`).

**Returns:**
- *Dict[str, Any]* – Dictionary of raw correlations per dimension. Use [`aggregate_embedding_property_correlations()`](#aggregate_embedding_property_correlations) on the output to get a summarized ranking.

---

#### `aggregate_embedding_property_correlations`

```python
aggregate_embedding_property_correlations(correlation_results: Dict[str, Any], aggregation: str = 'mean_abs', top_k: Optional[int] = 20) -> Dict[str, Any]
```
Aggregates pre-computed embedding-property correlations using the specified method.

**Parameters:**
- **correlation_results** (*Dict[str, Any]*) – Output dict from `compute_embedding_property_correlations()`.
- **aggregation** (*str*) – Aggregation method (`'mean_abs'`, `'max_abs'`, `'mean'`, `'std'`).
- **top_k** (*Optional[int]*) – Return only top-k properties in summary (None = all).

**Returns:**
- *Dict[str, Any]* – Aggregated correlation summary.

---

### Reporting

#### `generate_report`

```python
def generate_report(self, output_path: str = "pheno_report.html", title: str = "PhenoMe Analysis Report", config: Any | None = None, **overrides: Any) -> str:
```
Generate comprehensive standalone HTML report from phenotyping results. Uses [`ReportConfig`](report.md) as the primary source of options.

**Parameters:**
- **output_path** (*str*) – Path to save the HTML file. Default: `"pheno_report.html"`.
- **title** (*str*) – Report title displayed at the top. Default: `"PhenoMe Analysis Report"`.
- **config** (*Optional[Any]*) – Optional `ReportConfig` for defaults. If None, uses `ReportConfig()`.
- **\*\*overrides** (*Any*) – Any `ReportConfig` field to override (e.g., `include_plots=False`).
**Returns:**
- *str* – Path to the HTML output file.
