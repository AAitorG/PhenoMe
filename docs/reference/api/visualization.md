# Visualization Methods

Interactive plotting methods powered by Plotly, available on `PhenoMe` instances.

**Related:** [Core Concepts: Visualization](../../concepts.md#visualization-methods) · [Distances](distances.md) · [Interactive Explorer](interactive.md) · [Workflows](../../workflows.md)

---

### Dimensionality Reduction

#### `plot_pca`

```python
plot_pca(
    n_components: int = 2,
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    exclude: Optional[Dict[str, Any]] = None,
    hover_features: Optional[List[str]] = None,
    return_fig: bool = False,
    render_mode: Literal['auto', 'svg', 'webgl'] = 'webgl',
    normalize: bool = True,
    sample_size: Optional[int] = None,
    **dr_kwargs: Any,
)
```
Plots a Principal Component Analysis (PCA) scatter plot.

**Parameters:**
- **n_components** (*int*) – Number of components (2 for 2D, 3 for 3D plot; PCA supports any value).
- **color_by** (*Optional[str]*) – Metadata or property key for color grouping.
- **source** (*Literal*) – Data source: `'embeddings'`, `'properties'`, or `'combined'`.
- **property_keys** (*Optional[List[str]]*) – Property subset when `source='properties'` or `'combined'`.
- **filters** (*Optional[Dict[str, Any]]*) – Metadata filters for selecting a subset of images.
- **exclude** (*Optional[Dict[str, Any]]*) – Metadata exclusions applied after `filters`.
- **figsize** (*Tuple[int, int]*) – Figure size in inches.
- **hover_features** (*Optional[List[str]]*) – Data keys to show on hover.
- **return_fig** (*bool*) – Return Plotly figure object instead of displaying.
- **normalize** (*bool*) – Normalize data before DR.
- **sample_size** (*Optional[int]*) – Randomly subsample points for large datasets (for faster plotting).
- **\*\*dr_kwargs** – Additional keyword arguments passed to the underlying PCA implementation
  (`sklearn.decomposition.PCA` on CPU or `torchdr.ExactIncrementalPCA` on GPU when `use_gpu_for_dr=True`, e.g.
  `pca_batch_size` for batch size, default 4096). See the scikit-learn and TorchDR documentation for the complete list
  of options.

**Backend:** Uses scikit-learn's `PCA` (`sklearn.decomposition.PCA`) on CPU or TorchDR's `ExactIncrementalPCA`
(`torchdr.ExactIncrementalPCA`) on GPU when acceleration is enabled. GPU PCA processes data in batches to reduce
memory usage.

---

#### `plot_tsne`

```python
plot_tsne(
    n_components: int = 2,
    perplexity: float = 30.0,
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    exclude: Optional[Dict[str, Any]] = None,
    hover_features: Optional[List[str]] = None,
    return_fig: bool = False,
    render_mode: Literal['auto', 'svg', 'webgl'] = 'webgl',
    normalize: bool = True,
    sample_size: Optional[int] = None,
    **dr_kwargs: Any,
)
```
Plots t-SNE scatter plot.

**Parameters:**
- **n_components** (*int*) – Number of components (2 for 2D, 3 for 3D; t-SNE supports any value; sklearn uses `method='exact'` when >= 4).
- **perplexity** (*float*) – The number of nearest neighbors used in manifold learning.
- **color_by** (*Optional[str]*) – Metadata or property key for color grouping.
- **source** (*Literal*) – Data source: `'embeddings'`, `'properties'`, or `'combined'`.
- **property_keys** (*Optional[List[str]]*) – Property subset when `source='properties'` or `'combined'`.
- **filters** (*Optional[Dict[str, Any]]*) – Metadata filters for selecting a subset of images.
- **exclude** (*Optional[Dict[str, Any]]*) – Metadata exclusions applied after `filters`.
- **figsize** (*Tuple[int, int]*) – Figure size in inches.
- **hover_features** (*Optional[List[str]]*) – Data keys to show on hover.
- **return_fig** (*bool*) – Return Plotly figure object instead of displaying.
- **render_mode** (*Literal*) – Plotly render mode (`'auto'`, `'svg'`, or `'webgl'`).
- **normalize** (*bool*) – Normalize data before DR.
- **sample_size** (*Optional[int]*) – Randomly subsample points for large datasets.
- **\*\*dr_kwargs** – Additional keyword arguments passed to the underlying t-SNE implementation
  (`sklearn.manifold.TSNE` on CPU or `torchdr.TSNE` on GPU when `use_gpu_for_dr=True`, e.g.
  `learning_rate`, `n_iter`, `metric`, `init`, `angle`). See the scikit-learn and TorchDR
  documentation for full details.

**Backend:** Uses scikit-learn's `TSNE` (`sklearn.manifold.TSNE`) on CPU or TorchDR's `TSNE`
(`torchdr.TSNE`) on GPU when acceleration is enabled. With `pykeops` installed, GPU t-SNE uses
the KeOps backend for linear memory (avoids O(n²) OOM on large datasets).

---

#### `plot_umap`

```python
plot_umap(
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    property_keys: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    exclude: Optional[Dict[str, Any]] = None,
    hover_features: Optional[List[str]] = None,
    return_fig: bool = False,
    render_mode: Literal['auto', 'svg', 'webgl'] = 'webgl',
    normalize: bool = True,
    sample_size: Optional[int] = None,
    **dr_kwargs: Any,
)
```
Plots UMAP scatter plot. Requires `umap-learn`.

**Parameters:**
- **n_components** (*int*) – Number of components (2 for 2D, 3 for 3D; UMAP supports any value).
- **n_neighbors** (*int*) – Size of local neighborhood.
- **min_dist** (*float*) – Minimum distance between embedded points.
- **color_by** (*Optional[str]*) – Metadata or property key for color grouping.
- **source** (*Literal*) – Data source: `'embeddings'`, `'properties'`, or `'combined'`.
- **property_keys** (*Optional[List[str]]*) – Property subset when `source='properties'` or `'combined'`.
- **filters** (*Optional[Dict[str, Any]]*) – Metadata filters for selecting a subset of images.
- **exclude** (*Optional[Dict[str, Any]]*) – Metadata exclusions applied after `filters`.
- **figsize** (*Tuple[int, int]*) – Figure size in inches.
- **hover_features** (*Optional[List[str]]*) – Data keys to show on hover.
- **return_fig** (*bool*) – Return Plotly figure object instead of displaying.
- **render_mode** (*Literal*) – Plotly render mode (`'auto'`, `'svg'`, or `'webgl'`).
- **normalize** (*bool*) – Normalize data before DR.
- **sample_size** (*Optional[int]*) – Randomly subsample points for large datasets.
- **\*\*dr_kwargs** – Additional keyword arguments passed to the underlying UMAP implementation.
  On CPU this is `umap.UMAP` from the `umap-learn` package (e.g. `metric`, `spread`,
  `negative_sample_rate`); on GPU it uses `torchdr.UMAP` when `use_gpu_for_dr=True` and a CUDA
  device is available. Only parameters supported by the respective backend are forwarded.

**Backend:** Uses `umap-learn` (`umap.UMAP`) on CPU or TorchDR's `UMAP` (`torchdr.UMAP`) on GPU
when available. With `pykeops` installed, GPU UMAP prefers the KeOps backend for linear memory;
falls back to FAISS or raw PyTorch if KeOps is unavailable.

---

### Analysis Plots

#### `plot_distance_distribution`

```python
plot_distance_distribution(distance_results: dict, group_by: Optional[str] = None, dist_range: tuple = (0, 100), kde: bool = False, figsize: Tuple[int, int] = (8, 5))
```
Plots a histogram or KDE of computed distances grouped by a metadata key.

**Parameters:**
- **distance_results** (*dict*) – Output from [`compute_reference_distances()`](distances.md#compute_reference_distances).
- **group_by** (*Optional[str]*) – Metadata key to split distributions.
- **dist_range** (*tuple*) – Limit the X axis range.
- **kde** (*bool*) – Show continuous KDE instead of histogram.

---

#### `plot_property_correlations`

```python
plot_property_correlations(correlation_results: Dict[str, Any], top_k: int = 20, figsize: Tuple[int, int] = (10, 8), title: str = "Property Correlations with Embeddings")
```
Plots a horizontal bar chart of the highest correlated properties.

**Parameters:**
- **correlation_results** (*Dict*) – Output from [`compute_embedding_property_correlations()`](pipeline.md#compute_embedding_property_correlations).
- **top_k** (*int*) – Limit bar chart to the top K properties.

---

#### `plot_multivariate_interpretability`

```python
plot_multivariate_interpretability(
    results: Dict[str, Any],
    *,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    return_fig: bool = False
)
```
Plots a horizontal bar chart of the LASSO coefficients or Random Forest importances explaining an embedding axis.

**Parameters:**
- **results** (*Dict*) – **Required.** Output from [`compute_multivariate_interpretability()`](pipeline.md#compute_multivariate_interpretability). The plot does not run the model; it only visualizes this dict (same pattern as [`plot_property_correlations()`](visualization.md#plot_property_correlations)).
- **top_k** (*int*) – Number of top drivers to show.
- **figsize** (*Tuple*) – Figure size in pixels / 100.
- **return_fig** (*bool*) – Return Plotly figure object.

**Example:**
```python
mv = pheno.compute_multivariate_interpretability(method="tsne", component=1, model_type="lasso")
pheno.plot_multivariate_interpretability(mv, top_k=15)
```

---

#### `plot_centroids`

```python
plot_centroids(group_by: Union[str, List[str]], method: Literal['pca', 'tsne', 'umap'] = 'pca', n_components: int = 2, filters: Optional[Dict] = None, exclude: Optional[Dict] = None, source: str = 'embeddings', property_keys: Optional[List[str]] = None, show_points: bool = True, show_centroids: bool = True, figsize: Tuple[int, int] = (10, 8), return_fig: bool = False, normalize: bool = True, **dr_kwargs)
```
Plots group centroids in reduced embedding space (PCA/t-SNE/UMAP).

**Parameters:**
- **group_by** (*Union[str, List[str]]*) – Metadata key(s) defining centroid groups (e.g. `'drug'`, `['drug', 'cell_line']`).
- **method** (*Literal['pca', 'tsne', 'umap']*) – Dimensionality reduction method.
- **n_components** (*int*) – Number of components (2 or 3).
- **filters** (*Optional[Dict]*) – Metadata filters.
- **exclude** (*Optional[Dict]*) – Metadata values to exclude.
- **source** (*str*) – `'embeddings'`, `'properties'`, or `'combined'`.
- **property_keys** (*Optional[List[str]]*) – Property subset when using properties.
- **show_points** (*bool*) – Whether to show individual data points.
- **show_centroids** (*bool*) – Whether to show centroid markers.
- **figsize** (*Tuple[int, int]*) – Figure size in inches.
- **return_fig** (*bool*) – Return Plotly figure instead of displaying.
- **normalize** (*bool*) – Normalize before dimensionality reduction.
- **\*\*dr_kwargs** – Passed to the DR method (e.g. `perplexity`, `n_neighbors`).

**Examples:**
```python
pheno.plot_centroids(group_by='class', method='tsne')

# One centroid per (drug, time) combination
pheno.plot_centroids(group_by=['drug', 'time'], method='pca')
```

---

#### `plot_counts`

```python
plot_counts(group_by: Optional[List[str]] = None)
```
Plots a bar chart (1 grouping) or heatmap (2 groupings) of image counts.

**Parameters:**
- **group_by** (*Optional[List[str]]*) – 1 or 2 metadata keys.

---

### Image Display

#### `plot_image_by_index`

```python
plot_image_by_index(idx: int, distance_results: Optional[dict] = None, channels: Optional[Union[int, list]] = None, figsize: Tuple[int, int] = (5, 5), title_fields: Optional[List[str]] = None, show_extra_info: bool = True)
```
Displays an image from the dataset using Matplotlib.

**Parameters:**
- **idx** (*int*) – Image index in the dataset.
- **distance_results** (*Optional[dict]*) – Results dictionary to include distance info.
- **channels** (*Optional[Union[int, list]]*) – Sub-select image channels.
- **title_fields** (*Optional[List[str]]*) – Properties to show in the figure title.
- **show_extra_info** (*bool*) – Print the full metadata/properties block below the image.
