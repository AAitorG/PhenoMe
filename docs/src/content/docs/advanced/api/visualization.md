---
title: "Visualization"
description: "PCA, t-SNE, UMAP, plots, distance distributions, correlations."
editUrl: false
tableOfContents:
  maxHeadingLevel: 3
---

<p><span class="api-tier api-tier--public">Tier: Public API</span></p>

:::note[Auto-generated]
This page is rebuilt from docstrings in [`phenome.mixins.visualization._core`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/mixins/visualization/_core.py) (class `PhenoMeVisualization`).
:::

**See also:** [Pipeline](/PhenoMe/advanced/api/pipeline/) · [Distances](/PhenoMe/advanced/api/distances/) · [Concepts](/PhenoMe/concepts/)

Pipeline class providing visualization methods for PhenoMe.

**Module Breakdown:**
- `_dr_plots.py`: Dimensionality reduction visualizations (PCA, t-SNE, UMAP) and centroid plots.
- `_distance_plots.py`: Distance distributions, reference comparison plots, and property correlations.
- `_image_display.py`: Raw and transformed image visualization with channel-wise controls.
- `_interpretability_plots.py`: Visualizing feature importance and drivers for embedding axes.
- `_component_correlation_plots.py`: Faceted plots of correlations between embeddings and properties.
- `_group_enrichment_plots.py`: Faceted Z-score enrichment plots for metadata groups.
- `_helpers.py`: Shared utilities for Plotly and Matplotlib layout/styling.

**Interdependencies:**
The following private plot functions are intended for use by `PhenoMeAnalysis`:
- `_plot_property_correlations_plotly` (from `_distance_plots.py`): Used for embedding-property correlation analysis.
- `_display_multivariate_interpretability` (from `_interpretability_plots.py`): Used for explaining embedding axes.

## Visualization

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

<div class="api-method" role="region" aria-labelledby="api-phenomevisualization-plot_counts">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomevisualization-plot_counts"><code>plot_counts</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeVisualization.plot_counts(
    self,
    group_by: list[str] | None = None,
    return_fig: bool = False
) -> Any
```

</div>

<div class="api-body">

Plot count of images grouped by metadata using Plotly.

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
    distance_results: dict | None = None,
    channels: int | list | None = None,
    figsize: tuple = (6, 6),
    title_fields: list[str] | None = None,
    show_extra_info: bool = True,
    apply_transforms: bool = True,
    downsample: int | None = 720,
    ax: Any = None,
    return_fig: bool = False
) -> Any
```

</div>

<div class="api-body">

Plot a specific image by its index.

**Args:**

- **`idx`**: Image index in results
- **`distance_results`**: Optional distance computation results
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

</div>

</div>

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
    distance_results: dict | None = None,
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

<div class="api-method" role="region" aria-labelledby="api-_distanceplotsmixin-print_distance_summary">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-_distanceplotsmixin-print_distance_summary"><code>print_distance_summary</code></h4>
</div>

<div class="api-signature">

```python
_DistancePlotsMixin.print_distance_summary(
    self,
    distance_results: dict,
    group_by: str | None = None,
    dist_range: tuple = (0, 100)
) -> None
```

</div>

<div class="api-body">

Print distance summary statistics without plotting. Safe to use when enable_plots=False.

**Args:**

- **`distance_results`**: Dictionary containing 'distances' array
- **`group_by`**: Metadata key to group by
- **`dist_range`**: Tuple of (min, max) distances to include

</div>

</div>
