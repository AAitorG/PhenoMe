---
title: "Report generation"
description: ReportConfig and generate_report standalone API.
---

Auto-generated from `phenome.report.generator`. Rebuild with `npm run prebuild` in `docs/`.

**See also:** [Pipeline generate_report](pipeline.md)

Main report generator module.

Assembles the complete HTML report from individual section generators.

Author: Aitor González-Marfil (@AAitorG)

## Report configuration and generation

### `class ReportConfig`

## `ReportConfig`

Configuration for report generation. Pass to generate_report via config=.

Use ReportConfig for full control, or pass individual overrides to
generate_report(..., **overrides).

Attributes:
    include_plots: Include interactive PCA, t-SNE, and UMAP plots.
    include_distance_analysis: Run and visualize distances to the reference group.
    include_outliers: Detect outliers via Z-score and list them.
    include_correlations: List properties correlated with embeddings.
    include_property_stats: Export aggregated property statistics tables.
    include_clustering: Show GMM clusters and prototype images.
    include_image_gallery: Render the base64 thumbnails gallery.
    reference_filters: Dict mapping metadata keys to values for the reference
        group in distance analysis. Example: &#123;'condition': 'Control'&#125;.
    top_k_features: Number of top correlated features to show.
    overview_metadata_keys: Metadata keys for the overview section.
    metadata_keys: Metadata keys for plots and sections. If None, uses all.
    color_by: Metadata key for coloring points in plots. If None, uses first.
    outlier_threshold: Z-score threshold for outlier detection.
    outlier_group_by: Metadata key to group by when detecting outliers.
    n_clusters: Number of clusters for GMM/kmeans clustering.
    clustering_method: 'kmeans', 'dbscan', or 'gmm'.
    n_cluster_prototypes: Number of prototype images per cluster.
    clustering_reduce_dim: Max PCA dim before clustering. None = no reduction.
    clustering_reduce_method: 'pca', 'tsne', or 'umap' for pre-clustering.
    n_gallery_images: Number of images per gallery category.
    image_size_px: Thumbnail size in pixels for gallery and prototypes.
    property_group_by: Metadata key for grouping property statistics.
    property_table_max_rows: Max rows in property stats table.
    lite_mode: If True, optimizes for large datasets (sampling, WebGL).
    description: Optional custom description for the report.
    filters: Pre-report metadata filters applied to all results.
    exclude: Pre-report metadata exclusions (same structure as filters).

**Fields:**

- **`clustering_method`** (`typing.Literal['kmeans', 'dbscan', 'gmm']`):
- **`clustering_reduce_dim`** (`int | None`):
- **`clustering_reduce_method`** (`typing.Literal['pca', 'tsne', 'umap']`):
- **`color_by`** (`str | None`):
- **`description`** (`str | None`):
- **`exclude`** (`dict[str, typing.Any] | None`):
- **`filters`** (`dict[str, typing.Any] | None`):
- **`image_size_px`** (`<class 'int'>`):
- **`include_clustering`** (`<class 'bool'>`):
- **`include_correlations`** (`<class 'bool'>`):
- **`include_distance_analysis`** (`<class 'bool'>`):
- **`include_image_gallery`** (`<class 'bool'>`):
- **`include_interpretability`** (`<class 'bool'>`):
- **`include_outliers`** (`<class 'bool'>`):
- **`include_plots`** (`<class 'bool'>`):
- **`include_property_stats`** (`<class 'bool'>`):
- **`interpretability_model_type`** (`typing.Literal['lasso', 'random_forest']`):
- **`lite_mode`** (`<class 'bool'>`):
- **`metadata_keys`** (`list[str] | None`):
- **`n_cluster_prototypes`** (`<class 'int'>`):
- **`n_clusters`** (`<class 'int'>`):
- **`n_gallery_images`** (`<class 'int'>`):
- **`outlier_group_by`** (`str | None`):
- **`outlier_threshold`** (`<class 'float'>`):
- **`overview_metadata_keys`** (`list[str] | None`):
- **`property_group_by`** (`str | None`):
- **`property_table_max_rows`** (`<class 'int'>`):
- **`reference_filters`** (`dict[str, typing.Any] | None`):
- **`top_k_features`** (`<class 'int'>`):

### `generate_report`

```python
generate_report(
    pipeline: 'PhenoMe',
    output_path: str = 'pheno_report.html',
    title: str = 'PhenoMe Analysis Report',
    config: phenome.report.generator.ReportConfig | None = None,
    **overrides: Any
) -> str
```

Generate comprehensive standalone HTML report from phenotyping results.

Uses :class:`ReportConfig` as the primary source of options. Pass ``config=``
for full control, or use ``**overrides`` to tweak individual settings.

**Args:**

- **`pipeline`**: Processed PhenoMe instance with results.
- **`output_path`**: Path to save the HTML file.
- **`title`**: Report title displayed at the top.
- **`config`**: Optional ReportConfig for defaults. If None, uses ReportConfig().
  **overrides: Any ReportConfig field to override. Examples:
  include_plots=False, outlier_threshold=2.5, n_clusters=10,
- **`reference_filters=&#123;'condition'`**: 'Control'}, lite_mode=True.

**Returns:**

- **`str`**: Path to the generated HTML file (output_path). File is written to disk.

**Example:**

```python
>>> pheno.generate_report("report.html")
>>> pheno.generate_report("report.html", include_clustering=False, n_clusters=8)
>>> from phenome.report import generate_report
>>> from phenome.report.generator import ReportConfig
>>> cfg = ReportConfig(outlier_threshold=2.5, lite_mode=True)
>>> generate_report(pheno, config=cfg, title="My Report")

```
