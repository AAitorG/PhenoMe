# Report Generation

Methods to export a full PhenoMe analysis into a standalone HTML report.

**Related:** [Pipeline](pipeline.md) · [How to Read Your Report](../../guides/report-guide.md) · [Plugins: Report sections](../../guides/plugins.md#report-sections) · [Workflows](../../workflows.md) · [Utilities](utilities.md)

---

### Configuration: `ReportConfig` and overrides

`generate_report` uses a `ReportConfig` dataclass as the base for all options. You can:

- **Use defaults:** Call `pheno.generate_report("report.html")` to use `ReportConfig()` defaults.
- **Override specific fields:** Pass keyword arguments, e.g. `generate_report("report.html", include_clustering=False, n_clusters=8)`.
- **Use a config object:** Create `config=ReportConfig(...)` for full control; `**overrides` passed to `generate_report` take precedence over matching fields in `config`.

```python
from phenome.report import ReportConfig

# Override individual options
pheno.generate_report("report.html", outlier_threshold=2.5, lite_mode=True)

# Full control via ReportConfig
cfg = ReportConfig(outlier_threshold=2.5, lite_mode=True, n_clusters=10)
pheno.generate_report("report.html", config=cfg)
# Overrides still work: config=cfg, title="My Report"
```

The `output_path` and `title` are parameters to `generate_report`. All other options below are `ReportConfig` fields and can be passed as keyword arguments or set on a `ReportConfig` instance.

---

#### `generate_report`

Generates a standalone, interactive HTML document containing data summaries, plots, distance calculations, and embedded images (Base64). Embeds Plotly JS so the report can be shared and viewed offline. To compute distances, `reference_filters` must be passed.

**Returns:** *str*, path to the HTML output file.

**Report parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_path` | `"pheno_report.html"` | Path to save the `.html` file |
| `title` | `"PhenoMe Analysis Report"` | Main report title |

**Report content (include_* flags)**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_plots` | `True` | Interactive PCA, t-SNE, UMAP plots |
| `include_distance_analysis` | `True` | Distances to reference group (needs `reference_filters`) |
| `include_outliers` | `True` | Outliers via [Z-score](pipeline.md#detect_outliers) |
| `include_correlations` | `True` | Properties correlated with embeddings |
| `include_property_stats` | `True` | Aggregated properties tables |
| `include_clustering` | `True` | [Clusters](pipeline.md#compute_clustering) and prototypes |
| `include_image_gallery` | `True` | Base64 thumbnails gallery |
| `include_interpretability` | `True` | Multivariate explanation section |

**Analysis options**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reference_filters` | `None` | Group dict for distance baselines (e.g. `{'condition': 'Control'}`) |
| `color_by` | `None` | Default grouping for charts |
| `overview_metadata_keys` | `None` | Metadata keys in overview |
| `metadata_keys` | `None` | Metadata keys in detail tables |
| `top_k_features` | `15` | Top properties for correlation charts |
| `interpretability_model_type` | `"lasso"` | Model for multivariate explanation (`"lasso"` or `"random_forest"`) |
| `outlier_threshold` | `3.0` | Z-score threshold for outliers |
| `outlier_group_by` | `None` | Group key for outlier detection |
| `n_clusters` | `5` | Cluster count |
| `clustering_method` | `"kmeans"` | Clustering algorithm (`"kmeans"`, `"dbscan"`, `"gmm"`) |
| `n_cluster_prototypes` | `3` | Prototype images per cluster |
| `clustering_reduce_dim` | `100` | Max PCA dimension before clustering; `None` = no reduction |
| `clustering_reduce_method` | `"pca"` | DR method before clustering: `"pca"`, `"tsne"`, or `"umap"` |

**Gallery options**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_gallery_images` | `5` | Random sample images per category |
| `image_size_px` | `200` | Thumbnail size in pixels |
| `property_group_by` | `None` | Grouping for property tables |
| `property_table_max_rows` | `50` | Max rows in property tables |

**Other**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `description` | `None` | Custom text for report |
| `filters` | `None` | Metadata filters to subset data |
| `exclude` | `None` | Metadata exclusions. Applied after filters. |
| `lite_mode` | `False` | Optimize for large datasets (sampling, WebGL) |
