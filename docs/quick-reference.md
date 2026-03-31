# Quick Reference

Brief workflow overview and common options. For full signatures, see [API Reference](reference/index.md). For function lookup, see [Function Location Guide](function-location-guide.md).

[← Back to docs](index.md)

---

## 5-Step Workflow

1. **Setup**: Load model and create pipeline
2. **Find files**: Discover images and metadata
3. **Process**: Extract embeddings from images
4. **Properties** (optional): Compute image/mask features
5. **Analyze & visualize**: Distances, plots, reports

---

## Quick Example

```python
from phenome import PhenoMe, load_dinov2_model

# 1. Setup
model, wrapper = load_dinov2_model()
pheno = PhenoMe()  # pass seed=42 for reproducibility

# 2. Find files

file_df = pheno.find_files("path/to/images")


# 3. Process
pheno.process_images(wrapper)

# 4. Properties (optional)
pheno.compute_properties(property_preset='basic')

# 5. Analyze and visualize
distance_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    source='embeddings'
)
pheno.plot_pca(color_by='condition')
pheno.plot_distance_distribution(distance_results, group_by='condition')

# Generate report
pheno.generate_report(
    output_path="report.html",
    reference_filters={'condition': 'Control'},
    include_clustering=True,
    include_image_gallery=True
)
```

---

## Common Options

| Option | Where | What it does |
|--------|-------|--------------|
| `property_preset` | `compute_properties()` | Choose built-in feature set: `'none'`, `'basic'`, `'regionprops'`, `'intensity'`, `'full'`, `'full_extended'`. See [Property Reference](guides/property-reference.md). |
| `reference_filters` | `compute_reference_distances()`, `generate_report()` | Dict defining your control/reference group, e.g. `{'condition': 'Control'}`. |
| `filters` | Many methods | Dict to include only matching rows. Single value or list, e.g. `{'condition': ['Control', 'Treatment']}`. |
| `exclude` | Same as `filters` | Same format as `filters`; excludes matching rows. Applied after filters. |
| `source` | `compute_reference_distances()` | `'embeddings'`, `'properties'`, or `'combined'` (what to compute distances on). |
| `channel_mode` | `process_images()` | `'split'` (each channel separately) or `'combined'` (channels as RGB). |
| `metadata_fn` | `find_files()` | Custom function to extract metadata from paths. See [Experiment Details](guides/experiment-details.md). |
| `seed` | `PhenoMe()` | Set for reproducible results (t-SNE, clustering, sampling). |

For full parameter lists, see the [API Reference](reference/index.md).

---

## Quick Decisions

### Channel mode (split vs combined)

| Your data                          | Use       |
|------------------------------------|-----------|
| Fluorescence, distinct channels    | `channel_mode='split'` |
| Brightfield, phase contrast, RGB   | `channel_mode='combined'` |

### Property preset (when using masks)

| You have masks | Want shape + intensity | Use preset     |
|----------------|------------------------|----------------|
| No             | —                      | `intensity`    |
| Yes            | Basic                  | `basic`        |
| Yes            | Shape only             | `regionprops`  |
| Yes            | Full analysis          | `full` or `full_extended` |

See [Property Reference](guides/property-reference.md#choosing-properties) for details.

---

## Find by task

| I want to... | Go to |
|--------------|-------|
| Install and run my first analysis | [Getting Started](getting-started.md) |
| Extract metadata from paths or CSV | [Experiment Details](guides/experiment-details.md) |
| Add custom image properties | [Custom Properties](guides/custom-properties.md) |
| Build PCA, t-SNE, or distance plots | [Visualization](reference/api/visualization.md), [Distances](reference/api/distances.md) |
| Generate an HTML report | [Report](reference/api/report.md) · [Report Guide](guides/report-guide.md) |
| Explore interactively in Jupyter | [Interactive Explorer](reference/api/interactive.md) |
| Explore new images without re-running full processing | [Temporal Images](workflows.md#temporal-images-explore-new-data-in-memory) |
| See end-to-end workflows | [Common Workflows](workflows.md) |
