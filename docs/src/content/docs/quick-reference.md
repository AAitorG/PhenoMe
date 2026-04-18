---
title: "Quick Reference"
---

Brief workflow overview and common options. For full signatures, see [API Reference](/PhenoMe/advanced/). For function lookup, see [Function Location Guide](/PhenoMe/function-location-guide/).

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

wrapper = load_dinov2_model()
pheno = PhenoMe()  # pass device=torch.device("cuda") and seed=42 as needed

file_df = pheno.find_files("path/to/images")

pheno.process_images(wrapper)

pheno.compute_properties(property_preset='basic')

distance_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    source='embeddings'
)
pheno.plot_pca(color_by='condition')
pheno.plot_distance_distribution(distance_results, group_by='condition')

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
| `property_preset` | `compute_properties()` | Choose built-in feature set: `'none'`, `'basic'`, `'regionprops'`, `'intensity'`, `'full'`, `'full_extended'`. See [Property Reference](/PhenoMe/guides/property-reference/). |
| `reference_filters` | `compute_reference_distances()`, `generate_report()` | Dict defining your control/reference group, e.g. `{'condition': 'Control'}`. |
| `filters` | Many methods | Dict to include only matching rows. Single value or list, e.g. `{'condition': ['Control', 'Treatment']}`. |
| `exclude` | Same as `filters` | Same format as `filters`; excludes matching rows. Applied after filters. |
| `source` | `compute_reference_distances()` | `'embeddings'`, `'properties'`, or `'combined'` (what to compute distances on). |
| `channel_mode` | `process_images()` | `'split'` (each channel separately) or `'combined'` (channels as RGB). |
| `metadata_fn` | `find_files()` | Custom function to extract metadata from paths. See [Experiment Details](/PhenoMe/guides/experiment-details/). |
| `seed` | `PhenoMe()` | Set for reproducible results (t-SNE, clustering, sampling). |
| `device` | `PhenoMe()` | GPU/CPU for embedding extraction and analysis. |

For full parameter lists, see the [API Reference](/PhenoMe/advanced/).

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

See [Property Reference](/PhenoMe/guides/property-reference/#choosing-properties) for details.

---

## Find by task

| I want to... | Go to |
|--------------|-------|
| Install and run my first analysis | [Getting Started](/PhenoMe/getting-started/) |
| Extract metadata from paths or CSV | [Experiment Details](/PhenoMe/guides/experiment-details/) |
| Add custom image properties | [Custom Properties](/PhenoMe/guides/custom-properties/) |
| Build PCA, t-SNE, or distance plots | [Visualization](/PhenoMe/advanced/api/visualization/), [Distances](/PhenoMe/advanced/api/distances/) |
| Explain embedding axes with properties | [Interpretability Guide](/PhenoMe/guides/interpretability/) |
| Generate an HTML report | [Report](/PhenoMe/advanced/api/report/) · [Report Guide](/PhenoMe/guides/report-guide/) |
| Explore interactively in Jupyter | [Interactive Explorer](/PhenoMe/advanced/api/interactive/) |
| Explore new images without re-running full processing | [Temporal Images](/PhenoMe/workflows/#temporal-images-explore-new-data-in-memory) |
| See end-to-end workflows | [Common Workflows](/PhenoMe/workflows/) |
