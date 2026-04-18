---
title: "Reference"
---

Full API details and technical documentation for advanced use. For a brief overview, see the [Quick reference](../quick-reference.md). For a quick lookup of where each function lives, see the [Function location guide](../function-location-guide.md).

**API reference (auto-generated):** each docs build runs [`docs/scripts/generate_api_docs.py`](https://github.com/AAitorG/PhenoMe/blob/main/docs/scripts/generate_api_docs.py) and writes narrative Markdown under **`reference/api/`** from live Python docstrings (optional leading `@section` / `@order` lines in docstrings override grouping). Improve **docstrings** in `phenome/` and rebuild to update the site—there is no separate hand-maintained API copy.

---

## API reference

| Document | Description |
|----------|-------------|
| [PhenoMe](api/pipeline.md) | Main orchestrator class for the complete workflow |
| [Vision model wrappers](api/model-wrapper.md) | ModelWrapper (base), DinoV2ModelWrapper |
| [Visualization](api/visualization.md) | PCA, t-SNE, UMAP, centroid plots, distance distributions, correlations |
| [Distance computation](api/distances.md) | Computing distances to reference groups |
| [Property computation](api/properties.md) | Custom property extraction from images and masks |
| [Report generation](api/report.md) | `ReportConfig` and standalone HTML report generation |
| [Interactive explorer](api/interactive.md) | Widget-based dashboard for exploring embeddings in Jupyter |
| [Utilities](api/utilities.md) | Device helpers, transforms, image I/O, checkpoints, file discovery |
| [Metadata classes](api/metadata.md) | MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata |
| [Plugins](api/plugins.md) | Property and metadata registries, report sections |

---

## Extending the pipeline

- [Extending the pipeline](../guides/extending.md): Overview of extension points (metadata, properties, plugins)
- [Experiment details](../guides/experiment-details.md): Path templates, CSV lookup, OOP MetadataBase
- [Custom properties](../guides/custom-properties.md): Property presets, requirement types, property factories
- [Plugins and extensibility](../guides/plugins.md): Plugin registry, custom presets

---

## Technical reference

| Document | Description |
|----------|-------------|
| [HDF5 database protocol](DATABASE_PROTOCOL.md) | Internal checkpoint and result file structure |
