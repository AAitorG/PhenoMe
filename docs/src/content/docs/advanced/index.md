---
title: "Reference"
---

Full API details and technical documentation for advanced use. For a brief overview, see the [Quick reference](/PhenoMe/quick-reference/). For a quick lookup of where each function lives, see the [Function location guide](/PhenoMe/function-location-guide/).

**API reference (auto-generated):** each docs build runs [`docs/scripts/generate_api_docs.py`](https://github.com/AAitorG/PhenoMe/blob/main/docs/scripts/generate_api_docs.py) and writes narrative Markdown under **`advanced/api/`** from live Python docstrings (optional leading `@section` / `@order` lines in docstrings override grouping). Improve **docstrings** in `phenome/` and rebuild to update the site—there is no separate hand-maintained API copy.

---

## API reference

| Document | Description |
|----------|-------------|
| [PhenoMe](/PhenoMe/advanced/api/pipeline/) | Main orchestrator class for the complete workflow |
| [Vision model wrappers](/PhenoMe/advanced/api/model-wrapper/) | ModelWrapper (base), DinoV2ModelWrapper |
| [Visualization](/PhenoMe/advanced/api/visualization/) | PCA, t-SNE, UMAP, centroid plots, distance distributions, correlations |
| [Distance computation](/PhenoMe/advanced/api/distances/) | Computing distances to reference groups |
| [Property computation](/PhenoMe/advanced/api/properties/) | Custom property extraction from images and masks |
| [Report generation](/PhenoMe/advanced/api/report/) | `ReportConfig` and standalone HTML report generation |
| [Interactive explorer](/PhenoMe/advanced/api/interactive/) | Widget-based dashboard for exploring embeddings in Jupyter |
| [Utilities](/PhenoMe/advanced/api/utilities/) | Device helpers, transforms, image I/O, checkpoints, file discovery |
| [Metadata classes](/PhenoMe/advanced/api/metadata/) | MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata |
| [Plugins](/PhenoMe/advanced/api/plugins/) | Property and metadata registries, report sections |

---

## Extending the pipeline

- [Extending the pipeline](/PhenoMe/guides/extending/): Overview of extension points (metadata, properties, plugins)
- [Experiment details](/PhenoMe/guides/experiment-details/): Path templates, CSV lookup, OOP MetadataBase
- [Custom properties](/PhenoMe/guides/custom-properties/): Property presets, requirement types, property factories
- [Plugins and extensibility](/PhenoMe/guides/plugins/): Plugin registry, custom presets

---

## Technical reference

| Document | Description |
|----------|-------------|
| [HDF5 database protocol](/PhenoMe/advanced/database_protocol/) | Internal checkpoint and result file structure |
