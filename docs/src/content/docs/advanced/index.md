---
title: "API & data formats"
description: Hub for generated module docs, quick lookups, and the HDF5 checkpoint layout.
sidebar:
  order: 0
---

- For a cheat-sheet of available methods, see the
  [Quick reference](/PhenoMe/quick-reference/).
- To find out where a specific symbol lives, use the
  [Function location guide](/PhenoMe/function-location-guide/).

## API reference

The API pages under [`advanced/api/`](/PhenoMe/advanced/api/pipeline/) are
**auto-generated** from Python docstrings on every docs build
(`docs/scripts/generate_api_docs.py`). **To change them, edit the
docstrings in `phenome/` and rebuild.** There is no separate
hand-maintained copy.

| Document | Description |
|----------|-------------|
| [PhenoMe pipeline](/PhenoMe/advanced/api/pipeline/) | Main orchestrator class for the complete workflow. |
| [Vision model wrappers](/PhenoMe/advanced/api/model-wrapper/) | `ModelWrapper` (base), `DinoV2ModelWrapper`. |
| [Visualization](/PhenoMe/advanced/api/visualization/) | PCA, t-SNE, UMAP, centroid plots, distance distributions, correlations. |
| [Distance computation](/PhenoMe/advanced/api/distances/) | Computing distances to reference groups. |
| [Property computation](/PhenoMe/advanced/api/properties/) | Custom property extraction from images and masks. |
| [Report generation](/PhenoMe/advanced/api/report/) | `ReportConfig` and standalone HTML report generation. |
| [Interactive explorer](/PhenoMe/advanced/api/interactive/) | Widget-based dashboard for exploring embeddings in Jupyter. |
| [Utilities](/PhenoMe/advanced/api/utilities/) | Device helpers, transforms, image I/O, checkpoints, file discovery. |
| [Metadata classes](/PhenoMe/advanced/api/metadata/) | `MetadataBase`, `DefaultMetadata`, `PathTemplateMetadata`, `DataFrameMetadata`. |
| [Plugins](/PhenoMe/advanced/api/plugins/) | Property and metadata registries, report sections. |

## Extending the pipeline

- [Extending PhenoMe](/PhenoMe/guides/extending/) - overview of extension points.
- [Experiment details](/PhenoMe/guides/experiment-details/) - metadata
  extraction.
- [Custom properties](/PhenoMe/guides/custom-properties/) - property presets,
  requirement types, factories.
- [Plugins](/PhenoMe/guides/plugins/) - plugin registry, custom presets.

## Technical reference

| Document | Description |
|----------|-------------|
| [HDF5 database protocol](/PhenoMe/advanced/database_protocol/) | Internal checkpoint and result file structure. |
