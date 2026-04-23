---
title: "API & data formats"
description: Hub for auto-generated API docs, the function-location index, and the HDF5 checkpoint layout.
sidebar:
  order: 0
---

## API reference

The pages under [`advanced/api/`](/PhenoMe/advanced/api/pipeline/) are **auto-generated**
from Python docstrings on every docs build (`docs/scripts/generate_api_docs.py`).
**Edit docstrings in `phenome/` and rebuild** — there is no hand-maintained copy.

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

## Function location guide

The [Function location guide](/PhenoMe/function-location-guide/) is **auto-generated**
in the same build step from live `phenome` exports (`docs/scripts/api_docs/function_index.py`).
It lists top-level imports, `PhenoMe` methods, and submodule exports with tier badges (public /
advanced / internal) and deep links into the API pages above.

## Getting started

For a walkthrough of the complete pipeline, installation, and a first analysis, see the
[Getting started guide](/PhenoMe/getting-started/). For an overview of the core concepts,
see [Core concepts](/PhenoMe/concepts/).

## Extending the pipeline

- [Extending PhenoMe](/PhenoMe/guides/extending/) - overview of extension points.
- [Experiment details](/PhenoMe/guides/experiment-details/) - metadata
  extraction.
- [Select properties](/PhenoMe/guides/select-properties/) - property presets,
  requirement types, factories.
- [Plugins](/PhenoMe/guides/plugins/) - plugin registry, custom presets.

## Technical reference

| Document | Description |
|----------|-------------|
| [HDF5 database protocol](/PhenoMe/advanced/database_protocol/) | Internal checkpoint and result file structure. |
