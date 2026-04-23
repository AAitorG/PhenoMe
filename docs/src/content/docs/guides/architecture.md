---
title: "Architecture"
description: High-level layout of the phenome package, data flow, and extension points.
sidebar:
  order: 0
---


## Package layout

| Area | Path | Role |
|------|------|------|
| Orchestrator | `phenome.pipeline` | `PhenoMe` class; user-facing workflow. |
| Mixins | `phenome.mixins` | Behaviour split into dataset, embeddings, properties, distances, analysis, visualization, interactive. |
| Core algorithms | `phenome.core` | DR, correlation, interpretability, results container (`PhenoMeResults`), export, validation. |
| I/O | `phenome.io` | Images, checkpoints, file discovery. |
| Metadata | `phenome.metadata` | `MetadataBase` and concrete extractors. |
| Utilities | `phenome.utils` | Model wrappers, transforms, metadata helpers, property factories. |
| Plugins | `phenome.plugins` | Registries for properties, metadata extractors, report sections. |
| Report | `phenome.report` | HTML report generation. |

Imports for users are mostly from **`phenome`** (re-exports). See the
[Function location guide](/PhenoMe/function-location-guide/) for a full symbol
index.

The `PhenoMe` orchestrator composes the mixins in `phenome.mixins` and
delegates to **`phenome.io`** (discovery, HDF5), **`phenome.metadata`**
(`MetadataBase` and helpers), **`phenome.core`** (DR, correlations,
interpretability, `PhenoMeResults`), **`phenome.utils`** (model wrappers,
property factories), **`phenome.plugins`** (user registries), and
**`phenome.report`** (HTML output).

## Data flow (typical run)

![Pipeline Overview](../../../assets/pipeline_overview.png)

The typical data flow maps directly to the pipeline overview components:

1.  **Discovery (Input)** — Gather raw image paths and experimental metadata.
2.  **Embeddings (A)** — Batch inference through a `ModelWrapper` to extract visual fingerprints.
3.  **Properties (B)** — Parallel computation of classical features from images/masks.
4.  **Analysis & Visualization (C)** — Combine embeddings and properties for quantification and plotting.

## Extension points

| Goal | Where to start |
|------|----------------|
| Custom metadata from paths or tables | [Experiment details](/PhenoMe/guides/experiment-details/), `MetadataBase`, `get_metadata_from_path`, `make_dataframe_metadata_fn`. |
| Custom image features | [Select properties](/PhenoMe/guides/select-properties/), `phenome.plugins.register_property`. |
| Custom vision backbone | [Model wrapper API](/PhenoMe/advanced/api/model-wrapper/); subclass `ModelWrapper`. |
| Report sections | [Extending PhenoMe](/PhenoMe/guides/extending/), `register_report_section`. |

**Rule of thumb:** prefer hooks (`metadata_fn`, property factories,
plugins) before forking `PhenoMe` internals.

## Where should I add code?

| Adding... | Prefer |
|-----------|--------|
| New user-facing workflow step | `phenome.pipeline` or relevant `phenome.mixins`. |
| Reusable algorithm (DR, stats, export) | `phenome.core`. |
| File formats, HDF5, discovery | `phenome.io`. |
| User-registerable extension | `phenome.plugins`. |

## Further reading

- [HDF5 protocol](/PhenoMe/advanced/database_protocol/).
- [API & data formats](/PhenoMe/advanced/).
- [Developer guide](/PhenoMe/guides/developer-guide/).
