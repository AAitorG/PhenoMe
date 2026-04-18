---
title: "Architecture"
---

High-level layout of the `phenome` package, data flow, and where to extend the pipeline.

---

## Package layout

| Area | Path | Role |
|------|------|------|
| Orchestrator | `phenome.pipeline` | `PhenoMe` class; user-facing workflow |
| Mixins | `phenome.mixins` | Behavior split into dataset, embeddings, properties, distances, analysis, visualization, interactive |
| Core algorithms | `phenome.core` | DR, correlation, interpretability, results container (`PhenoMeResults`), export, validation |
| I/O | `phenome.io` | Images, checkpoints, file discovery |
| Metadata | `phenome.metadata` | `MetadataBase` and concrete extractors |
| Utilities | `phenome.utils` | Model wrapper, transforms, metadata helpers, property factories |
| Plugins | `phenome.plugins` | Registries for properties, metadata extractors, report sections |
| Report | `phenome.report` | HTML report generation |

Imports for users are mostly from **`phenome`** (re-exports). See the [function location guide](/PhenoMe/function-location-guide/) for a full symbol index.

---

## Data flow (typical run)

**Linear flow:** `Image files` → `find_files` / `FileDiscovery` → `file_df` (+ metadata columns) → `process_images` → embeddings (HDF5 checkpoint) → `compute_properties` → dimensionality reduction / distances / correlations → plots, interactive explorer, HTML report.

1. **Discovery** — Paths + optional `metadata_fn` → tabular `file_df`.
2. **Embeddings** — Batched inference through a `ModelWrapper` → stored vectors (+ checkpointing).
3. **Properties** — Classical features from images/masks (presets or custom callables).
4. **Analysis / viz / report** — Use embeddings + metadata + properties together.

---

## Extension points

| Goal | Where to start |
|------|----------------|
| Custom metadata from paths or tables | [Experiment details](/PhenoMe/guides/experiment-details/), `MetadataBase`, `get_metadata_from_path`, `make_dataframe_metadata_fn` |
| Custom image features | [Custom properties](/PhenoMe/guides/custom-properties/), `phenome.plugins.register_property` |
| Custom vision backbone | [Model wrapper API](/PhenoMe/advanced/api/model-wrapper/), subclass `ModelWrapper` |
| Report sections | [Extending](/PhenoMe/guides/extending/), `register_report_section` |

**Rule of thumb:** prefer hooks (`metadata_fn`, property factories, plugins) before forking `PhenoMe` internals.

---

## Where should I add code?

| Adding… | Prefer |
|---------|--------|
| New user-facing workflow step | `phenome.pipeline` or relevant `phenome.mixins` |
| Reusable algorithm (DR, stats, export) | `phenome.core` |
| File formats, HDF5, discovery | `phenome.io` |
| User-registerable extension | `phenome.plugins` |

---

## Further reading

- [HDF5 protocol](/PhenoMe/advanced/database_protocol/)
- [Hand-written API index](/PhenoMe/advanced/)
