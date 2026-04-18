---
title: "Function Location Guide"
---

Quick lookup of where each function and class lives, and links to extended documentation.

---

## Audience

|       | Meaning                                                                                |
| ----- | -------------------------------------------------------------------------------------- |
| ✅ **Public**   | Part of the main workflow. Use via `PhenoMe` or top-level imports.         |
| 🔧 **Advanced** | For power users extending the pipeline (custom models, metadata, properties, plugins). |
| 🔒 **Internal** | Used by the pipeline internally. Rarely needed directly.                               |

---

## Index

| Section | Module |
| ------- | ------ |
| [Top-level](#top-level-phenome) | `phenome` |
| [Pipeline methods](#pipeline-methods-on-phenome) | `PhenoMe` |
| [Metadata](#metadata-phenomemetadata) | `phenome.metadata` |
| [Utilities](#utilities-phenomeutils) | `phenome.utils` |
| [I/O](#io-phenomeio) | `phenome.io` |
| [Plugins](#plugins-phenomeplugins) | `phenome.plugins` |
| [Core](#core-phenomecore) | `phenome.core` |
| [Mixins](#mixins-phenomemixins) | `phenome.mixins` |

### Module one-liners

| Module | One-line role |
|--------|----------------|
| `phenome` | Stable imports: `PhenoMe`, model loader, metadata helpers, property factories. |
| `phenome.pipeline` | `PhenoMe` orchestrator wiring mixins into one user class. |
| `phenome.metadata` | OOP metadata extractors (`MetadataBase`, path / DataFrame sources). |
| `phenome.utils` | Model wrappers, transforms, device helpers, metadata + property utilities. |
| `phenome.io` | `read_image`, `FileDiscovery`, `CheckpointManager`, array layout helpers. |
| `phenome.plugins` | Registries for properties, metadata extractors, report sections. |
| `phenome.core` | Results container, DR, correlation, interpretability, export, validation. |
| `phenome.mixins` | Implementation pieces behind `PhenoMe` (dataset, viz, distances, …). |

### Typical import blocks

```python

from phenome import PhenoMe, load_dinov2_model

from phenome import get_metadata_from_path, make_dataframe_metadata_fn

from phenome.metadata import MetadataBase, PathTemplateMetadata, DataFrameMetadata

from phenome.io import read_image, FileDiscovery, CheckpointManager
```

---

## Top-level (`phenome`)

| Import                                                                                                                 | Description                                                        | Tier | Extended docs                                             |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---- | --------------------------------------------------------- |
| `PhenoMe`                                                                                                  | Main orchestrator for the workflow                                 | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/)                     |
| `load_dinov2_model`                                                                                                    | Load DINOv2 wrapped for PhenoMe                                    | ✅ | [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/)          |
| `default_metadata_from_path`                                                                                           | Minimal metadata from path (file_path, filename)                   | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#default_metadata_from_path) · [Experiment Details](/PhenoMe/guides/experiment-details/) |
| `get_metadata_from_path`                                                                                               | Parse metadata from path template string                           | ✅ | [Experiment Details](/PhenoMe/guides/experiment-details/)              |
| `make_dataframe_metadata_fn`                                                                                           | Build metadata function from DataFrame lookup                      | ✅ | [Experiment Details](/PhenoMe/guides/experiment-details/)              |
| `MetadataBase`, `DefaultMetadata`, `PathTemplateMetadata`, `DataFrameMetadata`                                         | OOP metadata extractors (pass to `find_files`)                     | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/)             |
| `create_regionprops_function`, `create_intensity_function`, `create_masked_intensity_function`                         | Property factories for masks/images (pass to `compute_properties`) | ✅ | [Custom Properties](/PhenoMe/guides/custom-properties/) · [Property Reference](/PhenoMe/guides/property-reference/) |
| `create_concentric_ring_function`, `create_texture_function`, `create_blur_effect_function`, `create_entropy_function` | Additional property factories                                      | ✅ | [Custom Properties](/PhenoMe/guides/custom-properties/)          |
| `get_preset_property_functions`                                                                                        | Get built-in property presets                                      | ✅ | [Property Reference](/PhenoMe/guides/property-reference/)        |
| `PhenoMeResults`                                                                                                      | Container returned by `load_results`                               | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#class-phenomeresults) |

---

## Pipeline methods (on `PhenoMe`)

*All public — this is the main API.*

Common methods on `PhenoMe` instances:

| Method                               | Description                          | Tier | Extended docs                                                       |
| ------------------------------------ | ------------------------------------ | ---- | ------------------------------------------------------------------- |
| `find_files`                         | Discover images and extract metadata | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#find_files)                    |
| `process_images`                     | Extract embeddings from images       | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#process_images)                |
| `compute_properties`                 | Compute image/mask features          | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#compute_properties)        |
| `compute_reference_distances`        | Distances to reference group         | ✅ | [Distances](/PhenoMe/advanced/api/distances/#compute_reference_distances) |
| `plot_pca`, `plot_tsne`, `plot_umap` | Dimensionality reduction plots       | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/)                     |
| `plot_distance_distribution`         | Distance distribution plots          | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/)                     |
| `generate_report`                    | Generate HTML report                 | ✅ | [Report](/PhenoMe/advanced/api/report/)                                   |
| `save_results`, `load_results`       | Save/load HDF5 results               | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/)                               |
| `create_interactive_explorer`        | Interactive Jupyter dashboard        | ✅ | [Interactive Explorer](/PhenoMe/advanced/api/interactive/)                |

---

## Metadata (`phenome.metadata`)

| Import                 | Description                                  | Tier | Extended docs                                 |
| ---------------------- | -------------------------------------------- | ---- | --------------------------------------------- |
| `MetadataBase`         | Abstract base for metadata extractors        | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/) |
| `DefaultMetadata`      | Minimal extractor (file_path, filename, id)  | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/) |
| `PathTemplateMetadata` | Parse from path template with capture groups | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/) |
| `DataFrameMetadata`    | Look up metadata from a DataFrame            | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/) |

---

## Utilities (`phenome.utils`)

| Import                                                                               | Description                              | Tier | Extended docs                                             |
| ------------------------------------------------------------------------------------ | ---------------------------------------- | ---- | --------------------------------------------------------- |
| `load_dinov2_model`                                                                  | Load DINOv2 wrapped for PhenoMe          | ✅ | [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/)          |
| `default_metadata_from_path`, `get_metadata_from_path`, `make_dataframe_metadata_fn` | Metadata helpers (same as top-level)     | ✅ | [Experiment Details](/PhenoMe/guides/experiment-details/) · [Metadata](/PhenoMe/advanced/api/metadata/) |
| `create_regionprops_function`, `create_intensity_function`, etc.                     | Property factories (same as top-level)   | ✅ | [Custom Properties](/PhenoMe/guides/custom-properties/)          |
| `ModelWrapper`, `DinoV2ModelWrapper`                                                 | For custom vision models                 | 🔧 | [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/)          |
| `get_default_device`                                                                 | Return global default device             | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#get_default_device) |
| `set_default_device`                                                                 | Set global default device                | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#set_default_device) |
| `set_determinism`                                                                    | Set random seeds for reproducibility     | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#set_determinism)   |
| `TransformBuilder`, `PadToSize`, `TypeMaxNorm`                                       | For custom model preprocessing           | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#transformers)      |
| `normalize_by_dtype_max`, `scale_minmax`                                             | Normalization helpers used in transforms | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/#transformers)      |

---

## I/O (`phenome.io`)

| Import              | Description                                                                     | Tier | Extended docs                                             |
| ------------------- | ------------------------------------------------------------------------------- | ---- | --------------------------------------------------------- |
| `read_image`        | Load image(s) from path(s) as numpy array                                       | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#read_image)        |
| `FileDiscovery`     | Find image files and extract metadata (pipeline uses internally)                | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#filediscovery)     |
| `CheckpointManager` | Low-level HDF5 checkpoint; most users use `checkpoint_path` on `process_images` | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#checkpointmanager) · [HDF5 Protocol](/PhenoMe/advanced/database_protocol/) |
| `ensure_hwc`        | Ensure image array is (H, W, C) format                                          | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/#ensure_hwc)        |

---

## Plugins (`phenome.plugins`)

*All advanced — for extending the pipeline.*

| Import                                                  | Description                           | Tier | Extended docs                      |
| ------------------------------------------------------- | ------------------------------------- | ---- | ---------------------------------- |
| `register_property`, `get_property`, `list_properties`  | Property registry                     | 🔧 | [Extending](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) |
| `register_metadata_extractor`, `get_metadata_extractor` | Metadata extractor registry           | 🔧 | [Extending](/PhenoMe/guides/extending/)   |
| `register_report_section`, `get_report_sections`        | Report section registry               | 🔧 | [Extending](/PhenoMe/guides/extending/)   |
| `get_blob_properties`                                   | Built-in blob-based property function | 🔧 | [Custom Properties](/PhenoMe/guides/custom-properties/) |

---

## Core (`phenome.core`)

*All advanced — for custom scripts extending or bypassing the pipeline.*

| Import                                                                                        | Description                            | Tier | Extended docs                                                |
| --------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ------------------------------------------------------------ |
| `filter_indices`                                                                              | Filter image indices by metadata       | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#core-module-advanced) |
| `build_metadata_columns`                                                                      | Build metadata columns for a DataFrame | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#core-module-advanced) |
| `build_export_dataframe`, `prepare_embedding_dataframe`                                       | Build export DataFrame from results    | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#core-module-advanced) |
| `run_dimensionality_reduction`, `run_dimensionality_reduction_matrix`                         | PCA, t-SNE, UMAP                       | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#core-module-advanced) |
| `compute_pearson_correlation`, `compute_spearman_correlation`, `compute_distance_correlation` | Correlation metrics                    | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#core-module-advanced) |
| `compute_entropy`, `compute_mutual_info`                                                      | Information-theoretic metrics          | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#core-module-advanced) |
| `build_combined_features`                                                                     | Concatenate embeddings + properties    | 🔒 | —                                                            |
| `validate_results`                                                                            | Validate pipeline results              | 🔒 | —                                                            |
| `metadata_to_stable_key`, `optimize_property_types`                                           | Property utilities                     | 🔒 | —                                                            |

---

## Mixins (`phenome.mixins`)

*Prefer using `PhenoMe`; these classes implement its methods.*

| Import                                                                                 | Description                                                                 | Tier | Extended docs                                  |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---- | ---------------------------------------------- |
| `PhenoMeInteractive`, `create_interactive_explorer`                                          | Interactive Jupyter explorer (also via `pheno.create_interactive_explorer`) | ✅ | [Interactive Explorer](/PhenoMe/advanced/api/interactive/) |
| `PhenoMeDataset`, `collate_fn`                                                     | Dataset for embedding extraction (used by `process_images`)                 | 🔒 | [Pipeline](/PhenoMe/advanced/api/pipeline/)          |
| `EmbeddingExtractor`                                                                   | Extracts embeddings from model                                              | 🔒 | —                                            |
| `PhenoMeProperties`, `PhenoMeDistances`, `PhenoMeAnalysis`, `PhenoMeVisualization` | Implement pipeline methods; use via `pheno.`*                               | 🔒 | [Properties](/PhenoMe/advanced/api/properties/) · [Distances](/PhenoMe/advanced/api/distances/) · [Visualization](/PhenoMe/advanced/api/visualization/) |
