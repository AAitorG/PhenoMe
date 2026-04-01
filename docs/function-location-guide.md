# Function Location Guide

Quick lookup of where each function and class lives, and links to extended documentation.

[← Documentation index](index.md)

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

---

## Top-level (`phenome`)

| Import                                                                                                                 | Description                                                        | Tier | Extended docs                                             |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---- | --------------------------------------------------------- |
| `PhenoMe`                                                                                                  | Main orchestrator for the workflow                                 | ✅ | [Pipeline](reference/api/pipeline.md)                     |
| `load_dinov2_model`                                                                                                    | Load DINOv2 wrapped for PhenoMe                                    | ✅ | [Model Wrappers](reference/api/model-wrapper.md)          |
| `default_metadata_from_path`                                                                                           | Minimal metadata from path (file_path, filename)                   | ✅ | [Utilities](reference/api/utilities.md#default_metadata_from_path) |
| `default_metadata_from_path`                                                                                           | Minimal metadata from path (file_path, filename)                   | ✅ | [Experiment Details](guides/experiment-details.md)              |
| `get_metadata_from_path`                                                                                               | Parse metadata from path template string                           | ✅ | [Experiment Details](guides/experiment-details.md)              |
| `make_dataframe_metadata_fn`                                                                                           | Build metadata function from DataFrame lookup                      | ✅ | [Experiment Details](guides/experiment-details.md)              |
| `MetadataBase`, `DefaultMetadata`, `PathTemplateMetadata`, `DataFrameMetadata`                                         | OOP metadata extractors (pass to `find_files`)                     | ✅ | [Metadata Classes](reference/api/metadata.md)             |
| `create_regionprops_function`, `create_intensity_function`, `create_masked_intensity_function`                         | Property factories for masks/images (pass to `compute_properties`) | ✅ | [Custom Properties](guides/custom-properties.md)          |
| `create_regionprops_function`, `create_intensity_function`, `create_masked_intensity_function`                         | Property factories for masks/images (pass to `compute_properties`) | ✅ | [Property Reference](guides/property-reference.md)        |
| `create_concentric_ring_function`, `create_texture_function`, `create_blur_effect_function`, `create_entropy_function` | Additional property factories                                      | ✅ | [Custom Properties](guides/custom-properties.md)          |
| `get_preset_property_functions`                                                                                        | Get built-in property presets                                      | ✅ | [Property Reference](guides/property-reference.md)        |
| `PhenoMeResults`                                                                                                      | Container returned by `load_results`                               | ✅ | [Utilities](reference/api/utilities.md#core-module-advanced) |

---

## Pipeline methods (on `PhenoMe`)

*All public — this is the main API.*

Common methods on `PhenoMe` instances:

| Method                               | Description                          | Tier | Extended docs                                                       |
| ------------------------------------ | ------------------------------------ | ---- | ------------------------------------------------------------------- |
| `find_files`                         | Discover images and extract metadata | ✅ | [Pipeline](reference/api/pipeline.md#find_files)                    |
| `process_images`                     | Extract embeddings from images       | ✅ | [Pipeline](reference/api/pipeline.md#process_images)                |
| `compute_properties`                 | Compute image/mask features          | ✅ | [Properties](reference/api/properties.md#compute_properties)        |
| `compute_reference_distances`        | Distances to reference group         | ✅ | [Distances](reference/api/distances.md#compute_reference_distances) |
| `plot_pca`, `plot_tsne`, `plot_umap` | Dimensionality reduction plots       | ✅ | [Visualization](reference/api/visualization.md)                     |
| `plot_distance_distribution`         | Distance distribution plots          | ✅ | [Visualization](reference/api/visualization.md)                     |
| `generate_report`                    | Generate HTML report                 | ✅ | [Report](reference/api/report.md)                                   |
| `save_results`, `load_results`       | Save/load HDF5 results               | ✅ | [Pipeline](reference/api/pipeline.md)                               |
| `create_interactive_explorer`        | Interactive Jupyter dashboard        | ✅ | [Interactive Explorer](reference/api/interactive.md)                |

---

## Metadata (`phenome.metadata`)

| Import                 | Description                                  | Tier | Extended docs                                 |
| ---------------------- | -------------------------------------------- | ---- | --------------------------------------------- |
| `MetadataBase`         | Abstract base for metadata extractors        | ✅ | [Metadata Classes](reference/api/metadata.md) |
| `DefaultMetadata`      | Minimal extractor (file_path, filename, id)  | ✅ | [Metadata Classes](reference/api/metadata.md) |
| `PathTemplateMetadata` | Parse from path template with capture groups | ✅ | [Metadata Classes](reference/api/metadata.md) |
| `DataFrameMetadata`    | Look up metadata from a DataFrame            | ✅ | [Metadata Classes](reference/api/metadata.md) |

---

## Utilities (`phenome.utils`)

| Import                                                                               | Description                              | Tier | Extended docs                                             |
| ------------------------------------------------------------------------------------ | ---------------------------------------- | ---- | --------------------------------------------------------- |
| `load_dinov2_model`                                                                  | Load DINOv2 wrapped for PhenoMe          | ✅ | [Model Wrappers](reference/api/model-wrapper.md)          |
| `default_metadata_from_path`, `get_metadata_from_path`, `make_dataframe_metadata_fn` | Metadata helpers (same as top-level)     | ✅ | [Experiment Details](guides/experiment-details.md)              |
| `default_metadata_from_path`, `get_metadata_from_path`, `make_dataframe_metadata_fn` | Metadata helpers (same as top-level)     | ✅ | [Metadata](reference/api/metadata.md)                     |
| `create_regionprops_function`, `create_intensity_function`, etc.                     | Property factories (same as top-level)   | ✅ | [Custom Properties](guides/custom-properties.md)          |
| `ModelWrapper`, `DinoV2ModelWrapper`                                                 | For custom vision models                 | 🔧 | [Model Wrappers](reference/api/model-wrapper.md)          |
| `get_default_device`                                                                 | Return global default device             | 🔧 | [Utilities](reference/api/utilities.md#get_default_device) |
| `set_default_device`                                                                 | Set global default device                | 🔧 | [Utilities](reference/api/utilities.md#set_default_device) |
| `set_determinism`                                                                    | Set random seeds for reproducibility     | 🔧 | [Utilities](reference/api/utilities.md#set_determinism)   |
| `TransformBuilder`, `PadToSize`, `TypeMaxNorm`                                       | For custom model preprocessing           | 🔧 | [Utilities](reference/api/utilities.md#transformers)      |
| `normalize_by_dtype_max`, `scale_minmax`                                             | Normalization helpers used in transforms | 🔒 | [Utilities](reference/api/utilities.md#transformers)      |

---

## I/O (`phenome.io`)

| Import              | Description                                                                     | Tier | Extended docs                                             |
| ------------------- | ------------------------------------------------------------------------------- | ---- | --------------------------------------------------------- |
| `read_image`        | Load image(s) from path(s) as numpy array                                       | 🔧 | [Utilities](reference/api/utilities.md#read_image)        |
| `FileDiscovery`     | Find image files and extract metadata (pipeline uses internally)                | 🔧 | [Utilities](reference/api/utilities.md#filediscovery)     |
| `CheckpointManager` | Low-level HDF5 checkpoint; most users use `checkpoint_path` on `process_images` | 🔧 | [Utilities](reference/api/utilities.md#checkpointmanager) |
| `CheckpointManager` | Low-level HDF5 checkpoint; most users use `checkpoint_path` on `process_images` | 🔧 | [HDF5 Protocol](reference/DATABASE_PROTOCOL.md)           |
| `ensure_hwc`        | Ensure image array is (H, W, C) format                                          | 🔒 | [Utilities](reference/api/utilities.md#ensure_hwc)        |

---

## Plugins (`phenome.plugins`)

*All advanced — for extending the pipeline.*

| Import                                                  | Description                           | Tier | Extended docs                      |
| ------------------------------------------------------- | ------------------------------------- | ---- | ---------------------------------- |
| `register_property`, `get_property`, `list_properties`  | Property registry                     | 🔧 | [Extending](guides/extending.md)   |
| `register_property`, `get_property`, `list_properties`  | Property registry                     | 🔧 | [Plugins](guides/plugins.md)       |
| `register_metadata_extractor`, `get_metadata_extractor` | Metadata extractor registry           | 🔧 | [Extending](guides/extending.md)   |
| `register_report_section`, `get_report_sections`        | Report section registry               | 🔧 | [Extending](guides/extending.md)   |
| `get_blob_properties`                                   | Built-in blob-based property function | 🔧 | [Custom Properties](guides/custom-properties.md) |

---

## Core (`phenome.core`)

*All advanced — for custom scripts extending or bypassing the pipeline.*

| Import                                                                                        | Description                            | Tier | Extended docs                                                |
| --------------------------------------------------------------------------------------------- | -------------------------------------- | ---- | ------------------------------------------------------------ |
| `filter_indices`                                                                              | Filter image indices by metadata       | 🔧 | [Utilities](reference/api/utilities.md#core-module-advanced) |
| `build_metadata_columns`                                                                      | Build metadata columns for a DataFrame | 🔧 | [Utilities](reference/api/utilities.md#core-module-advanced) |
| `build_export_dataframe`, `prepare_embedding_dataframe`                                       | Build export DataFrame from results    | 🔧 | [Utilities](reference/api/utilities.md#core-module-advanced) |
| `run_dimensionality_reduction`, `run_dimensionality_reduction_matrix`                         | PCA, t-SNE, UMAP                       | 🔧 | [Utilities](reference/api/utilities.md#core-module-advanced) |
| `compute_pearson_correlation`, `compute_spearman_correlation`, `compute_distance_correlation` | Correlation metrics                    | 🔧 | [Utilities](reference/api/utilities.md#core-module-advanced) |
| `compute_entropy`, `compute_mutual_info`                                                      | Information-theoretic metrics          | 🔧 | [Utilities](reference/api/utilities.md#core-module-advanced) |
| `build_combined_features`                                                                     | Concatenate embeddings + properties    | 🔒 | —                                                            |
| `validate_results`                                                                            | Validate pipeline results              | 🔒 | —                                                            |
| `metadata_to_stable_key`, `optimize_property_types`                                           | Property utilities                     | 🔒 | —                                                            |

---

## Mixins (`phenome.mixins`)

*Prefer using `PhenoMe`; these classes implement its methods.*

| Import                                                                                 | Description                                                                 | Tier | Extended docs                                  |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---- | ---------------------------------------------- |
| `PhenoMeInteractive`, `create_interactive_explorer`                                          | Interactive Jupyter explorer (also via `pheno.create_interactive_explorer`) | ✅ | [Interactive Explorer](reference/api/interactive.md) |
| `PhenoMeDataset`, `collate_fn`                                                     | Dataset for embedding extraction (used by `process_images`)                 | 🔒 | [Pipeline](reference/api/pipeline.md)          |
| `EmbeddingExtractor`                                                                   | Extracts embeddings from model                                              | 🔒 | —                                            |
| `PhenoMeProperties`, `PhenoMeDistances`, `PhenoMeAnalysis`, `PhenoMeVisualization` | Implement pipeline methods; use via `pheno.`*                               | 🔒 | [Properties](reference/api/properties.md)      |
| `PhenoMeProperties`, `PhenoMeDistances`, `PhenoMeAnalysis`, `PhenoMeVisualization` | Implement pipeline methods; use via `pheno.`*                               | 🔒 | [Distances](reference/api/distances.md)        |
| `PhenoMeProperties`, `PhenoMeDistances`, `PhenoMeAnalysis`, `PhenoMeVisualization` | Implement pipeline methods; use via `pheno.`*                               | 🔒 | [Visualization](reference/api/visualization.md) |
