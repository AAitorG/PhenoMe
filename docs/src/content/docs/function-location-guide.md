---
title: "Function location guide"
description: Quick lookup of which module exports each PhenoMe function or class, grouped by tier (Public, Advanced, Internal). Auto-generated on each docs build.
sidebar:
  order: 1
---

:::note[Auto-generated]
This page is rebuilt from live ``phenome`` exports and API routing in ``docs/scripts/api_docs/function_index.py`` (invoked by ``docs/scripts/generate_api_docs.py``). **Do not edit by hand** — change code or docstrings and rebuild.
:::

## Audience

|  | Meaning |
| --- | --- |
| ✅ **Public** | Part of the main workflow. Use via `PhenoMe` or top-level imports. |
| 🔧 **Advanced** | For power users extending the pipeline (custom models, metadata, properties, plugins). |
| 🔒 **Internal** | Used by the pipeline internally. Rarely needed directly. |

---

## Index

| Section | Module |
| --- | --- |
| [Top-level](#top-level-phenome) | `phenome` |
| [Pipeline methods](#pipeline-methods-on-phenome) | `PhenoMe` |
| [Metadata](#metadata-phenomemetadata) | `phenome.metadata` |
| [Utilities](#utilities-phenomeutils) | `phenome.utils` |
| [I/O](#io-phenomeio) | `phenome.io` |
| [Plugins](#plugins-phenomeplugins) | `phenome.plugins` |
| [Core](#core-phenomecore) | `phenome.core` |
| [Report](#report-phenomereportgenerator) | `phenome.report.generator` |
| [Mixins](#mixins-phenomemixins) | `phenome.mixins` |

### Module one-liners

| Module | One-line role |
| --- | --- |
| `phenome` | Stable imports: `PhenoMe`, model loader, metadata helpers, property factories. |
| `phenome.pipeline` | `PhenoMe` orchestrator wiring mixins into one user class. |
| `phenome.metadata` | OOP metadata extractors (`MetadataBase`, path / DataFrame sources). |
| `phenome.utils` | Model wrappers, transforms, device helpers, metadata + property utilities. |
| `phenome.io` | `read_image`, `FileDiscovery`, `CheckpointManager`, array layout helpers. |
| `phenome.plugins` | Registries for properties, metadata extractors, report sections. |
| `phenome.core` | Results container, DR, correlation, interpretability, export, validation. |
| `phenome.report.generator` | `ReportConfig` and standalone HTML report generation. |
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

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `create_blur_effect_function` | Create property function that computes blur strength (Laplacian variance). | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_concentric_ring_function` | Create property function that computes stats per concentric ring (edge to core). | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_entropy_function` | Create property function that computes Shannon entropy of intensity distribution. | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_intensity_function` | Create intensity property function over entire image (no mask). | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_masked_intensity_function` | Create intensity property function over masked pixels only (mask > 0.5). | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_regionprops_function` | Create property function that extracts skimage regionprops from a mask. | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_texture_function` | Create property function that computes GLCM texture features within the mask. | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `DataFrameMetadata` | Metadata lookup from DataFrame. Supports single and multi-channel modes. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#dataframemetadata) |
| `default_metadata_from_path` | Simple, dataset-agnostic metadata extractor used by default. | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#default_metadata_from_path) · [Experiment Details](/PhenoMe/guides/experiment-details/) |
| `DefaultMetadata` | Minimal metadata extractor: file_path and filename from path. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#defaultmetadata) |
| `get_metadata_from_path` | Create metadata extractor function from a path template. | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#get_metadata_from_path) · [Experiment Details](/PhenoMe/guides/experiment-details/) |
| `get_preset_property_functions` | Return a preset dict of property functions for use with compute_properties. | ✅ | [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `load_dinov2_model` | Load a DINOv2 model and return a DinoV2ModelWrapper ready for processing. | ✅ | [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/#load_dinov2_model) |
| `make_dataframe_metadata_fn` | Build metadata function using a dataframe for metadata lookup. | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#make_dataframe_metadata_fn) · [Experiment Details](/PhenoMe/guides/experiment-details/) |
| `MetadataBase` | Base class for metadata extraction with configurable columns and ID handling. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#metadatabase) |
| `PathTemplateMetadata` | Metadata extractor from path template with capture groups. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#pathtemplatemetadata) |
| `PhenoMe` | Main class for phenotyping analysis using deep learning embeddings. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#class-phenome) |
| `PhenoMeResults` | Typed container for per-image phenotyping data. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#class-phenomeresults) |

---

## Pipeline methods (on `PhenoMe`)

*All public — this is the main API surface on `PhenoMe`.*

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `aggregate_embedding_property_correlations` | Aggregate pre-computed embedding-property correlations using the specified method. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-aggregate_embedding_property_correlations) |
| `analyze_group_enrichment` | Compute z-score enrichment of properties per group (e.g. clusters). | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-analyze_group_enrichment) |
| `checkpoint_context` | Context manager that loads a checkpoint and guarantees it is closed on exit. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-checkpoint_context) |
| `clear_temporal_data` | Remove all temporal images (metadata['source'] == 'NEW') from the session. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-clear_temporal_data) |
| `compute_clustering` | Perform clustering and store labels in ``metadata[i]['cluster']`` for each image. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-compute_clustering) |
| `compute_component_correlation` | Correlate dim-reduction components with phenotypic properties. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-compute_component_correlation) |
| `compute_embedding_property_correlations` | Compute correlations between embedding dimensions and phenotypic scalar properties. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-compute_embedding_property_correlations) |
| `compute_multivariate_interpretability` | Explain a DR component with LASSO or Random Forest; optional Plotly plot, text summary via logger, or `interpretability_fig` when `return_fig=True`. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-compute_multivariate_interpretability) |
| `compute_properties` | Compute per-image properties using presets and/or custom functions. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-compute_properties) |
| `compute_reference_distances` | Compute distances to a reference group; optional ``group_by`` for violin/summary. | ✅ | [Distances](/PhenoMe/advanced/api/distances/#api-phenomedistances-compute_reference_distances) |
| `create_interactive_explorer` | Launch the interactive explorer for this pipeline's results. | ✅ | [Interactive Explorer](/PhenoMe/advanced/api/interactive/#create_interactive_explorer) |
| `detect_outliers` | Detect outliers based on distance to centroid. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-detect_outliers) |
| `embedding_dim` | Return the embedding dimensionality, or 0 if unavailable. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-embedding_dim) |
| `export_dataset_table` | Export the dataset as a table (CSV, Parquet, or Excel). | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-export_dataset_table) |
| `export_experiment_config` | Export experiment configuration for reproducibility. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-export_experiment_config) |
| `filter_properties_by_group` | Group property DataFrame and compute per-group statistics. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-filter_properties_by_group) |
| `find_files` | Discover image and mask files from directories, cache internally, and return file_df. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-find_files) |
| `find_prototypes` | Find images closest to each group centroid. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-find_prototypes) |
| `generate_report` | Generate a comprehensive standalone HTML report from phenotyping results. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-generate_report) |
| `get_available_metadata_keys` | Return sorted metadata keys. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-get_available_metadata_keys) |
| `get_available_property_keys` | Return sorted list of property keys stored in results. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-get_available_property_keys) |
| `get_embeddings` | Return embeddings for the given row indices. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-get_embeddings) |
| `get_image_info` | Return metadata, properties, and optional distance for image idx. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-get_image_info) |
| `has_embeddings` | Return True if embedding data is available (lazy or eager). | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-has_embeddings) |
| `image_preview_png_bytes` | Rasterize the same view as ``plot_image_by_index`` to PNG bytes. | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_imagedisplaymixin-image_preview_png_bytes) |
| `inspect_data` | Inspect image and mask dimensions, shapes, and data ranges. Delegates to FileDiscovery. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-inspect_data) |
| `load_results` | Load and use an existing results/checkpoint file. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-load_results) |
| `plot_centroids` | Plot centroids of groups in reduced embedding space. | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_drplotsmixin-plot_centroids) |
| `plot_counts` | Plot count of images grouped by metadata using Plotly. | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-phenomevisualization-plot_counts) |
| `plot_image_by_index` | Plot a specific image by its index. | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_imagedisplaymixin-plot_image_by_index) |
| `plot_pca` | Plot PCA of embeddings, properties, or combined features (Plotly, WebGL by default). | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_drplotsmixin-plot_pca) |
| `plot_property_correlations` | Plot the top correlated properties as a horizontal bar chart. | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_distanceplotsmixin-plot_property_correlations) |
| `plot_tsne` | Plot t-SNE of embeddings, properties, or combined features (Plotly, WebGL by default). | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_drplotsmixin-plot_tsne) |
| `plot_umap` | Plot UMAP of embeddings, properties, or combined features (Plotly, WebGL by default). | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_drplotsmixin-plot_umap) |
| `print_distance_summary` | Print distance summary statistics without plotting. Safe to use when enable_plots=False. | ✅ | [Visualization](/PhenoMe/advanced/api/visualization/#api-_distanceplotsmixin-print_distance_summary) |
| `print_property_stats_by_group` | Print formatted table of grouped property statistics. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-print_property_stats_by_group) |
| `process_images` | Process images through the model and store embeddings. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-process_images) |
| `process_temporal_images` | Process new images in-memory (temporary) and append to the current session. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-process_temporal_images) |
| `reset` | Reset all stored data to a clean state. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-reset) |
| `reset_properties` | Reset only the computed properties. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-reset_properties) |
| `save_results` | Save results to HDF5 (atomic write or in-place flush). | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-save_results) |
| `set_file_df` | Set the internal file DataFrame used for processing. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenome-set_file_df) |
| `top_properties_different_from_reference` | For each non-reference group, return the top k properties that most differentiate it from the reference. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-top_properties_different_from_reference) |
| `transfer_metadata_to_properties` | Transfer specified metadata columns to properties. | ✅ | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-transfer_metadata_to_properties) |

---

## Metadata (`phenome.metadata`)

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `DataFrameMetadata` | Metadata lookup from DataFrame. Supports single and multi-channel modes. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#dataframemetadata) |
| `DefaultMetadata` | Minimal metadata extractor: file_path and filename from path. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#defaultmetadata) |
| `MetadataBase` | Base class for metadata extraction with configurable columns and ID handling. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#metadatabase) |
| `PathTemplateMetadata` | Metadata extractor from path template with capture groups. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#pathtemplatemetadata) |

---

## Utilities (`phenome.utils`)

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `create_blur_effect_function` | Create property function that computes blur strength (Laplacian variance). | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_blur_effect_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_concentric_ring_function` | Create property function that computes stats per concentric ring (edge to core). | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_concentric_ring_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_entropy_function` | Create property function that computes Shannon entropy of intensity distribution. | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_entropy_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_intensity_function` | Create intensity property function over entire image (no mask). | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_intensity_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_masked_intensity_function` | Create intensity property function over masked pixels only (mask > 0.5). | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_masked_intensity_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_regionprops_function` | Create property function that extracts skimage regionprops from a mask. | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_regionprops_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `create_texture_function` | Create property function that computes GLCM texture features within the mask. | ✅ | [Properties](/PhenoMe/advanced/api/properties/#create_texture_function) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `DataFrameMetadata` | Metadata lookup from DataFrame. Supports single and multi-channel modes. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#dataframemetadata) |
| `default_metadata_from_path` | Simple, dataset-agnostic metadata extractor used by default. | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#default_metadata_from_path) · [Experiment Details](/PhenoMe/guides/experiment-details/) · [Metadata](/PhenoMe/advanced/api/metadata/) |
| `DefaultMetadata` | Minimal metadata extractor: file_path and filename from path. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#defaultmetadata) |
| `DinoV2ModelWrapper` | Heritage wrapper for DINOv2 models from Meta AI. | 🔧 | [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/#class-dinov2modelwrapper) |
| `get_default_device` | Return a default device for phenotyping operations. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#get_default_device) |
| `get_metadata_from_path` | Create metadata extractor function from a path template. | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#get_metadata_from_path) · [Experiment Details](/PhenoMe/guides/experiment-details/) · [Metadata](/PhenoMe/advanced/api/metadata/) |
| `get_preset_property_functions` | Return a preset dict of property functions for use with compute_properties. | ✅ | [Properties](/PhenoMe/advanced/api/properties/#get_preset_property_functions) · [Select properties](/PhenoMe/guides/select-properties/) · [Property interpretation](/PhenoMe/concepts/property-interpretation/) |
| `load_dinov2_model` | Load a DINOv2 model and return a DinoV2ModelWrapper ready for processing. | ✅ | [Model-Wrapper](/PhenoMe/advanced/api/model-wrapper/#load_dinov2_model) · [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/#load_dinov2_model) |
| `make_dataframe_metadata_fn` | Build metadata function using a dataframe for metadata lookup. | ✅ | [Utilities](/PhenoMe/advanced/api/utilities/#make_dataframe_metadata_fn) · [Experiment Details](/PhenoMe/guides/experiment-details/) · [Metadata](/PhenoMe/advanced/api/metadata/) |
| `MetadataBase` | Base class for metadata extraction with configurable columns and ID handling. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#metadatabase) |
| `ModelWrapper` | Base wrapper for vision model embedding extraction. | 🔧 | [Model Wrappers](/PhenoMe/advanced/api/model-wrapper/#class-modelwrapper) |
| `normalize_by_dtype_max` | Normalize an image to [0, 1] based on dtype-inferred maximum. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `PadToSize` | Pad image (C, H, W) equally on all sides to pad_size. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `PathTemplateMetadata` | Metadata extractor from path template with capture groups. | ✅ | [Metadata Classes](/PhenoMe/advanced/api/metadata/#pathtemplatemetadata) |
| `scale_minmax` | Apply min-max scaling to each channel independently. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `set_determinism` | Set random seeds for reproducibility across all frameworks. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#set_determinism) |
| `TransformBuilder` | Builds torchvision transform pipelines for image preprocessing. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#transforms--transformbuilder) |
| `TypeMaxNorm` | Normalize image by its dtype-inferred maximum (255 or 65535). | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |

---

## I/O (`phenome.io`)

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `CheckpointManager` | Crash-safe incremental checkpoint backed by a single HDF5 file. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#checkpointmanager) · [HDF5 Protocol](/PhenoMe/advanced/database_protocol/) |
| `ensure_hwc` | Ensure image is in (H, W, C) format. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/#ensure_hwc) |
| `FileDiscovery` | Finds image files and extracts metadata from directories. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#file-discovery--filediscovery) · [HDF5 Protocol](/PhenoMe/advanced/database_protocol/) |
| `read_image` | Read an image and return as a numpy array. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/#read_image) |

---

## Plugins (`phenome.plugins`)

*All advanced — for extending the pipeline.*

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `get_blob_properties` | Extract blob properties using Difference of Gaussians (DoG) blob detection. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#get_blob_properties) · [Extending](/PhenoMe/guides/extending/) · [Select properties](/PhenoMe/guides/select-properties/) |
| `get_metadata_extractor` | Get a registered metadata extractor by name. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#get_metadata_extractor) · [Extending](/PhenoMe/guides/extending/) |
| `get_property` | Get a registered property function by name. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#get_property) · [Extending](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) |
| `get_report_sections` | Return all registered report section generators. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#get_report_sections) · [Extending](/PhenoMe/guides/extending/) |
| `list_properties` | Return names of all registered property functions. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#list_properties) · [Extending](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) |
| `my_custom_max_intensity` | Use as a template for creating custom property functions. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#my_custom_max_intensity) · [Extending](/PhenoMe/guides/extending/) |
| `register_metadata_extractor` | Register a custom metadata extractor. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#register_metadata_extractor) · [Extending](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) |
| `register_property` | Register a custom property function. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#register_property) · [Extending](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) |
| `register_report_section` | Register a custom report section generator. | 🔧 | [Plugins](/PhenoMe/advanced/api/plugins/#register_report_section) · [Extending](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) |

---

## Core (`phenome.core`)

*Advanced helpers for custom scripts extending or bypassing the pipeline.*

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `build_combined_features` | Build combined embedding + property matrix with balanced normalization. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `build_export_dataframe` | Build a DataFrame with the complete dataset for export. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `build_metadata_columns` | Build dict of metadata columns suitable for DataFrame creation. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `clean_correlation_inputs` | Clean correlation inputs by removing NaN/inf values. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_distance_correlation` | Compute distance correlation between x and y. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_entropy` | Compute entropy of data using histogram (PyTorch on GPU/CPU). | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_lasso_interpretability` | Explain a target variable y using a LASSO model on features x. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_mutual_info` | Compute normalized mutual information between x and y. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_pearson_correlation` | Compute Pearson correlation between x and y using torch.corrcoef on GPU/CPU. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_rf_interpretability` | Explain a target variable y using a Random Forest model on features x. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `compute_spearman_correlation` | Compute Spearman rank correlation between x and y using scipy. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `filter_indices` | Return indices that satisfy provided metadata filters and exclusions. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `get_all_metadata_keys` | Get sorted unique metadata keys present in results. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `get_metadata_value` | Safely fetch a metadata value for an image index. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `get_metadata_value_from_dict` | Get value from a single metadata dict with case-insensitive key matching. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `metadata_to_stable_key` | Build a deterministic, portable key from metadata for matching across devices. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `optimize_property_types` | Reduce memory usage of property values. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `PhenoMeProtocol` | Protocol describing attributes and methods mixins expect from PhenoMe. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `PhenoMeResults` | Typed container for per-image phenotyping data. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `prepare_embedding_dataframe` | Create a DataFrame with embedding/reduction coordinates, metadata, and properties. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `run_dimensionality_reduction` | Run dimensionality reduction using pipeline data and return DataFrame. | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `run_dimensionality_reduction_matrix` | Run dimensionality reduction on a raw matrix (pure, no pipeline). | 🔧 | [Utilities](/PhenoMe/advanced/api/utilities/) |
| `validate_results` | Check that pipeline results have consistent lengths across all arrays. | 🔒 | [Utilities](/PhenoMe/advanced/api/utilities/) |

---

## Report (`phenome.report.generator`)

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `ReportConfig` | Configuration for report generation. Pass to generate_report via config=. | ✅ | [Report](/PhenoMe/advanced/api/report/#reportconfig) |
| `generate_report` | Generate comprehensive standalone HTML report from phenotyping results. | ✅ | [Report](/PhenoMe/advanced/api/report/#generate_report) |

---

## Mixins (`phenome.mixins`)

*Prefer using `PhenoMe`; these classes implement its methods.*

| Import | Description | Tier | Extended docs |
| --- | --- | --- | --- |
| `collate_fn` | Custom collate function that handles None values and split/combined channels. | 🔒 | [Pipeline](/PhenoMe/advanced/api/pipeline/) |
| `create_interactive_explorer` | Launch an interactive explorer for phenotyping results in Jupyter. | ✅ | [Interactive Explorer](/PhenoMe/advanced/api/interactive/#create_interactive_explorer) |
| `EmbeddingExtractor` | Extracts embeddings from images using a ModelWrapper and DataLoader. | 🔒 | [Pipeline](/PhenoMe/advanced/api/pipeline/) |
| `PhenoMeAnalysis` | Pipeline providing analysis methods for PhenoMe. | 🔒 | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeanalysis-compute_clustering) |
| `PhenoMeDataset` | Dataset for loading and preprocessing images for phenotyping analysis. | 🔒 | [Pipeline](/PhenoMe/advanced/api/pipeline/) |
| `PhenoMeDistances` | Provides distance computation methods for PhenoMe. | 🔒 | [Distances](/PhenoMe/advanced/api/distances/#api-phenomedistances-compute_reference_distances) |
| `PhenoMeInteractive` | High-performance interactive explorer for PhenoMe results. | ✅ | [Interactive Explorer](/PhenoMe/advanced/api/interactive/#class-phenomeinteractive) |
| `PhenoMeProperties` | Pipeline providing property computation and reporting for PhenoMe. | 🔒 | [Pipeline](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-compute_properties) · [Properties](/PhenoMe/advanced/api/properties/) |
| `PhenoMeVisualization` | Pipeline class providing visualization methods for PhenoMe. | 🔒 | [Visualization](/PhenoMe/advanced/api/visualization/#api-_drplotsmixin-plot_pca) |

---
