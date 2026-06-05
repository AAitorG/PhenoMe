# Changelog

All notable changes to PhenoMe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.3.0] – 2026-06

### Changed

- **Breaking:** `PhenoMeResults` is attribute-only (`results.embeddings`, `results.img_path`, …); dict-style `__getitem__`, `get`, `keys`, etc. removed
- **Breaking:** Absent or lazy embeddings are `None`, not `[]`
- **Breaking:** `CheckpointManager.write_results_to_hdf5` requires `PhenoMeResults` (not a plain dict)
- **Breaking:** HDF5 checkpoints must store internal tracking flags under `/internal`; keys embedded only in `/properties` are no longer migrated on load
- **Breaking:** Multivariate interpretability display helpers accept `DataFrame` + meta dict only (no dict/tuple result shims)
- Metadata discovery uses `file_path` only (no `filepath` alias)
- Cluster metadata in reports uses lowercase `cluster` only (no `Cluster` alias)
- `normalize_by_dtype_max` and `TypeMaxNorm` share `resolve_intensity_scale`; signed integer images use min-max scaling to [0, 1]
- Split-channel embedding extraction L2-normalizes each channel vector before concatenation
- Property matrix normalization uses per-column NaN-aware mean/std instead of listwise-complete rows

### Fixed

- Signed integer images no longer divide by dtype max without shifting negative values

## [1.2.0] – 2026-05

### Added

- `phenome.core.dataframe_contract` – Tier-A helpers (`per_image_dataframe`, `IMAGE_INDEX`, …)
- `export_experiment_config()` – Export experiment configuration to JSON for reproducibility
- `envs/` – Conda and pip environment files
- `CONTRIBUTING.md`, `CHANGELOG.md`
- API doc: [Results DataFrames](/PhenoMe/advanced/api/results-dataframes/)

### Changed

- **Breaking:** Unified per-image API DataFrames on Tier A: `image_index`, `image_path`, `image_name`, raw metadata keys
- **Breaking:** Slim default per-image returns: `compute_properties`, `compute_clustering`, and `compute_reference_distances` omit metadata columns unless `include_metadata=True`; use `export_dataset_table()` for full tables
- **Breaking:** `compute_clustering` columns `idx`/`path` → `image_index`/`image_path`
- **Breaking:** `detect_outliers` columns `idx`/`file_path` → `image_index`/`image_path`
- **Breaking:** `find_prototypes` uses `image_index` and `rank_in_group` (was `prototype_index`)
- **Breaking:** `analyze_group_enrichment` metric columns snake_case (`property`, `score`, `mean_group`, …)
- **Breaking:** `compute_component_correlation` returns `property` + `component_1`, … columns
- **Breaking:** `compute_reference_distances` and `compute_multivariate_interpretability` return a DataFrame by default; use `return_meta=True` for `(df, meta)` or `(df, meta, fig)` with `return_fig=True`
- **Breaking:** `get_image_info` keys `image_index`, `image_name`, `image_path`
- **Python 3.12+** required
- Unified logging in validation and DR modules

### Fixed

- Interactive explorer color-by cardinality across full dataset
- `prepare_embedding_dataframe` unions property keys across all rows
- `build_export_dataframe` skips misaligned distance columns with a warning
- Centroid reference distances batched for GPU memory
- Checkpoint alignment and TorchDR fallback behavior

## [1.1.0] – 2026-03

### Added

- Lazy embedding loading for large datasets
- Temporal images support (process_temporal_images)
- Plugin registries for properties, metadata extractors, report sections
