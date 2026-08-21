# Changelog

All notable changes to PhenoMe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.4.0] – 2026-08

### Added

- Optional MCP extra (`pip install "phenome[mcp]"`) and `phenome-mcp` stdio server: search docs/recipes, list the public API, inspect a folder (no embeddings), and format an exported config JSON
- Silent run log of pipeline steps; opt-in dump via `get_run_settings()`, `export_methods_markdown()`, richer `export_experiment_config()` JSON, and `generate_report(..., include_run_settings=True)`
- Agent/docs surface: `AGENTS.md`, [Using with LLMs](/PhenoMe/guides/using-with-llms/), [MCP for PhenoMe](/PhenoMe/guides/mcp/), and `llms.txt` / `llms-small.txt` / `llms-full.txt` on the docs site
- `sample_size` on dimensionality-reduction plots so large sessions can subsample before loading embeddings
- `channel_names` on `find_files` / `set_file_df` so multi-channel properties use labels like `_DAPI` instead of `_ch0`
- Optional `progress_callback` on `process_images` and `compute_properties` for GUI/notebook progress bars (`phenome.utils.report_progress`)

### Changed

- Checkpoint resume matches paths with portable keys (suffix/basename) only when the key is unique on both the session and the checkpoint
- Property checkpoints can resume into a larger file: session rows map onto checkpoint indices, extra checkpoint keys are kept, and additive properties merge instead of wiping the table
- **Breaking:** `find_files` fails closed on duplicate sample `id` values (distinct files must have distinct ids)
- **Breaking:** multi-channel groups require integer `channel_index` values `0..n-1` with no gaps
- **Breaking:** `correct_batches` maps blank/NaN batch labels to `__missing__` and integer-like floats to `str(int)`
- **Breaking:** temporal data from `process_temporal_images()` is in-memory only; checkpoints never store those rows as embeddings or properties. `save_results` now skips them on both sides.
- Default `Normalize` channel count follows dataset output when `force_rgb=False` (split → 1, explicit `channels` → `len(channels)`)
- `collate_fn` keeps failed sample identities as `(None, item)` instead of dropping the row silently
- Docs site copy, generated API pages, and the function-location guide updated for the new methods

### Fixed

- Sequential `compute_properties(..., checkpoint_path=..., n_jobs=1)` no longer crashes after `save_every` images (`tqdm` had replaced the indexable path list)
- Aligned property resume no longer clears and rewrites the full HDF5 property table on every incremental flush (rewrite once at finalize)
- Portable path matches no longer drop property rows when session and checkpoint canonical paths differ
- `generate_report()` no longer overwrites existing `compute_clustering()` labels with report defaults (`n_clusters=5`); it only pre-computes clustering when none are stored
- HTML report table headers and cluster labels are escaped
- Default log level is unchanged: per-sample load/property failures stay at DEBUG
- API doc generation still writes the narrative API pages if only the function-location guide fails

## [1.3.0] – 2026-06

### Added

- `PhenoMe.load_embeddings()` and `PhenoMe.load_properties()` – load separate embeddings and properties HDF5 checkpoints into one pipeline (e.g. `*_embeddings.h5` then `*_properties.h5`)

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
