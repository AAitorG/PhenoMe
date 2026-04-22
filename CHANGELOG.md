# Changelog

All notable changes to PhenoMe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `export_experiment_config()` – Export experiment configuration (seed, reference_filters, model name, etc.) to JSON for reproducibility
- `envs/` – Conda (`environment-cpu.yml`, `environment-gpu.yml`) and pip requirements (CPU/GPU + shared `requirements-base.txt`)
- `CONTRIBUTING.md` – Contribution guidelines
- `CHANGELOG.md` – Version history

### Changed

- **Python 3.12+** required (`requires-python`, Trove classifiers); Ruff `target-version` and Mypy `python_version` set to 3.12 for consistency with CI and modern dependencies (e.g. `tifffile` typing syntax)
- Unified logging: `results_validation` and `dimensionality_reduction` now use the package logger (notebook-safe)
- Pre-commit: Removed Black; use Ruff for linting and formatting only
- `.editorconfig`: Indent style changed from tabs to spaces for PEP 8 compliance

### Fixed

- Environment: Simplified `environment-gpu.yml` to use standard backends; `pykeops` is now optional for extreme-scale datasets.
- `prepare_embedding_dataframe` unions property keys across rows (not only row 0)
- TorchDR→CPU fallback only on likely GPU/transient failures; `ValueError` always propagates; integer `random_state` for TorchDR PCA/TSNE/UMAP
- Pearson correlation: constant `x` columns yield NaN; 1D result is always a length-1 `ndarray`
- Checkpoint alignment: strict `zip` for metadata vs paths; duplicate path/key warnings; `path_repr` preserves multi-channel order; NIfTI detection limited to `.nii` / `.nii.gz`
- `EmbeddingExtractor` checkpoint `embedding_dim` from last feature dimension with 2D requirement
- `build_export_dataframe` skips misaligned `distances` with a warning
- Centroid reference distances batched to reduce GPU OOM risk
- Logging in Jupyter: validation and DR modules now respect the package logger configuration

## [1.1.0] – 2026-03

### Added

- HDF5 checkpoint protocol v2.0 (DATABASE_PROTOCOL.md)
- Lazy embedding loading for large datasets
- Temporal images support (process_temporal_images)
- Plugin registries for properties, metadata extractors, report sections
