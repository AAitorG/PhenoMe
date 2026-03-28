# Changelog

All notable changes to PhenoMe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `export_experiment_config()` – Export experiment configuration (seed, reference_filters, model name, etc.) to JSON for reproducibility
- `environment.yml` – Conda environment with pinned dependencies for reproducible installs
- `CONTRIBUTING.md` – Contribution guidelines
- `CHANGELOG.md` – Version history

### Changed

- Unified logging: `results_validation` and `dimensionality_reduction` now use the package logger (notebook-safe)
- Pre-commit: Removed Black; use Ruff for linting and formatting only
- `.editorconfig`: Indent style changed from tabs to spaces for PEP 8 compliance

### Fixed

- Logging in Jupyter: validation and DR modules now respect the package logger configuration

## [1.1.0] – 2026-03

### Added

- HDF5 checkpoint protocol v2.0 (DATABASE_PROTOCOL.md)
- Lazy embedding loading for large datasets
- Temporal images support (process_temporal_images)
- Plugin registries for properties, metadata extractors, report sections
