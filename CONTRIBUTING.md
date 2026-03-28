# Contributing to PhenoMe

Thank you for your interest in contributing to PhenoMe. This document explains how to set up a development environment, run the same code quality checks as CI (Ruff, mypy), optionally use pre-commit locally, and propose changes.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AAitorG/PhenoMe.git
   cd PhenoMe
   ```

2. **Create environment**
   Manual setup with conda:
   ```bash
   conda env create -f environment.yml
   conda activate phenome
   ```
   Or with pip:
   ```bash
   conda create -n phenome python=3.13
   conda activate phenome
   conda install pytorch torchvision -c pytorch
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Quality

GitHub Actions runs Gitleaks, Ruff (lint + format check), mypy, and a smoke import on pushes and pull requests to `main` / `master` (see `.github/workflows/ci.yml`).

### Linting and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Lint and auto-fix
ruff check phenome --fix

# Format
ruff format phenome
```

### Type Checking

We use [mypy](https://mypy-lang.org/) for type checking:

```bash
mypy phenome
```

### Security and local paths

Do **not** commit API keys, tokens, passwords, or other secrets. Use environment variables (for example via a local `.env` file; that name is gitignored) or configuration outside the repository. Avoid embedding machine-specific dataset paths, home directories, or internal server paths in examples or scripts that you intend to merge—prefer placeholders such as `path/to/images` or environment-driven paths.

### Reliable Checkpoint Cleanup

Use `checkpoint_context` when you need to ensure the HDF5 file handle is released:

```python
pipeline.find_files("path/to/images")
with pipeline.checkpoint_context("results.h5") as p:
    p.plot_pca(color_by="condition")
# Checkpoint closed automatically
```

## Coding Style

- Follow PEP 8. Ruff enforces style (see `[tool.ruff]` in `pyproject.toml`).
- Use 4 spaces for indentation (see `.editorconfig`).
- Add docstrings to public functions and classes.
- Use type hints where possible.

## Proposing Changes

1. Fork the repository and create a branch: `git checkout -b feature/your-feature`
2. Make your changes. Ensure linting succeeds.
3. Commit with clear messages: `git commit -m "Add: description of change"`
4. Push and open a Pull Request.

## Project Structure

- `phenome/` – Core pipeline and utilities
- `phenome/mixins/` – Mixins (properties, distances, analysis, visualization)
- `phenome/plugins/` – Property and metadata registries
- `src/data/preprocessing/` – Preprocessing scripts
- `docs/` – Documentation

## Common Edit Tasks

Quick reference for maintainers: which file to edit for each type of change.

| To do this | Edit these files |
|------------|------------------|
| Bump version | `pyproject.toml` → `[project].version` (canonical; `phenome.__version__` reads it after install) |
| Add/change dependencies | `pyproject.toml` → `[project].dependencies` or `[project.optional-dependencies]` |
| Update conda environment | `environment.yml` – adjust version pins under `dependencies:` |
| Document a new API | `docs/reference/api/` – add or update the relevant `.md` file |
| Add a guide or tutorial | `docs/guides/` or `docs/examples/` |
| Record a release | `CHANGELOG.md` – move [Unreleased] items under `[X.Y.Z] – YYYY-MM-DD` |
| Change lint/format rules | `pyproject.toml` → `[tool.ruff]`, `[tool.mypy]` |

## Questions

Open an issue on GitHub for questions or discussions.
