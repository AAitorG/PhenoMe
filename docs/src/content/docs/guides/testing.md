---
title: "Testing and local quality checks"
---

Commands used in CI and recommended before opening a PR.

---

## Install dev tools

```bash
pip install -e ".[dev]"
```

---

## Lint and format

```bash
python -m ruff check phenome tests
python -m ruff format phenome tests --check
```

Auto-fix (local only):

```bash
python -m ruff check phenome tests --fix
python -m ruff format phenome tests
```

---

## Type checking

```bash
python -m mypy phenome
```

---

## Docstring coverage

```bash
interrogate phenome
```

The project targets **≥95%** coverage (`pyproject.toml` → `[tool.interrogate]`).

---

## Spell check

```bash
codespell
```

---

## Documentation build

```bash
cd docs
npm install
npm run build
```

---

## Dependency compatibility smoke

```bash
python -m pip check
python -m pytest
```

These match the main CI workflow. `pip check` fails if installed packages conflict with each other. `pytest` also:

- Imports the public PhenoMe API and every `phenome.*` module
- Constructs `PhenoMe` on CPU, reads PNG/NPY images, and runs a tiny find → embed → properties → clustering loop
- Reloads embeddings from an HDF5 checkpoint and checks that advertised property presets resolve

A new library version that breaks an import, I/O, or the core pipeline is caught even when the top-level `PhenoMe` class still loads.
