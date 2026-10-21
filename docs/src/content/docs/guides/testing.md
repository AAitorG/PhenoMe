---
title: "Testing and local quality checks"
---

Commands to run before a pull request. GitHub Actions runs Gitleaks, Ruff, mypy, and `python -m pytest`. Docstring coverage is a pre-commit hook. Spell check is local only.

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

The project targets **≥95%** coverage (`pyproject.toml` → `[tool.interrogate]`). The pre-commit hook enforces that threshold. The GitHub Actions job does not run interrogate.

---

## Spell check

```bash
codespell
```

`codespell` is not part of the GitHub Actions workflow or the pre-commit hooks. Run it locally when you want a spelling pass.

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
python -m pytest
```

This matches the main CI workflow. The suite:

- Runs `pip check` so conflicting installed packages fail
- Imports the public PhenoMe API and every `phenome.*` module
- Constructs `PhenoMe` on CPU, reads PNG/NPY images, and runs a tiny find → embed → properties → clustering loop
- Reloads embeddings from an HDF5 checkpoint and checks that advertised property presets resolve
