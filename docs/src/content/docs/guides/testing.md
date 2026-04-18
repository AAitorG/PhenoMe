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
python -m ruff check phenome
python -m ruff format phenome --check
```

Auto-fix (local only):

```bash
python -m ruff check phenome --fix
python -m ruff format phenome
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

## Smoke import

```bash
python -c "from phenome import PhenoMe; print(PhenoMe)"
```

This matches the lightweight check in the main CI workflow.
