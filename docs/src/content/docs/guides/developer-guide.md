---
title: "Developer guide: code and documentation"
---

How to keep PhenoMe maintainable and how **API documentation** stays aligned with the code.

---

## Docstrings as the source of truth

- Public modules and classes should have **NumPy-style** docstrings (see [NumPy doc guide](https://numpydoc.readthedocs.io/en/latest/format.html)).
- Explain **tensor/array shapes** where non-obvious (e.g. `(N, D)` embeddings).
- The repo enforces high docstring coverage with **interrogate** (`fail-under=95` in `pyproject.toml`).

Run locally from the repo root:

```bash
pip install -e ".[dev]"
interrogate phenome
```

---

## Build the docs site

The public site uses **Astro Starlight**. Narrative guides live under [`docs/src/content/docs/`](https://github.com/AAitorG/PhenoMe/tree/main/docs/src/content/docs). **API reference** pages under [`reference/api/`](https://github.com/AAitorG/PhenoMe/tree/main/docs/src/content/docs/reference/api) are produced from live imports by `docs/scripts/generate_api_docs.py` before each build (same style as the former hand-written topic pages, but sourced from docstrings).

### Prerequisites

- **Node.js** 20+ and **npm**
- **Python** 3.12+ with the package installed editable (`pip install -e .`) so docstrings can be imported

### Local build

From the `docs/` directory:

```bash
cd docs
npm install
npm run build
```

`npm run build` runs `python3 scripts/generate_api_docs.py` first, then `astro build`. Output is written to `docs/dist/`. Preview locally:

```bash
npm run preview
```

### What gets auto-generated?

The script [`docs/scripts/generate_api_docs.py`](https://github.com/AAitorG/PhenoMe/blob/main/docs/scripts/generate_api_docs.py) writes one Markdown file per **topic** under `docs/src/content/docs/reference/api/` (e.g. `pipeline.md`, `visualization.md`). When you add a major new public surface, extend that script so the new API appears on the right page.

| Topic page | Main Python sources |
|------------|---------------------|
| `pipeline.md` | `PhenoMe` (excluding viz / distances / interactive-only methods), `PhenoMeResults` |
| `visualization.md` | `PhenoMeVisualization` |
| `distances.md` | `PhenoMeDistances` |
| `properties.md` | `phenome.utils.property_factories` |
| `model-wrapper.md` | `phenome.utils.model_wrapper` |
| `metadata.md` | `phenome.metadata` |
| `utilities.md` | `phenome.utils.device`, `phenome.io`, `TransformBuilder` |
| `plugins.md` | `phenome.plugins` |
| `report.md` | `ReportConfig`, `generate_report` in `phenome.report.generator` |
| `interactive.md` | `create_interactive_explorer`, `PhenoMeInteractive` |

Docstrings may start with optional `@section` / `@order` lines to override the built-in grouping for that page. Legacy `api-reference/` output is no longer produced.

---

## CI

Pull requests run the **Documentation** workflow: Node + Python install, `npm run build`, then deploy to **GitHub Pages** on pushes to `main` / `master`. Fix broken internal links before merging.

---

## Optional checks

- **Preview:** `npm run preview` after a production build.
- **Doctest:** the project does not run doctest on all modules by default; add targeted tests under `tests/` when adding executable examples in docstrings.

---

## Related

- [Contributing to docs](contributing-guide.md)
- [Testing and quality](testing.md)
- [Contributing (repository)](https://github.com/AAitorG/PhenoMe/blob/main/CONTRIBUTING.md)
