# Local Development Guide

This file collects the bits of PhenoMe development that are not specific to the
documentation site itself.

> **Building the docs?** See [`docs/README.md`](docs/README.md) for the
> preview, build, troubleshooting and CI details. This file only links to it
> and documents a few development conventions.

## TL;DR

- **Python 3.12+**, **Node.js 20+**, **npm 9.6.5+**.
- Install the package editable from the repo root: `pip install -e .` (so
  `import phenome` works when the docs generator runs).
- Preview the docs:

  ```bash
  cd docs
  npm install    # first time only
  npm run dev    # http://localhost:4321/PhenoMe/
  ```

For the full pipeline (sync assets → generate API → fix links → Astro build)
and troubleshooting, read [`docs/README.md`](docs/README.md).

## Updating Documentation

### Narrative Docs (Guides, Examples)

Edit Markdown files directly:

```bash
docs/src/content/docs/guides/*.md
docs/src/content/docs/examples/*.md
docs/src/content/docs/index.md
```

Changes appear in the dev server automatically.

### API Reference (from docstrings)

Edit Python docstrings in the `phenome/` package:

```bash
phenome/pipeline.py          # Main Pipeline class
phenome/mixins/*.py          # Mixins (visualization, properties, etc.)
phenome/utils/*.py           # Utility functions
phenome/plugins/*.py         # Plugin registry
```

Changes appear on the next `npm run dev` or `npm run build` (the pre-scripts
regenerate `docs/src/content/docs/reference/api/`).

**Optional**: Control section grouping with metadata in docstrings:

```python
def my_function():
    """
    @section Advanced Analysis
    @order 10

    Function description here...
    """
```

### Site Configuration

Edit Astro/Starlight configuration:

```bash
docs/astro.config.mjs    # Site title, logo, GitHub Pages base path, sidebar
```

Restart the dev server for changes to take effect.

## Project Structure

```
docs/
├── src/
│   ├── assets/
│   │   └── Logo.png         # Site logo (single file: nav, splash, GitHub README)
│   ├── content/docs/        # Markdown content
│   │   ├── index.mdx        # Home page (splash)
│   │   ├── guides/          # How-to guides
│   │   ├── examples/        # Usage examples
│   │   ├── reference/       # API & database protocol
│   │   │   └── api/         # Auto-generated API docs
│   │   └── *.md             # Other pages (FAQ, glossary, etc.)
│   └── styles/              # Custom CSS
├── scripts/
│   ├── generate_api_docs.py  # API doc generator
│   └── fix_markdown_links.py # Link normalizer
├── public/                  # Static assets (favicon, etc.)
├── astro.config.mjs         # Astro configuration
└── package.json             # npm scripts and dependencies
```

## Questions?

See:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) – General contribution guidelines.
- [`docs/README.md`](docs/README.md) – Documentation setup, build and deploy.
- [`docs/src/content/docs/guides/developer-guide.md`](docs/src/content/docs/guides/developer-guide.md)
  – In-site developer guide for docstring style and the build pipeline.
