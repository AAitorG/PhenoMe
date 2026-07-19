---
title: "Contributing to documentation"
---

How to propose documentation changes and keep them consistent with the code.

---

## What lives where

| Audience | Entry |
|----------|--------|
| All users | [Documentation home](/PhenoMe/), [Getting started](/PhenoMe/getting-started/), [Learning paths](/PhenoMe/user-paths/) |
| Data / metadata | [Data setup](/PhenoMe/guides/data-setup/), [Experiment details](/PhenoMe/guides/experiment-details/) |
| Developers | [Architecture](/PhenoMe/guides/architecture/), [Developer guide](/PhenoMe/guides/developer-guide/), [Testing](/PhenoMe/guides/testing/) |
| Auto API | [generate_api_docs.py](https://github.com/AAitorG/PhenoMe/blob/main/docs/scripts/generate_api_docs.py) (writes `advanced/api/` on each build) |

---

## Workflow

1. **Small fixes** (typos, broken links) — open a PR directly.
2. **New behavior** — update **docstrings** first (generated API picks them up), then narrative Markdown if needed.
3. **Large restructures** — discuss in an issue first if navigation changes are broad.

---

## Reporting issues

Use GitHub **Issues** with label `documentation` when possible. Include:

- Page or file path (e.g. `docs/src/content/docs/guides/data-setup.mdx`)
- What you expected vs what you saw
- Environment (OS, Python version, install method)

---

## Review checklist (for maintainers)

- [ ] `npm run build` succeeds under `docs/`
- [ ] New public API has docstrings (interrogate)
- [ ] Spot-check [Pipeline API](/PhenoMe/advanced/api/pipeline/) (generated)

---

## Style

- Prefer **clear headings** and short paragraphs.
- Use **code fences** with language tags for Python and shell.
- Avoid duplicating long signatures; link to [Pipeline API](/PhenoMe/advanced/api/pipeline/) or [API & data formats](/PhenoMe/advanced/) instead.
