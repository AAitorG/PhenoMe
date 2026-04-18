---
title: "Contributing to documentation"
---

How to propose documentation changes and keep them consistent with the code.

---

## What lives where

| Audience | Entry |
|----------|--------|
| All users | [Documentation home](..), [Getting started](../getting-started.mdx), [Learning paths](../user-paths.md) |
| Data / metadata | [Data setup](data-setup.md), [Experiment details](experiment-details.md) |
| Developers | [Architecture](architecture.md), [Developer guide](developer-guide.md), [Testing](testing.md) |
| Auto API | [generate_api_docs.py](https://github.com/AAitorG/PhenoMe/blob/main/docs/scripts/generate_api_docs.py) (writes `reference/api/` on each build) |

---

## Workflow

1. **Small fixes** (typos, broken links) — open a PR directly.
2. **New behavior** — update **docstrings** first (generated API picks them up), then narrative Markdown if needed.
3. **Large restructures** — discuss in an issue first if navigation changes are broad.

---

## Reporting issues

Use GitHub **Issues** with label `documentation` when possible. Include:

- Page or file path (e.g. `docs/src/content/docs/guides/data-setup.md`)
- What you expected vs what you saw
- Environment (OS, Python version, install method)

---

## Review checklist (for maintainers)

- [ ] `npm run build` succeeds under `docs/`
- [ ] New public API has docstrings (interrogate)
- [ ] Spot-check generated pages under [Reference / API](../reference/api/pipeline.md)

---

## Style

- Prefer **clear headings** and short paragraphs.
- Use **code fences** with language tags for Python and shell.
- Avoid duplicating long signatures; link to [generated API](../reference/api/pipeline.md) or [reference overview](../reference/) instead.
