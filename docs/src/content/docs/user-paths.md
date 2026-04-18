---
title: "Learning paths"
---

Curated routes through PhenoMe documentation and notebooks. Pick **one** path and follow it in order.

---

## Path A — First day (no prior PhenoMe experience)

Goal: run something today and understand the big picture.

1. [Getting started](getting-started.mdx) — install, verify `import phenome`, CPU/GPU note.
2. [Quick start card](quick-start-card.md) — copy-paste minimal script.
3. [Notebook 01 — Interactive quickstart](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/01_interactive_quickstart.ipynb) — widgets, toy or your data.
4. [Core concepts](concepts.md) — what embeddings and metadata are (skim OK).
5. [FAQ — Getting started](faq.md#installation-and-setup) — common blockers.

---

## Path B — Your own dataset (scientist / analyst)

Goal: reproducible workflow on real images with sensible metadata.

1. [Getting started](getting-started.mdx) — environment locked in.
2. [Data setup](guides/data-setup.md) — decide if you need metadata; path templates vs CSV.
3. [Notebook 02 — Inspect images](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/02_inspect_your_images.ipynb) — QC.
4. [Notebook 03 — Core workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb) — embeddings, properties, plots.
5. [Property reference](guides/property-reference.md) — choose a `property_preset`.
6. [Workflows](workflows.md) or [Examples gallery](examples/index.md) — pattern match your study design.
7. Optional: [Notebook 04 — Metadata](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_advanced_metadata_handling.ipynb).

---

## Path C — Reporting and communication

Goal: explain results to collaborators.

1. Complete **Path B** through notebook 03 (or load saved results).
2. [Report guide](guides/report-guide.md) — what sections mean.
3. `PhenoMe.generate_report` — see [Report API](reference/api/report.md).
4. [FAQ — Results](faq.md#results-and-interpretation).

---

## Path D — Developer / extension author

Goal: safe changes in the codebase and up-to-date API docs.

1. [Architecture](guides/architecture.md) — packages, data flow, extension points.
2. [Hand-written API index](reference/) — narrative reference.
3. [Developer guide](guides/developer-guide.md) — docstrings, Astro build, CI.
4. [Testing](guides/testing.md) — ruff, mypy, interrogate.
5. [Extending PhenoMe](guides/extending.md) · [Plugins](guides/plugins.md) · [Custom properties](guides/custom-properties.md).
6. [Notebook 06 — Plugins](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Path E — Interpretability and exploration

Goal: DR plots, correlations, explainability.

1. [Notebook 05 — Exploratory analysis](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb).
2. [Interpretability guide](guides/interpretability.md).
3. [Glossary](glossary.md) — embedding, correlation engine, channel mode.

---

## Quick map: doc vs notebook

| Topic | Read first | Practice in |
|-------|------------|-------------|
| Install | [Getting started](getting-started.mdx) | Notebook 01 (Colab cell if needed) |
| Metadata | [Data setup](guides/data-setup.md) | Notebook 04 |
| Full pipeline | [Getting started §3](getting-started.mdx#3-quick-start-your-first-analysis) | Notebook 03 |
| Plugins | [Extending](guides/extending.md) | Notebook 06 |
