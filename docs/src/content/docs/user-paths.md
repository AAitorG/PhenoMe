---
title: "Learning paths"
---

Curated routes through PhenoMe documentation and notebooks. Pick **one** path and follow it in order.

---

## Path A — First day (no prior PhenoMe experience)

Goal: run something today and understand the big picture.

1. [Getting started](/PhenoMe/getting-started/) — install, verify `import phenome`, CPU/GPU note.
2. [Quick start card](/PhenoMe/quick-start-card/) — copy-paste minimal script.
3. [Notebook 01 — Interactive quickstart](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/01_interactive_quickstart.ipynb) — widgets, toy or your data.
4. [Core concepts](/PhenoMe/concepts/) — what embeddings and metadata are (skim OK).
5. [FAQ — Getting started](/PhenoMe/faq/#installation-and-setup) — common blockers.

---

## Path B — Your own dataset (scientist / analyst)

Goal: reproducible workflow on real images with sensible metadata.

1. [Getting started](/PhenoMe/getting-started/) — environment locked in.
2. [Data setup](/PhenoMe/guides/data-setup/) — decide if you need metadata; path templates vs CSV.
3. [Notebook 02 — Inspect images](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/02_inspect_your_images.ipynb) — QC.
4. [Notebook 03 — Core workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb) — embeddings, properties, plots.
5. [Property reference](/PhenoMe/guides/property-reference/) — choose a `property_preset`.
6. [Workflows](/PhenoMe/workflows/) or [Examples gallery](/PhenoMe/examples/) — pattern match your study design.
7. Optional: [Notebook 04 — Metadata](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_advanced_metadata_handling.ipynb).

---

## Path C — Reporting and communication

Goal: explain results to collaborators.

1. Complete **Path B** through notebook 03 (or load saved results).
2. [Report guide](/PhenoMe/guides/report-guide/) — what sections mean.
3. `PhenoMe.generate_report` — see [Report API](/PhenoMe/advanced/api/report/).
4. [FAQ — Results](/PhenoMe/faq/#results-and-interpretation).

---

## Path D — Developer / extension author

Goal: safe changes in the codebase and up-to-date API docs.

1. [Architecture](/PhenoMe/guides/architecture/) — packages, data flow, extension points.
2. [Hand-written API index](/PhenoMe/advanced/) — narrative reference.
3. [Developer guide](/PhenoMe/guides/developer-guide/) — docstrings, Astro build, CI.
4. [Testing](/PhenoMe/guides/testing/) — ruff, mypy, interrogate.
5. [Extending PhenoMe](/PhenoMe/guides/extending/) · [Plugins](/PhenoMe/guides/plugins/) · [Custom properties](/PhenoMe/guides/custom-properties/).
6. [Notebook 06 — Plugins](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Path E — Interpretability and exploration

Goal: DR plots, correlations, explainability.

1. [Notebook 05 — Exploratory analysis](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb).
2. [Interpretability guide](/PhenoMe/guides/interpretability/).
3. [Glossary](/PhenoMe/glossary/) — embedding, correlation engine, channel mode.

---

## Quick map: doc vs notebook

| Topic | Read first | Practice in |
|-------|------------|-------------|
| Install | [Getting started](/PhenoMe/getting-started/) | Notebook 01 (Colab cell if needed) |
| Metadata | [Data setup](/PhenoMe/guides/data-setup/) | Notebook 04 |
| Full pipeline | [Getting started §3](/PhenoMe/getting-started/#3-quick-start-your-first-analysis) | Notebook 03 |
| Plugins | [Extending](/PhenoMe/guides/extending/) | Notebook 06 |
