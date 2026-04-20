---
title: "Examples gallery"
---

Short **patterns** that map common studies to PhenoMe docs and notebooks. Use them as templates, then adapt paths and column names.

---

## Drug or compound screening

**Goal:** Compare treatments and doses; color plots and reports by compound and concentration.

- **Metadata:** CSV with `filename`, `compound`, `concentration_uM`, optional `plate`, `well`. See [Experiment details — CSV lookup](/PhenoMe/guides/experiment-details/csv-lookup/).
- **Workflow:** [Notebook 03 — Core workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb) then [Notebook 05](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb).
- **Distances / references:** [Distances API](/PhenoMe/advanced/api/distances/), [Workflows](/PhenoMe/workflows/).

---

## Time course (same condition over time)

**Goal:** Track embedding trajectories across time points.

- **Metadata:** Path template with `(timepoint)` or CSV column `time_h`.
- **Plots:** Trajectory / centroid helpers in [Visualization API](/PhenoMe/advanced/api/visualization/) (see narrative sections in that doc).
- **Concepts:** [Core concepts](/PhenoMe/concepts/) for embeddings and checkpoints.

---

## Two cohorts (control vs treated)

**Goal:** Simple two-group comparison without a full screen.

- **Metadata:** Parent folder per group or a minimal CSV with `condition`.
- **Quick path:** [Quick start card](/PhenoMe/quick-start-card/) and `plot_pca(color_by='condition')`.

---

## Custom vision model (not DINOv2)

**Goal:** Swap backbone while keeping the same pipeline.

- **Docs:** [Model wrapper](/PhenoMe/advanced/api/model-wrapper/), [Extending](/PhenoMe/guides/extending/).
- **Notebook:** [06 — Extending plugins](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Custom classical properties

**Goal:** Add biologically meaningful measurements beyond presets.

- **Docs:** [Custom properties](/PhenoMe/guides/custom-properties/), [Property reference](/PhenoMe/guides/property-reference/).
- **Notebook:** [06 — Extending](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Next steps

- [Learning paths](/PhenoMe/user-paths/) for ordered tutorials.
- [FAQ](/PhenoMe/faq/) for edge cases (filters, CSV overlap, GPU OOM).
