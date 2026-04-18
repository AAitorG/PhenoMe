---
title: "Examples gallery"
---

Short **patterns** that map common studies to PhenoMe docs and notebooks. Use them as templates, then adapt paths and column names.

---

## Drug or compound screening

**Goal:** Compare treatments and doses; color plots and reports by compound and concentration.

- **Metadata:** CSV with `filename`, `compound`, `concentration_uM`, optional `plate`, `well`. See [Experiment details — CSV](../guides/experiment-details.md#2-using-a-csv-or-spreadsheet).
- **Workflow:** [Notebook 03 — Core workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb) then [Notebook 05](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb).
- **Distances / references:** [Distances API](../reference/api/distances.md), [Workflows](../workflows.md).

---

## Time course (same condition over time)

**Goal:** Track embedding trajectories across time points.

- **Metadata:** Path template with `(timepoint)` or CSV column `time_h`.
- **Plots:** Trajectory / centroid helpers in [Visualization API](../reference/api/visualization.md) (see narrative sections in that doc).
- **Concepts:** [Core concepts](../concepts.md) for embeddings and checkpoints.

---

## Two cohorts (control vs treated)

**Goal:** Simple two-group comparison without a full screen.

- **Metadata:** Parent folder per group or a minimal CSV with `condition`.
- **Quick path:** [Quick start card](../quick-start-card.md) and `plot_pca(color_by='condition')`.

---

## Custom vision model (not DINOv2)

**Goal:** Swap backbone while keeping the same pipeline.

- **Docs:** [Model wrapper](../reference/api/model-wrapper.md), [Extending](../guides/extending.md).
- **Notebook:** [06 — Extending plugins](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Custom classical properties

**Goal:** Add biologically meaningful measurements beyond presets.

- **Docs:** [Custom properties](../guides/custom-properties.md), [Property reference](../guides/property-reference.md).
- **Notebook:** [06 — Extending](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Next steps

- [Learning paths](../user-paths.md) for ordered tutorials.
- [FAQ](../faq.md) for edge cases (filters, CSV overlap, GPU OOM).
