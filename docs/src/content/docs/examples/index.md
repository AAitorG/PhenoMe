---
title: "Examples gallery"
---

Short **patterns** that map common studies to PhenoMe docs and notebooks. Use them as templates, then adapt paths and column names.

Full notebooks live under **[`Notebooks/examples/`](https://github.com/AAitorG/PhenoMe/tree/main/Notebooks/examples)** on GitHub.

These walkthroughs are **use-case** examples on public datasets: they show realistic pipelines you can adapt, and they double as **paper reproducibility** notebooks—see the **Paper reproducibility** block in each file (set `PHENOME_*` paths and checkpoints locally). They are **not** formal benchmark competitions or timing leaderboards.

| Notebook | Use case / dataset |
|----------|-------------------|
| [`bbbc014_two_cell_lines.ipynb`](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/examples/bbbc014_two_cell_lines.ipynb) | [BBBC014](https://bbbc.broadinstitute.org/BBBC014) — two lines, dose response |
| [`bbbc021_compound_screen.ipynb`](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/examples/bbbc021_compound_screen.ipynb) | [BBBC021](https://bbbc.broadinstitute.org/BBBC021) — compound / MoA morphological screen |
| [`hpa_subcellular_phenotypes.ipynb`](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/examples/hpa_subcellular_phenotypes.ipynb) | FoV phenotyping patterned on the [Human Protein Atlas](https://www.proteinatlas.org/) resource |
| [`cellcognition_time_lapse.ipynb`](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/examples/cellcognition_time_lapse.ipynb) | Segmented time-lapse / cell-cycle classes |
| [`ecoli_drug_timecourse.ipynb`](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/examples/ecoli_drug_timecourse.ipynb) | *E. coli* CLSM — drug × time screen |

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
- **Quick path:** [Getting started](/PhenoMe/getting-started/#3-run-your-first-analysis) and `plot_pca(color_by='condition')`.

---

## Custom vision model (not DINOv2)

**Goal:** Swap backbone while keeping the same pipeline.

- **Docs:** [Model wrapper](/PhenoMe/advanced/api/model-wrapper/), [Extending](/PhenoMe/guides/extending/).
- **Notebook:** [06 — Extending plugins](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Custom classical properties

**Goal:** Add biologically meaningful measurements beyond presets.

- **Docs:** [Select properties](/PhenoMe/guides/select-properties/), [Property interpretation](/PhenoMe/concepts/property-interpretation/).
- **Notebook:** [06 — Extending](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).

---

## Next steps

- [Learning paths](/PhenoMe/user-paths/) for ordered tutorials.
- [FAQ](/PhenoMe/faq/) for edge cases (filters, CSV overlap, GPU OOM).
