---
title: "Collaboration"
description: Share PhenoMe experiments with collaborators, version data with DVC, and keep environments aligned.
sidebar:
  order: 4
---

## Sharing experiments

Generate a self-contained bundle:

```python
pheno.save_results("experiment_2026_01.h5")

pheno.generate_report(
    output_path="experiment_2026_01/report.html",
    reference_filters={"condition": "Control"},
)

pheno.export_experiment_config(
    "experiment_2026_01/config.json",
    reference_filters={"condition": "Control"},
)
```

A collaborator reloads it with:

```python
pheno = PhenoMe()
pheno.find_files("/path/to/experiment_2026_01/images")
pheno.load_results("experiment_2026_01/phenome_results.h5")
```

:::tip
Pair the bundle with the environment spec so colleagues install the same
stack - see [`envs/environment-gpu.yml`](https://github.com/AAitorG/PhenoMe/blob/main/envs/environment-gpu.yml).
:::

## Recommended shared project layout

See [Data organization - recommended project layout](/PhenoMe/guides/best-practices/data-organization/#recommended-project-layout).
For collaboration, add `configs/` (template configs) and environment
specs under `envs/` (conda YAML and pip requirements); the repo root
keeps a short
[`requirements.txt`](https://github.com/AAitorG/PhenoMe/blob/main/requirements.txt)
that defaults to CPU.

## Data versioning with DVC (optional)

**When to use DVC.** Large raw image datasets, checkpoint / HDF5 result
files, and reproducibility of data lineage across machines.

**What the checkpoint already provides.** The HDF5 from `save_results()`
or `checkpoint_path` stores:

- `processing_params` (`channel_mode`, `resize_size`, `pad_size`, `force_rgb`).
- `version` (format version).
- Embeddings, metadata, properties.

You do **not** need to duplicate this configuration elsewhere. The
checkpoint is self-contained for processing state.

**How DVC complements the checkpoint.** DVC handles **data and assets**
(versioning, storage, sharing). The checkpoint handles **processing
state** (what was computed, with which params). They work together:

```bash
dvc add data/
dvc add results/phenome_results.h5

git add data.dvc results/phenome_results.h5.dvc
git commit -m "Add data and results"
```

**Summary.** DVC does not replace the checkpoint. Use the checkpoint for
crash-safe processing and parameter traceability; use DVC for data
versioning and large-file storage.

## Naming conventions

- Use consistent metadata keys across experiments (for example
  `condition`, `replicate`, `timepoint`).
- Name config files by experiment: `drug_screen_2026.json`.
- Commit configs alongside scripts, not checkpoints - checkpoints are
  regenerable.

## See also

- [Reproducibility](/PhenoMe/guides/best-practices/reproducibility/) - seed and DR-backend choice.
- [External checkpoints](/PhenoMe/guides/external-checkpoints/) - exchange
  embeddings with non-PhenoMe tooling.
- [Report API](/PhenoMe/advanced/api/report/) - `generate_report()` and the primary HTML artefact shared
  with non-technical reviewers.
