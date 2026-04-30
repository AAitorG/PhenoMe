---
title: "Best practices"
description: Reproducibility, performance, data organization, and collaboration patterns for PhenoMe.
sidebar:
  order: 0
---

This section collects habits that keep PhenoMe analyses **reproducible**,
**fast enough for the hardware you have**, and **easy to share**. Each
topic is a short focused page:

| Topic | What you'll find |
|-------|------------------|
| [Reproducibility](/PhenoMe/guides/best-practices/reproducibility/) | Seeds, deterministic mode, saving configurations, DR backend choice. |
| [Performance](/PhenoMe/guides/best-practices/performance/) | GPU sizing, batch size, memory management, troubleshooting OOM and slow runs. |
| [Data organization](/PhenoMe/guides/best-practices/data-organization/) | Directory layout, naming conventions, inspection, filtering. |
| [Batch correction](/PhenoMe/workflows/#7-plate--batch-correction) | Correcting for technical variation (plate effects) using controls. |
| [Collaboration](/PhenoMe/guides/best-practices/collaboration/) | Sharing experiments, DVC, environment specs. |

## Quick reference

| Topic | Key tip |
|-------|---------|
| Reproducibility | `PhenoMe(seed=42)`; use `export_experiment_config()` after `save_results()`. |
| DR backends | `use_gpu_for_dr=False` for strict reproducibility (sklearn / umap-learn). |
| DR GPU OOM | Install `pykeops` for memory-efficient t-SNE / UMAP; PCA uses batched `ExactIncrementalPCA`. |
| GPU OOM | Reduce `batch_size`, pick a smaller model, process in chunks. |
| Portability | HDF5 stores relative paths - call `find_files` first, then `load_results`. |
| Batch correction | Use `pheno.correct_batches(batch_metadata_key="Plate", control_filters={"condition": "Control"})`. |
| Data inspection | Run `pheno.inspect_data()` before processing (after `find_files`). |
