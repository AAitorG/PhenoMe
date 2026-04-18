---
title: "Frequently Asked Questions"
---

Common questions grouped by topic. For learning order, see [Learning paths](user-paths.md).

---

## Installation and setup

### Do I need to train a model?

No. PhenoMe is **zero-shot**: it uses pre-trained models like DINOv2 out-of-the-box. You point the pipeline at your images and run. No training, fine-tuning, or manual annotation is required.

### Do I need a GPU?

A GPU is **strongly recommended** for reasonable speed. The pipeline works on CPU, but processing thousands of images will be slow. For small datasets (hundreds of images), CPU is feasible. See [Best Practices: GPU Optimization](guides/best-practices.md#gpu-optimization).

### `conda` / `pip` failed or PyTorch is wrong — what should I try?

1. Use the pinned env files under `envs/` (see [Getting started — Installation options](getting-started.mdx#installation-options)).
2. For GPU, ensure your driver matches the CUDA wheel you install; see [PyTorch get started](https://pytorch.org/get-started/locally/).
3. If `pykeops` fails on your OS, omit it and use slower backends (see [Best practices](guides/best-practices.md)).

### How do I verify the install?

```python
from phenome import PhenoMe
print("PhenoMe is ready!", PhenoMe)
```

See [Getting started](getting-started.mdx).

### Do I need masks?

- **For embeddings**: No. Embeddings are extracted directly from images.
- **For classical properties**: Depends on the preset. The `intensity` preset works on images only (no masks). Presets like `basic`, `regionprops`, and `full` require masks for shape-based features. See [Property Reference](guides/property-reference.md).

### How long does processing take?

Rough order of magnitude (GPU, ViT-B/14, batch_size=32):

- ~1–5 seconds per 100 images (embedding extraction)
- PCA/t-SNE/UMAP: seconds to minutes depending on sample count
- For tens of thousands of images, expect minutes to hours; use checkpointing to resume if interrupted.

---

## Data and metadata

### What folder structure should I use?

The pipeline is flexible:

- **Flat**: All images in one folder. Use `pheno.find_files("path/to/images")`.
- **By condition**: `data/Control/img1.tif`, `data/Drug1/img1.tif`. Use a **path template** (`get_metadata_from_path`) or custom `metadata_fn`.

See [Data setup](guides/data-setup.md) and [Experiment details](guides/experiment-details.md).

### I use a CSV for metadata—why don’t my CSV row count and `file_df` row count match?

`find_files` first scans your images folder, then matches those images to your CSV. **Only the overlap is used:** image files must exist in **both** the folder and the CSV to be processed. Extra rows in your CSV are ignored, and images without a CSV row are skipped by default. See [Experiment details: CSV](guides/experiment-details.md#2-using-a-csv-or-spreadsheet).

### My filter returns no images. Why?

Check that:

1. The metadata keys exist (`pheno.get_available_metadata_keys()`).
2. The filter values match your data exactly (case-sensitive).
3. The column values are as expected (`df['condition'].unique()`).

See [Best Practices: Filter Returns Empty Results](guides/best-practices.md#issue-filter-returns-empty-results).

---

## Analysis and compute

### I get "CUDA out of memory." What do I do?

1. Reduce `batch_size` (e.g., from 32 to 16 or 8).
2. Use a smaller model (e.g., ViT-S instead of ViT-g).
3. Clear GPU memory between runs: `gc.collect(); torch.cuda.empty_cache()`.

See [Best Practices: GPU Out of Memory](guides/best-practices.md#issue-gpu-out-of-memory).

### Processing is very slow. How can I speed it up?

1. Ensure you are using a GPU (`wrapper.device` should show `cuda:X`).
2. Increase `batch_size` if GPU memory allows.
3. Increase `num_workers` for data loading.
4. Use the `intensity` preset instead of `full` if you do not need all properties.

### Results differ between runs. Why?

t-SNE and UMAP are stochastic. For reproducibility, set `seed=42` when creating the pipeline: `PhenoMe(seed=42)`. Also use `use_gpu_for_dr=False` if you need exact match with CPU sklearn/umap-learn. For large datasets on GPU, install `pykeops` to avoid OOM. See [Best Practices: Reproducibility](guides/best-practices.md#reproducibility) and [Dimensionality Reduction Backends](guides/best-practices.md#dimensionality-reduction-backends).

---

## Results and interpretation

### What is the "correlation engine" and how do I use it?

The correlation engine links embedding dimensions (or PCA/UMAP components) to classical properties. It answers: "What morphological feature does this axis encode?" For example, PC1 might correlate with eccentricity (r=0.88), suggesting elongation drives the separation. See [Interpretability Guide](guides/interpretability.md).

### How do I choose channel mode (split vs combined)?

- **Split**: Fluorescence images with distinct channels (DAPI, GFP, etc.). Each channel is processed separately.
- **Combined**: Brightfield, phase contrast, or RGB images. Channels are treated as a single color image.

See [Core Concepts: Channel Modes](concepts.md#channel-modes).

### Which property preset should I use?

| You have...             | Use preset        |
|-------------------------|-------------------|
| No masks                | `intensity`       |
| Masks, want basic stats | `basic`           |
| Masks, shape only       | `regionprops`     |
| Masks, full analysis    | `full` or `full_extended` |

See [Property Reference: Choosing Properties](guides/property-reference.md#choosing-properties).

---

## Extending PhenoMe

### What is the relationship between PhenoMe and the `phenome` package?

**PhenoMe** is the project name. The installable Python package is **`phenome`** (`pip install` from this repo; `import phenome`).

### Where do I add custom models, properties, or plugins?

See [Extending the framework](guides/extending.md), [Custom properties](guides/custom-properties.md), [Plugins](guides/plugins.md), and notebook [06 — Extending](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).
