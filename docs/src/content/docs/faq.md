---
title: "FAQ"
description: Grouped answers to the most common PhenoMe questions - install, data, compute, results, and extension.
---

Common questions grouped by topic.

---

## Installation and setup

### Do I need to train a model?

No. PhenoMe is **zero-shot**: it uses pre-trained AI models like DINOv2 out-of-the-box. You point the pipeline at your images and run. No training, fine-tuning, or manual annotation is required.

### Do I need a GPU?

A GPU is **strongly recommended** for reasonable speed. The pipeline works on CPU, but processing thousands of images will be slow. For small datasets (hundreds of images), CPU is feasible. See [Best practices - performance](/PhenoMe/guides/best-practices/performance/).

### `conda` / `pip` failed or PyTorch is wrong - what should I try?

1. Use the pinned env files under `envs/` (see [Getting started - installation](/PhenoMe/getting-started/#2-installation)).
2. For GPU, ensure your driver matches the CUDA wheel you install; see [PyTorch get started](https://pytorch.org/get-started/locally/).
3. If `pykeops` fails on your OS, omit it and use slower backends (see [Best practices - performance](/PhenoMe/guides/best-practices/performance/)).

### I get a `[KeOps] Warning : CUDA libraries not found or could not be loaded; Switching to CPU only` error, what should I do?

This happens if `pykeops` is installed but cannot find your system's CUDA toolkit or compiler. In most cases, you can ignore this: PhenoMe will automatically fall back to standard PyTorch or Scikit-Learn backends for visualizing patterns (using PCA, t-SNE, or UMAP), which is fast enough for most datasets.

If you are working with very large datasets (e.g., >100,000 images) and need the memory efficiency of KeOps, ensure you have a functional C++ compiler and that your `CUDA_HOME` environment variable is set. For standard usage, you can safely ignore the warning or simply uninstall `pykeops`.

If you have a working CUDA setup but still see the error, try clearing the cache:

```python
import pykeops
pykeops.clean_pykeops()
```

### How do I verify the install?

See the verification steps in [Getting started — Installation](/PhenoMe/getting-started/#2-installation).

### Do I need masks?

- **For embeddings**: No. These are extracted directly from images.
- **For classical properties**: Depends on the preset. The `intensity` preset works on images only (no masks). Presets like `basic`, `regionprops`, and `full` require masks for shape-based features. See [Understanding properties](/PhenoMe/concepts/property-interpretation/).

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

See [Preparing your data](/PhenoMe/guides/data-setup/) and [Adding experiment info](/PhenoMe/guides/experiment-details/).

### I use a CSV for metadata—why don’t my CSV row count and `file_df` row count match?

`find_files` first scans your images folder, then matches those images to your CSV. **Only the overlap is used:** image files must exist in **both** the folder and the CSV to be processed. Extra rows in your CSV are ignored, and images without a CSV row are skipped by default. See [Adding experiment info — Using a spreadsheet](/PhenoMe/guides/experiment-details/csv-lookup/).

### My filter returns no images. Why?

Check that:

1. The metadata keys exist (`pheno.get_available_metadata_keys()`).
2. The filter values match your data exactly (case-sensitive).
3. The column values are as expected (`df['condition'].unique()`).

See [Best practices - data organization](/PhenoMe/guides/best-practices/data-organization/).

---

## Analysis and compute

### I get "CUDA out of memory." What do I do?

1. Reduce `batch_size` (e.g., from 32 to 16 or 8).
2. Use a smaller model (e.g., ViT-S instead of ViT-g).
3. Clear GPU memory between runs: `gc.collect(); torch.cuda.empty_cache()`.

See [Best practices - performance](/PhenoMe/guides/best-practices/performance/#gpu-out-of-memory).

### Processing is very slow. How can I speed it up?

1. Ensure you are using a GPU (`wrapper.device` should show `cuda:X`).
2. Increase `batch_size` if GPU memory allows.
3. Increase `num_workers` for data loading.
4. Use the `intensity` preset instead of `full` if you do not need all properties.

### Results differ between runs. Why?

t-SNE and UMAP have random elements (they are stochastic). For reproducibility, set `seed=42` when creating the pipeline: `PhenoMe(seed=42)`. Also use `use_gpu_for_dr=False` if you need exact match with CPU sklearn/umap-learn. For large datasets on GPU, install `pykeops` to avoid OOM. See [Best practices - reproducibility](/PhenoMe/guides/best-practices/reproducibility/) and [Best practices - performance](/PhenoMe/guides/best-practices/performance/).

---

## Results and interpretation

### What is the "correlation engine" (linking AI to biology) and how do I use it?

The correlation engine links the abstract AI descriptions to classical physical properties (like size or shape). It answers: "What physical trait does this AI pattern represent?" For example, a certain axis might represent how elongated a cell is. See [Explaining AI results](/PhenoMe/concepts/embeddings-interpretability/).

### How do I choose channel mode (split vs combined)?

- **Split**: Fluorescence images with distinct channels (DAPI, GFP, etc.). Each channel is processed separately.
- **Combined**: Brightfield, phase contrast, or RGB images. Channels are treated as a single color image.

See [Core concepts — channel modes](/PhenoMe/concepts/channel-modes/).

### Which property preset should I use?

| You have...             | Use preset        |
|-------------------------|-------------------|
| No masks                | `intensity`       |
| Masks, want basic stats | `basic`           |
| Masks, shape only       | `regionprops`     |
| Masks, full analysis    | `full` or `full_extended` |

See [Understanding properties — choosing properties](/PhenoMe/concepts/property-interpretation/#choosing-properties).

---

## Extending PhenoMe

### What is the relationship between PhenoMe and the `phenome` package?

**PhenoMe** is the project name. The installable Python package is **`phenome`** (`pip install` from this repo; `import phenome`).

### Where do I add custom models, properties, or plugins?

See [Extending PhenoMe](/PhenoMe/guides/extending/), [Select properties](/PhenoMe/guides/select-properties/), [Plugins](/PhenoMe/guides/plugins/), and notebook [06 - Extending](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_extending_phenome_plugins.ipynb).
