---
title: "Core concepts"
description: What embeddings, channel modes, distances, metadata, and properties mean in PhenoMe.
sidebar:
  order: 4
---

This page explains the ideas behind PhenoMe. Skim it first - the rest of
the documentation assumes the vocabulary introduced here.

## What is phenotyping?

**Phenotyping** characterises the observable traits of biological samples.
In microscopy this means analysing cell shape (morphology), where
proteins are located (fluorescence patterns), or how structures change
under treatment.

PhenoMe automates this with deep learning. It is **model-agnostic** (it
works with almost any vision model) and **dataset-agnostic** (it does not
care how your files are organised).

## Visual fingerprints (embeddings)

An **embedding** is a digital "fingerprint" of an image: a fixed-size
vector of numbers that captures the most important visual features -
textures, shapes, patterns - without requiring a human to define them.

```text
Image -> Vision model (the "brain") -> Visual fingerprint (the numbers)
```

### Supported models

Common choices:

- **DINOv2** (default): self-supervised vision transformer by Meta AI.
- **CLIP**: vision-language models.
- **MAE**: masked autoencoders.
- **Custom models**: any model exposing a `forward_features()` method.

DINOv2 transfers well to diverse microscopy images without training or
fine-tuning, which is why it is the default.

### Embedding dimensions

| Model | Embedding dimension |
|-------|---------------------|
| ViT-S/14 (Small) | 384 |
| ViT-B/14 (Base) | 768 |
| ViT-L/14 (Large) | 1024 |
| ViT-g/14 (Giant) | 1536 |

For custom models see [Vision model wrappers](/PhenoMe/advanced/api/model-wrapper/).

## Framework architecture

PhenoMe is organised as a core orchestrator surrounded by specialised
mixins. You interact with a single `PhenoMe` instance; internally each
responsibility lives in its own module.

**How the pieces connect:** discovery feeds **embedding extraction** and
**property** computation. **Distances** use embeddings and/or properties.
**Analysis** builds on properties (and embeddings where relevant); **visualization**
consumes analysis and distance outputs; **interactive** tooling and the
**HTML report** sit on top of analysis and visualization.

### Processing flow

![Pipeline Overview](../../assets/pipeline_overview.png)

PhenoMe processes data through a bifurcated 4-step pipeline (visualized above):

1.  **`find_files()`** (Input) — Discover images and extract experimental metadata.
2.  **`process_images()`** (A) — Load images, apply transforms, and extract deep learning embeddings (visual fingerprints).
3.  **`compute_properties()`** (B) — Compute interpretable classical features (shape, texture, intensity) in parallel.
4.  **Analysis & Visualization** (C) — The correlation engine integrates both sources to quantify phenotypic distances and produce interactive plots and reports.

### Temporal images (in-memory)

Use `process_temporal_images()` to add new images **temporarily** to an
existing analysis without re-running `process_images`. Temporal images
are kept in memory only (not persisted to checkpoint) and are tagged
with `metadata['source'] = 'NEW'`. Call `clear_temporal_data()` to remove
them. See [Temporal images](/PhenoMe/workflows/#5-temporal-images-in-memory-exploration).

### Embedding lifecycle: eager vs lazy

| Mode | When | Where | Access |
|------|------|-------|--------|
| **Eager** | No checkpoint used during `process_images()` | `results.embeddings` (in-memory `np.ndarray`) | Direct slice |
| **Lazy** | Checkpoint path provided or `load_results()` used | On disk (HDF5); `results.embeddings` is `None` | `get_embeddings()` loads rows on demand |

Use `get_embeddings()` for lazy-safe access. For full HDF5 storage
details, see the [Database protocol](/PhenoMe/advanced/database_protocol/).

## Portability and data management

Sharing and moving analysis results across environments is a core design
goal. Checkpoints store **relative** POSIX paths so they travel between
machines.

**Moving a dataset:**

1. Copy the HDF5 checkpoint and image files to the new location.
2. Call `find_files` at the new location, then `load_results`:

   ```python
   pheno.find_files("/new/path/to/images")
   pheno.load_results("my_analysis.h5")
   ```

3. The pipeline resolves paths automatically.

### Results container: `PhenoMeResults`

A typed `PhenoMeResults` object (`pheno.results`) holds `img_path`,
`metadata`, `properties`, and `embeddings`. When you move the dataset,
call `find_files` at the new location and the pipeline resolves paths
automatically. See
[API: PhenoMeResults](/PhenoMe/advanced/api/pipeline/#class-phenomeresults).

## Channel modes

Microscopy images often have multiple channels (e.g. DAPI for the
nucleus, GFP for a specific protein). PhenoMe offers two ways to handle
this.

### Split mode (recommended for fluorescence)

Each channel is processed **separately**, then results are combined.
This ensures the model captures channel-specific information (e.g. if
only one protein changes localisation).

### Combined mode (recommended for brightfield / RGB)

All channels are treated as a single colour image and processed together.

### Choosing a mode

```python
pheno.process_images(wrapper, channel_mode="split")     # fluorescence
pheno.process_images(wrapper, channel_mode="combined")  # brightfield / RGB
```

## Distance metrics

Once images are converted to embeddings, we measure mathematical distance
between them to quantify phenotypic change.

- **Low distance** = similar to the reference (e.g. no treatment effect).
- **High distance** = different from the reference (e.g. strong
  phenotypic change).

### Common methods

- **Euclidean vs cosine**: Euclidean measures absolute straight-line
  distance; cosine measures the angle between vectors (often better for
  high-dimensional embeddings because it ignores magnitude).
- **Centroid vs all-to-all**: compare each image to the **average**
  (centroid) of the reference group (fast and stable), or to **every
  individual** reference image (more sensitive but slower).

For a complete guide, see [Common workflows](/PhenoMe/workflows/).

## Metadata and properties

### Metadata

**Metadata** links an image to its experimental context:

- Which drug was applied?
- At what concentration?
- Which plate well is this from?
- What timepoint was this taken at?

By organising images with metadata, PhenoMe can colour plots by drug
concentration, calculate distances relative to controls, and compare
across plates. See [Experiment details](/PhenoMe/guides/experiment-details/).

### Properties

Where **embeddings** are abstract vectors from a deep learning model,
**properties** are classical, explainable image measurements (cell
count, average intensity, nuclear area, ...).

PhenoMe computes and stores properties alongside embeddings, so you can
ask whether simple properties (like nucleus size) already explain the
grouping a deep-learning model sees.

See [Custom properties](/PhenoMe/guides/custom-properties/) for user-defined
metrics and [Property reference](/PhenoMe/guides/property-reference/) for the
built-ins.

## Visualization methods

### Dimensionality reduction

High-dimensional embeddings (768+ dimensions) are reduced to 2D/3D for
plotting:

:::note
GPU-accelerated DR (`torchdr`) may differ slightly from CPU due to
numerical precision. For strict reproducibility use `use_gpu_for_dr=False`.
See [Best practices - reproducibility](/PhenoMe/guides/best-practices/reproducibility/).
:::

| Method | Description | Best for |
|--------|-------------|----------|
| **PCA** | Linear projection preserving variance | Quick overview, global structure |
| **t-SNE** | Non-linear, preserves local structure | Clustering, local relationships |
| **UMAP** | Non-linear, preserves global + local | General purpose, faster than t-SNE |

```python
pheno.plot_pca(n_components=2, color_by="condition")
pheno.plot_tsne(n_components=3, color_by="condition")
pheno.plot_umap(n_neighbors=15, min_dist=0.1, color_by="drug")

pheno.plot_centroids(group_by="condition", method="pca")
pheno.plot_centroids(group_by=["drug", "time"], method="pca")
```

### Distance distributions

```python
pheno.plot_distance_distribution(
    distance_results=dist_results,
    group_by="condition",
    kde=True,  # smooth density curves
)
```

### Data sources

| `source` | Uses |
|----------|------|
| `"embeddings"` | Embedding vectors (default) |
| `"properties"` | Computed classical properties |
| `"combined"` | Concatenation of both |

```python
pheno.plot_pca(source="properties", property_keys=["area", "intensity"])
```

## Results structure

Results live in `PhenoMeResults` (`pheno.results`): `img_path`,
`metadata`, `properties`, and `embeddings` (see
[Embedding lifecycle](#embedding-lifecycle-eager-vs-lazy) for when
`embeddings` is in-memory vs lazy).

### Accessing results

| Access | When to use |
|--------|-------------|
| `pheno.get_embeddings()` | Preferred. Loads from disk when using checkpoint/lazy; returns `None` if no embeddings. |
| `pheno.export_dataset_table()` | Metadata, properties, and distances as a DataFrame. Use for filtering, grouping, external analysis. |
| `pheno.get_image_info(idx)` | Full metadata, properties, and optional distance for one image. |

Results are saved and loaded as HDF5 via `save_results()` and
`load_results()`. Use `checkpoint_path` in `process_images` for
crash-safe, resumable runs.

## Next steps

- [API reference](/PhenoMe/advanced/api/pipeline/) - method documentation.
- [Getting started](/PhenoMe/getting-started/) - installation and quick start.
- [Experiment details](/PhenoMe/guides/experiment-details/) - dataset-specific extractors.
- [Custom properties](/PhenoMe/guides/custom-properties/) - domain-specific features.
- [Workflows](/PhenoMe/workflows/) - end-to-end analysis examples.
- [HDF5 protocol](/PhenoMe/advanced/database_protocol/) - embedding lifecycle and checkpoint structure.
