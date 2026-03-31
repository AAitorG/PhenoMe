# Core Concepts

This guide explains the fundamental concepts behind PhenoMe.

[← Documentation index](index.md)

---

## Table of Contents

1. [What is Phenotyping?](#what-is-phenotyping)
2. [Deep Learning Embeddings](#deep-learning-embeddings)
3. [Framework Architecture](#framework-architecture)
4. [Channel Modes](#channel-modes)
5. [Distance Metrics](#distance-metrics)
6. [Metadata and Properties](#metadata-and-properties)
7. [Visualization Methods](#visualization-methods)

---

## What is Phenotyping?

**Summary:** Identifying and measuring the visible characteristics of cells or tissues.

Phenotyping is the process of characterizing the observable traits of biological samples. In microscopy, this means analyzing cell shape (morphology), where proteins are located (fluorescence patterns), or how structures change when treated with a drug.

PhenoMe automates this using deep learning. It is **model-agnostic** (it works with almost any vision model) and **dataset-agnostic** (it doesn't care how your files are organized).

---

## Visual Fingerprints (Embeddings)

**Summary:** Turning an image into a list of numbers that describes its appearance.

### What are Embeddings?

An **embedding** is like a digital "fingerprint" of an image. It is a fixed-size list of numbers (a vector) that captures the most important visual features of an image—such as textures, shapes, and patterns—without needing a human to define them.

Think of it this way: instead of describing a person by their height and hair color, an embedding is like a detailed DNA profile that captures everything about their appearance in a compact form.

```
Image → Vision Model (The "Brain") → Visual Fingerprint (The Numbers)
```

### Supported Models

Common choices:
- **DINOv2** (default): Self-supervised vision transformer by Meta AI
- **CLIP**: Vision-language models
- **MAE**: Masked Autoencoders
- **Custom models**: Any model with `forward_features()` method

DINOv2 is popular for strong transfer learning to diverse microscopy images without training or fine-tuning.

### Embedding Dimensions

| Model | Embedding Dimension |
|-------|---------------------|
| ViT-S/14 (Small) | 384 |
| ViT-B/14 (Base) | 768 |
| ViT-L/14 (Large) | 1024 |
| ViT-g/14 (Giant) | 1536 |

See [Vision Model Wrappers](reference/api/model-wrapper.md) for custom models.

---

## Framework Architecture

The framework follows a modular design:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         PhenoMe                                │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐           │
│  │PhenoMeProperties│ │ PhenoMeDistances│ │ PhenoMeVisualization│           │
│  │                  │ │                │ │                    │           │
│  │ compute_properties│ Centroid        │ │ PCA, t-SNE, UMAP   │           │
│  │ filter_properties │ All-to-all      │ │ Distance plots     │           │
│  └─────────────────┘ │ Euclidean      │  │ Image display      │           │
│                      │ Cosine         │  └─────────────────────┘           │
│  ┌─────────────────┐ └────────────────┘                                  │
│  │ PhenoMeAnalysis │                                                      │
│  │                  │                                                      │
│  │ compute_clustering│                                                     │
│  │ detect_outliers  │                                                      │
│  │ find_prototypes  │                                                      │
│  └─────────────────┘                                                      │
│                                                                            │
│  Core: find_files() → process_images(wrapper, ...) → analyze/visualize   │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### Processing Flow

1. **find_files()**: Discover images and extract metadata
2. **process_images()**: Load images, apply transforms, extract embeddings
3. **compute_reference_distances()**: Calculate phenotypic distances
4. **Visualize**: Create PCA/t-SNE/UMAP plots, distance distributions
5. **generate_report()**: Produce standalone HTML report

### Temporal Images (In-Memory)

Use `process_temporal_images()` to add new images **temporarily** to an existing analysis without re-running `process_images`. Temporal images are kept in memory only (not persisted to checkpoint) and are tagged with `metadata['source'] = 'NEW'`. Call `clear_temporal_data()` to remove them. See [Temporal Images](workflows.md#temporal-images-explore-new-data-in-memory) in the workflows guide.

### Embedding Lifecycle

Embeddings can be stored two ways:

| Mode | When | Where | Access |
|------|------|-------|--------|
| **Eager** | No checkpoint used during `process_images()` | `results.embeddings` (in-memory `np.ndarray`) | Direct slice |
| **Lazy** | Checkpoint path provided or `load_results()` used | On disk (HDF5); `results.embeddings` is `None` | `get_embeddings()` loads rows on demand |

Use `get_embeddings()` for lazy-safe access. For full HDF5 storage details, see [Database Protocol](reference/DATABASE_PROTOCOL.md).

---

## Portability and Data Management

A key feature is sharing and moving analysis results across environments. The framework converts absolute paths to relative paths in HDF5 checkpoints for portability.

**Moving a Dataset:**
1. Copy the HDF5 checkpoint and image files to the new location
2. Call `find_files` at the new location, then `load_results`:
   ```python
   pheno.find_files("/new/path/to/images")
   pheno.load_results("my_analysis.h5")
   ```
3. The pipeline resolves paths automatically

### Results Container: `PhenoMeResults`

The framework uses a typed **`PhenoMeResults`** class (`pheno.results`) for embeddings, metadata, and properties. When you move the dataset, call `find_files` at the new location and the framework resolves paths automatically. For structure, properties, and full reference, see [API Reference: Results container](reference/api/pipeline.md#results-container-phenomeresults).

---

## Channel Modes

**Summary:** How images with multiple "colors" (fluorophores) are processed.

Microscopy images often have multiple channels (e.g., DAPI for the nucleus, GFP for a specific protein). PhenoMe offers two ways to handle this:

### Split Mode (Recommended for Fluorescence)

Each channel is processed **separately** and then the results are combined. This ensures that the model captures the unique information in each channel (e.g., if only one protein changes localization).

### Combined Mode (Recommended for Brightfield/RGB)

All channels are treated as a single color image and processed together.

### Choosing a Mode

```python
# Split mode (default) - for fluorescence
pheno.process_images(wrapper, channel_mode='split')

# Combined mode - for brightfield/RGB
pheno.process_images(wrapper, channel_mode='combined')
```

---

## Distance Metrics

**Summary:** Quantifying how different images are from a reference group (e.g., untreated controls).

Once images are converted to embeddings, we can measure the mathematical distance between them to understand phenotypic changes.

- **Low distance** = Conceptually similar to the reference (e.g., no treatment effect).
- **High distance** = Conceptually different from the reference (e.g., strong phenotypic change).

### Common Distance Methods

- **Euclidean vs. Cosine**: Euclidean measures absolute straight-line distance, while Cosine measures the angle between vectors (often better for high-dimensional embeddings because it ignores magnitude).
- **Centroid vs. All-to-All**: You can either compare an image to the *average* (centroid) of the reference group (fast and stable), or compare it to *every individual* reference image (more sensitive but slower).

For a complete guide on computing and analyzing these metrics, see the **[Common Workflows](workflows.md)**.

---

## Metadata and Properties

**Summary:** The facts about your experiment and the hard numbers from your images.

### Metadata

**Metadata** links an image to its experimental context. It answers questions like:
- Which drug was applied?
- At what concentration?
- Which plate well is this from?
- What timepoint was this taken at?

By organizing images with this metadata, PhenoMe allows you to automatically color plots by drug concentration, calculate distances relative to controls, and compare results across plates. Let PhenoMe automatically extract this information for you.

See the **[Experiment Details Pipeline](guides/experiment-details.md)** to learn how to inject this metadata.

### Properties

While *embeddings* are abstract numbers generated by Deep Learning, **Properties** are classical, explainable image measurements (e.g., cell count, average intensity, nuclear area).

PhenoMe computes and stores these properties alongside the embeddings. This allows you to find out if standard properties (like nucleus size) can fully explain the grouping seen by the deep learning embeddings.

See the **[Custom Properties](guides/custom-properties.md)** guide to learn how to add your own metrics, or check the built-in properties in the **[Property Reference](guides/property-reference.md)**.

---

## Visualization Methods

### Dimensionality Reduction

High-dimensional embeddings (768+ dimensions) are reduced to 2D/3D for visualization:

> **Note:** GPU-accelerated DR (`torchdr`) may differ slightly from CPU due to numerical precision. For strict reproducibility, use `use_gpu_for_dr=False`. See [Best Practices](guides/best-practices.md#dimensionality-reduction-backends).

| Method | Description | Best For |
|--------|-------------|----------|
| **PCA** | Linear projection preserving variance | Quick overview, global structure |
| **t-SNE** | Non-linear, preserves local structure | Clustering, local relationships |
| **UMAP** | Non-linear, preserves global+local | General purpose, faster than t-SNE |

```python
# 2D PCA colored by condition
pheno.plot_pca(n_components=2, color_by='condition')

# 3D t-SNE
pheno.plot_tsne(n_components=3, color_by='condition')

# UMAP with custom parameters
pheno.plot_umap(n_neighbors=15, min_dist=0.1, color_by='drug')

# Centroids in reduced space
pheno.plot_centroids(group_by='condition', method='pca')
pheno.plot_centroids(group_by=['drug', 'time'], method='pca')
```

### Distance Distributions

Visualize how distances vary across experimental groups:

```python
pheno.plot_distance_distribution(
    distance_results=dist_results,
    group_by='condition',
    kde=True  # Smooth density curves
)
```

### Data Sources for Visualization

Plots can use two data sources:

- **`source='embeddings'`**: Use embedding vectors (default)
- **`source='properties'`**: Use computed properties

```python
# Plot based on extracted properties
pheno.plot_pca(source='properties', property_keys=['area', 'intensity'])
```

---

## Results Structure

Results live in **`PhenoMeResults`** (`pheno.results`): `img_path`, `metadata`, `properties`, `embeddings` (see [Embedding Lifecycle](#embedding-lifecycle-eager-vs-lazy) for when `embeddings` is in-memory vs lazy-loaded).

### Accessing Results

| Access | When to use |
|--------|-------------|
| `pheno.get_embeddings()` | **Preferred for embeddings.** Loads from disk when using checkpoint/lazy loading; returns `None` if no embeddings. |
| `pheno.export_dataset_table()` | Metadata, properties, and distances as a DataFrame. Use for filtering, grouping, and external analysis. |
| `pheno.get_image_info(idx)` | Full metadata, properties, and optional distance for one image. |

Results are saved/loaded as **HDF5** via `save_results()` and `load_results()`. Use `checkpoint_path` in `process_images` for crash-safe, resumable runs. See [Moving a Dataset](#moving-a-dataset) for relocating the dataset.

---

## Next Steps

- [API Reference](reference/api/pipeline.md): Complete method documentation
- [Getting Started](getting-started.md): Installation and quick start
- [Experiment Details](guides/experiment-details.md): Create dataset-specific extractors
- [Custom Properties](guides/custom-properties.md): Define domain-specific features
- [Workflows](workflows.md): End-to-end analysis examples
- [HDF5 Protocol](reference/DATABASE_PROTOCOL.md): Embedding lifecycle and checkpoint structure
