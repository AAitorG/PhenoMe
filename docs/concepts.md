# Core Concepts

This guide explains the fundamental concepts behind PhenoMe.

[← Documentation index](index.md)

---

## Table of Contents

1. [What is Phenotyping?](#what-is-phenotyping)
2. [Deep Learning Embeddings](#deep-learning-embeddings)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Channel Modes](#channel-modes)
5. [Distance Metrics](#distance-metrics)
6. [Metadata and Properties](#metadata-and-properties)
7. [Visualization Methods](#visualization-methods)

---

## What is Phenotyping?

**Phenotyping** is characterizing observable traits of biological samples. In microscopy, this means analyzing cell morphology, fluorescence patterns, structural changes, and comparing conditions.

PhenoMe automates this using deep learning. It is **model-agnostic** (any vision model producing global representations) and **dataset-agnostic** (you provide your data layout and metadata).

---

## Deep Learning Embeddings

### What are Embeddings?

An **embedding** is a fixed-size numerical vector representing an image. A pre-trained neural network transforms each image into a vector that captures its visual characteristics.

```
Image (H × W × C) → Vision Model → Embedding (D dimensions)
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

## Pipeline Architecture

The pipeline follows a modular design:

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

Use `process_temporal_images()` to add new images **temporarily** to an existing analysis without re-running `process_images`. Temporal images are kept in memory only (not persisted to checkpoint) and are tagged with `metadata['source'] = 'NEW'`. Call `clear_temporal_data()` to remove them. See [Temporal Images](examples/workflows.md#temporal-images-explore-new-data-in-memory) in the workflows guide.

### Embedding Lifecycle

Embeddings can be stored two ways:

| Mode | When | Where | Access |
|------|------|-------|--------|
| **Eager** | No checkpoint used during `process_images()` | `results.embeddings` (in-memory `np.ndarray`) | Direct slice |
| **Lazy** | Checkpoint path provided or `load_results()` used | On disk (HDF5); `results.embeddings` is `None` | `get_embeddings()` loads rows on demand |

Use `get_embeddings()` for lazy-safe access. For full HDF5 storage details, see [Database Protocol](reference/DATABASE_PROTOCOL.md).

---

## Portability and Data Management

A key feature is sharing and moving analysis results across environments. The pipeline converts absolute paths to relative paths in HDF5 checkpoints for portability.

**Moving a Dataset:**
1. Copy the HDF5 checkpoint and image files to the new location
2. Call `find_files` at the new location, then `load_results`:
   ```python
   pheno.find_files("/new/path/to/images")
   pheno.load_results("my_analysis.h5")
   ```
3. The pipeline resolves paths automatically

### Results Container: `PhenoMeResults`

The pipeline uses a typed **`PhenoMeResults`** class (`pheno.results`) for embeddings, metadata, and properties. When you move the dataset, call `find_files` at the new location and the pipeline resolves paths automatically. For structure, properties, and full reference, see [Pipeline API: Results container](reference/api/pipeline.md#results-container-phenomeresults).

---

## Channel Modes

Microscopy images often have multiple channels (e.g., different fluorophores). The pipeline supports two modes:

### Split Mode (Default)

Each channel is processed **independently** through the model, and embeddings are **concatenated**:

```
Image (H x W x 3 channels)
    ├── Channel 0 → Model → Embedding (D)
    ├── Channel 1 → Model → Embedding (D)
    └── Channel 2 → Model → Embedding (D)
                          ↓
              Concatenate → Final Embedding (3×D)
```

**When to use:**
- Fluorescence microscopy with distinct channels (DAPI, GFP, etc.)
- When channel-specific information is important
- Multi-modal imaging

### Combined Mode

All channels are treated as an **RGB image** and processed together:

```
Image (H x W x 3 channels)
    ↓
Vision Model (expects RGB input)
    ↓
Single Embedding (D)
```

**When to use:**
- Brightfield or phase contrast images
- Natural color images
- When channels represent color, not separate signals

### Choosing a Mode

```python
# Split mode (default) - for fluorescence
pheno.process_images(wrapper, channel_mode='split')

# Combined mode - for brightfield/RGB
pheno.process_images(wrapper, channel_mode='combined')
```

---

## Distance Metrics

Distances quantify how different images are from a reference group (e.g., untreated controls).

### Distance Types

| Type | Formula | Use Case |
|------|---------|----------|
| **Euclidean** | √(Σ(a-b)²) | General purpose, magnitude-sensitive |
| **Cosine** | 1 - (a·b)/(‖a‖‖b‖) | Direction-focused, scale-invariant |

### Distance Modes

**Centroid Mode:**
- Compute mean embedding of reference group
- Measure distance of each image to this centroid
- Fast and stable for large reference groups

```python
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    mode='centroid'
)
```

**All-to-All Mode:**
- Compute distance to every reference image
- Report minimum distance
- More sensitive to outliers

```python
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    mode='all_to_all'
)
```

### Interpreting Distances

- **Low distance** = Similar to reference (e.g., no effect)
- **High distance** = Different from reference (e.g., strong phenotypic change)

---

## Metadata and Properties

### Metadata

**Metadata** describes where an image comes from and its experimental context. See [Custom Metadata](guides/custom-metadata.md) for path templates and CSV lookup:

- File path
- Experimental condition (drug, concentration)
- Time point
- Plate/well information
- Replicate number

Example metadata extractor from file paths:

```python
def my_metadata_fn(path):
    # /data/Plate1/Drug_10uM/image_001.tif
    parts = path.split('/')
    return {
        'file_path': path,
        'plate': parts[-3],
        'condition': parts[-2],
        'image_id': parts[-1].replace('.tif', '')
    }
```

### Properties

**Properties** are numerical features computed from the image content. See [Custom Properties](guides/custom-properties.md) and [Property Reference](guides/property-reference.md):

- Average intensity
- Cell area, perimeter
- Texture measurements
- Custom domain-specific features

Properties require user-defined functions:

```python
def avg_intensity(image2d, mask2d):
    return {'mean_intensity': float(np.mean(image2d))}

pheno.find_files("images/", mask_dir="masks/")
df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={"image": avg_intensity}
)
```

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
- [Custom Metadata](guides/custom-metadata.md): Create dataset-specific extractors
- [Custom Properties](guides/custom-properties.md): Define domain-specific features
- [Workflows](examples/workflows.md): End-to-end analysis examples
- [HDF5 Protocol](reference/DATABASE_PROTOCOL.md): Embedding lifecycle and checkpoint structure
