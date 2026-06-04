---
title: "Visualization & Results"
description: Dimensionality reduction, results structure, and portability.
---

## Dimensionality reduction

High-dimensional embeddings (768+ dimensions) are reduced to 2D/3D for plotting:

:::note
GPU-accelerated DR (`torchdr`) may differ slightly from CPU due to numerical precision. For strict reproducibility use `use_gpu_for_dr=False`. See [Best practices - reproducibility](/PhenoMe/guides/best-practices/reproducibility/).
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
```

## Data sources

| `source` | Uses |
|----------|------|
| `"embeddings"` | Embedding vectors (default) |
| `"properties"` | Computed classical properties |
| `"combined"` | Concatenation of both |

```python
pheno.plot_pca(source="properties", property_keys=["area", "intensity"])
```

## Results structure

Results live in `PhenoMeResults` (`pheno.results`): `img_path`, `metadata`, `properties`, and `embeddings`.

Use **attribute access** (not dict-style `results["key"]` or `results.get()`):

```python
paths = pheno.results.img_path
meta = pheno.results.metadata
props = pheno.results.properties
emb = pheno.results.embeddings  # np.ndarray when eager; None if absent or lazy (HDF5)
```

### Accessing results

| Access | When to use |
|--------|-------------|
| `pheno.get_embeddings()` | Preferred. Loads from disk when using checkpoint/lazy; returns `None` if no embeddings. |
| `pheno.results.embeddings` | Same underlying data when stored eagerly in memory; `None` when not computed or lazy-backed. |
| `pheno.export_dataset_table()` | Metadata, properties, and distances as a DataFrame. Use for filtering, grouping, external analysis. |
| `pheno.get_image_info(idx)` | Full metadata, properties, and optional distance for one image. |

Analysis methods return [per-image DataFrames](/PhenoMe/advanced/api/results-dataframes/) (Tier A) with `image_index`, `image_path`, and `image_name` so tables join on one schema.

Results are saved and loaded as HDF5 via `save_results()` and `load_results()`.

## Portability and data management

Sharing and moving analysis results across environments is a core design goal. Checkpoints store **relative** POSIX paths so they travel between machines.

**Moving a dataset:**

1. Copy the HDF5 checkpoint and image files to the new location.
2. Call `find_files` at the new location, then `load_results`:

   ```python
   pheno.find_files("/new/path/to/images")
   pheno.load_results("my_analysis.h5")
   ```

3. The pipeline resolves paths automatically.
