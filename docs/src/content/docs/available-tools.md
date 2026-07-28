---
title: "Available tools"
description: Single-table summary of PhenoMe capabilities — formats, embeddings, analysis, and visualization.
---

Capability map of what PhenoMe can do. For method names and signatures, see the
[Function location guide](/PhenoMe/function-location-guide/) and the
[API reference](/PhenoMe/advanced/).

Most analysis methods accept a feature `source`: `"embeddings"`, `"properties"`, or `"combined"`.

| Category | Tools / options | Notes |
|----------|-----------------|-------|
| **Image I/O** | TIFF, NIfTI, NumPy (`.npy`/`.npz`), PNG/JPEG/BMP | `tifffile`, `nibabel`, `numpy`, OpenCV; discovery also matches many microscopy extensions (e.g. `.czi`, `.nd2`) without dedicated readers |
| **Embeddings** | DINOv2 (built-in); custom via `ModelWrapper` | e.g. `dinov2_vitb14_reg`; channel modes `"split"` / `"combined"` |
| **Dimensionality reduction** | PCA, t-SNE, UMAP | CPU: sklearn + umap-learn; optional GPU: TorchDR (`use_gpu_for_dr=True`) |
| **Clustering** | k-means, GMM, DBSCAN | Optional normalize, silhouette, and PCA/t-SNE/UMAP pre-reduce |
| **Distances & similarity** | Euclidean, cosine; prototypes; outliers | Reference modes `centroid` / `all_to_all`; outliers via z-score or IQR |
| **Group enrichment** | Leave-one-out z-scores; Welch + BH-FDR; property stats | Also top properties vs reference (`cohens_d` / `mean_diff`) |
| **Properties** | Presets: none, intensity, shape, basic, standard, complete | Backends: scikit-image, scipy, numpy; factories for regionprops, intensity, texture, rings, blur, entropy; blob plugin |
| **Batch correction** | Sphering, z-score | Control-based per batch/plate (`correct_batches`) |
| **Interpretability** | Pearson, Spearman, dCor, MI; lasso, random forest | Correlate or explain DR/embedding axes with properties |
| **Visualization & reporting** | Plotly plots; matplotlib image grids; Jupyter explorer; HTML report | DR scatters, enrichment, correlations, gallery, etc. |
| **Preprocessing & I/O** | Transforms; L2 / StandardScaler; HDF5 checkpoints; CSV/Parquet/Excel export | Orchestrated on [`PhenoMe`](/PhenoMe/advanced/api/pipeline/) |

**See also:** [Embeddings](/PhenoMe/concepts/embeddings/), [Distances](/PhenoMe/concepts/distances/), [Select properties](/PhenoMe/guides/select-properties/), [Visualization & results](/PhenoMe/concepts/visualization-and-results/), [Embeddings interpretability](/PhenoMe/concepts/embeddings-interpretability/).
