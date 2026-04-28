---
title: "Glossary"
description: Plain-language definitions of the terms used across PhenoMe.
---

Definitions of key terms used in the PhenoMe documentation, in plain language.

## A–D

### Centroid
The average (mean) position of a group of points in embedding space. When computing "distance to centroid," we measure how far each image's embedding is from the mean embedding of a reference group (e.g., controls).

### Channel
A single color or wavelength layer in a microscopy image. A 3-channel fluorescence image might have DAPI (nucleus), GFP (protein), and mCherry (membrane). The pipeline can process each channel separately (split mode) or combine them as RGB (combined mode).

### Channel mode
How multi-channel images are handled. **Split mode**: each channel is processed independently; embeddings are concatenated. **Combined mode**: channels are stacked as RGB and processed together. Use split for fluorescence; use combined for brightfield or natural color.

### Checkpoint
A saved state of your analysis (embeddings, metadata, properties) stored in HDF5 format. Enables resuming after interruption and sharing results.

### Classical properties
Hand-crafted numerical features (area, intensity, eccentricity, etc.) computed from images and masks. Interpretable and used as the "vocabulary" to explain what embeddings encode.

### CLS token
In Vision Transformers (ViTs), a special token whose output represents a summary of the entire image. Used as the embedding vector.

### Correlation coefficient (r)
A number between -1 and 1 measuring how two variables move together. Positive r: both increase together. Negative r: one increases as the other decreases. Near 0: little linear relationship. Used to link embedding dimensions to classical properties for interpretability.

### Dimensionality reduction (visualizing patterns)
Reducing complex high-dimensional information (like 768 numbers describing an image) into simpler representations (2 or 3 numbers) so we can plot them on a map. Methods: PCA, t-SNE, UMAP.

### DINOv2
A self-supervised vision model by Meta AI. Learns visual features without labels. Default example model in the pipeline.

### Distance metric
How "far apart" two embeddings are. **Euclidean**: straight-line distance. **Cosine**: angle between vectors (scale-invariant).

---

## E–M

### Embedding
A fixed-size numerical vector that represents an image. Produced by a deep learning model; captures visual characteristics in a compact form.

### Metadata
Information about each image: file path, condition, drug, timepoint, replicate, etc. Extracted from file paths or CSV.

### Model-agnostic
The pipeline works with any vision model that produces embeddings (DINOv2, CLIP, BioCLIP, custom models). No lock-in to a single architecture.

### Morphometry
Quantitative measurement of shape and form. Classical morphometry uses hand-crafted features (area, perimeter, eccentricity); the pipeline also uses deep embeddings.

---

## P–R

### PCA (Principal Component Analysis)
Linear dimensionality reduction. Finds directions of maximal variance. Fast, deterministic. Good for a quick overview.

### Phenotypic distance
A scalar value indicating how different an image is from a reference group (e.g., controls). Higher distance = more phenotypically different.

### Phenotyping
Characterizing observable traits (phenotypes) of biological samples, e.g., cell morphology, fluorescence patterns, response to treatments.

### Property preset
A predefined set of classical properties: `basic`, `regionprops`, `intensity`, `full`, `full_extended`, or `none`. See [Property interpretation](/PhenoMe/concepts/property-interpretation/).

### Reference group
The set of images used as baseline (e.g., untreated controls). Defined by `reference_filters` when computing phenotypic distances.

### Report
A standalone HTML file with interactive plots, distance analysis, clustering, correlations, and image galleries. Shareable without re-running the pipeline.

---

## S–Z

### Split mode
See [Channel mode](#channel-mode). Each channel processed independently.

### t-SNE
Non-linear dimensionality reduction. Preserves local structure; good for clustering visualization. Can be stochastic (results vary with seed).

### UMAP
Non-linear dimensionality reduction. Balances global and local structure; often faster than t-SNE. Default for many exploratory workflows.

### Zero-shot
Using a pre-trained model without training or fine-tuning on your data. DINOv2 and similar models work out-of-the-box for phenotyping.

---

## See also

- **Visual fingerprints and labels** - [Core concepts](/PhenoMe/concepts/), [Preparing your data](/PhenoMe/guides/data-setup/)
- **Channel / split mode** - [Concepts — channel modes](/PhenoMe/concepts/channel-modes/)
- **Checkpoints and HDF5** - [HDF5 protocol](/PhenoMe/advanced/database_protocol/), [Best practices](/PhenoMe/guides/best-practices/)
- **Linking AI to biology** - [Explaining AI results](/PhenoMe/concepts/embeddings-interpretability/)
