---
title: "Common workflows"
description: High-level overview of the phenotyping tasks PhenoMe supports, with links to the canonical notebook for each one.
sidebar:
  order: 5
---


Full code for each workflow lives in the Jupyter notebooks under
[`Notebooks/tutorials/`](https://github.com/AAitorG/PhenoMe/tree/main/Notebooks/tutorials).
This page keeps only a short summary and pointers.

## 1. Basic analysis

The standard end-to-end workflow - from loading images and extracting
visual fingerprints to computing distances and generating reports - is
covered in the core tutorial.

- **Tutorial:** [03 - Core phenotyping workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb)

## 2. Drug screening and time-course

Analysing the effects of different drug treatments or mapping changes
over time requires robust metadata extraction. Whether your data is
organised hierarchically (`{drug}/{concentration}/{image}.tif`) or
encoded in filenames (`Treated_24h`), you can handle it through the
metadata pipeline.

- **Tutorial:** [04 - Advanced metadata handling](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_advanced_metadata_handling.ipynb)
- **Guide:** [Experiment details (metadata)](/PhenoMe/guides/experiment-details/)

## 3. Property-based and explainability analysis

To connect abstract AI patterns to real biology, compute classical
physical properties (area, eccentricity, intensity) and correlate them
with your embeddings.

- **Tutorial:** [05 - Exploratory analysis and explainability](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb)
- **Guide:** [Property reference](/PhenoMe/guides/property-reference/), [Interpretability](/PhenoMe/guides/interpretability/)

## 4. Subpopulations and outlier detection

Discover hidden functional states (clusters) within a single condition
or find anomalous cells dynamically via clustering and distance
thresholds.

- **Tutorial:** [05 - Exploratory analysis and explainability](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb)

## 5. Temporal images (in-memory exploration)

To test how **new** images map onto your pre-calculated PCA / t-SNE
spaces without reprocessing a checkpoint, use temporal processing.

```python
pheno.process_temporal_images(
    wrapper,
    image_paths=["new_data/test1.tif", "new_data/test2.tif"],
)

pheno.plot_pca(color_by="condition", hover_features=["source"])

pheno.clear_temporal_data()
```

## 6. Batch processing large datasets

For datasets that do not fit in memory, enable checkpointing so
embeddings are written to disk as they are computed.

```python
pheno.process_images(
    wrapper,
    checkpoint_path="results/my_large_dataset.h5",
    batch_size=128,
)
```

See [Best practices - performance](/PhenoMe/guides/best-practices/performance/)
for memory sizing.

## See also

| Topic | Where |
|-------|-------|
| Installation and quick start | [Getting started](/PhenoMe/getting-started/) |
| Folder structure and metadata | [Data setup](/PhenoMe/guides/data-setup/), [Experiment details](/PhenoMe/guides/experiment-details/) |
| Embeddings, channels, results | [Core concepts](/PhenoMe/concepts/) |
| Full API reference | [API & data formats](/PhenoMe/advanced/), [Pipeline API](/PhenoMe/advanced/api/pipeline/) |
| Property presets and custom functions | [Property reference](/PhenoMe/guides/property-reference/), [Custom properties](/PhenoMe/guides/custom-properties/) |
| Reproducibility and optimization | [Best practices](/PhenoMe/guides/best-practices/) |
