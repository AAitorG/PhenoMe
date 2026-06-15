---
title: "Common workflows"
description: Simple overview of what you can do with PhenoMe, with links to the best tutorial for each task.
sidebar:
  order: 5
---


Full code for each workflow lives in the Jupyter notebooks under
[`Notebooks/tutorials/`](https://github.com/AAitorG/PhenoMe/tree/main/Notebooks/tutorials).
This page keeps only a short summary and pointers.

## 1. Basic analysis

The standard end-to-end workflow - from loading images and extracting
visual fingerprints (AI descriptions) to comparing images and generating reports - is
covered in the core tutorial.

- **Tutorial:** [03 - Core phenotyping workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb)

## 2. Drug screening and time-course

Analysing the effects of different drug treatments or mapping changes
over time requires using experimental info (metadata). Whether your data is
organised hierarchically (`{drug}/{concentration}/{image}.tif`) or
encoded in filenames (`Treated_24h`), you can handle it through the
metadata pipeline.

- **Guide:** [Adding experiment info (metadata)](/PhenoMe/guides/experiment-details/)

## 3. Property-based analysis and explaining results

To connect abstract AI patterns to real biology, compute classical physical properties (like area, brightness, and texture) and link them to your visual fingerprints.

- **Tutorial:** [04 - Exploratory analysis and explainability](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_exploratory_analysis_and_explainability.ipynb)
- **Guide:** [Understanding properties](/PhenoMe/concepts/property-interpretation/), [Explaining AI results](/PhenoMe/concepts/embeddings-interpretability/)

## 4. Finding groups and outliers

Discover hidden groups of images (clusters) within a single condition or find unusual cells (outliers) using clustering and similarity checks.

- **Tutorial:** [04 - Exploratory analysis and explainability](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_exploratory_analysis_and_explainability.ipynb)

## 5. Temporal images (testing new data quickly)

To see how **new** images fit into your existing pattern visualizations (PCA/t-SNE) without starting over, use temporal processing.

```python
pheno.process_temporal_images(
    wrapper,
    files=["new_data/test1.tif", "new_data/test2.tif"],
)

pheno.plot_pca(color_by="condition", hover_features=["source"])

pheno.clear_temporal_data()
```

## 6. Batch processing large datasets

For very large datasets, use "checkpointing" so results are saved to disk as they are calculated.

```python
pheno.process_images(
    wrapper,
    checkpoint_path="results/my_large_dataset.h5",
    batch_size=128,
)
```

## 7. Plate / Batch correction

If your data comes from multiple plates or batches, you can correct for technical variation (plate effects) using control wells.

```python
# Correct embeddings using 'Control' wells in each 'Plate'
pheno.correct_batches(
    batch_metadata_key="Plate",
    control_filters={"condition": "Control"},
    method="sphering",  # or "zscore"
    source="embeddings", # or "properties"
)
```

See [Best practices - performance](/PhenoMe/guides/best-practices/performance/)
for memory sizing.

## See also

| Topic | Where |
|-------|-------|
| Installation and quick start | [Getting started](/PhenoMe/getting-started/) |
| Folder structure and metadata | [Preparing your data](/PhenoMe/guides/data-setup/), [Adding experiment info](/PhenoMe/guides/experiment-details/) |
| Visual fingerprints, channel modes, and results | [Core concepts](/PhenoMe/concepts/) |
| Full API reference | [API & data formats](/PhenoMe/advanced/), [Pipeline API](/PhenoMe/advanced/api/pipeline/) |
| Property sets and custom functions | [Select properties](/PhenoMe/guides/select-properties/), [Understanding properties](/PhenoMe/concepts/property-interpretation/) |
| Reproducibility and optimization | [Best practices](/PhenoMe/guides/best-practices/) |
