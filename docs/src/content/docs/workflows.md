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

Analyzing the effects of different drug treatments or mapping changes
over time requires using experimental info (metadata). Whether your data is
organized hierarchically (`.../(drug)/(concentration)/(filename).*`) or
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

For very large datasets, discover files first, then checkpoint during extraction. With a `checkpoint_path`, the file is written every 5 batches (`save_every=5`).

```python
pheno.find_files("path/to/your/images")
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

## 8. Other modalities (experimental)

A volume, an audio clip, a text file, or another non-image file uses a new session. `find_files` takes the file extensions. `set_data_io(load_fn, show_fn)` stores a reader and a picture of shape `(H, W)` or `(H, W, C)`. `process_data` writes one embedding per file. Equal shapes are stacked. Different shapes pass `collate_fn`. DINOv2 stays on `process_images`. Draw a row with `plot_data_by_index`.

```python
pm.find_files("volumes", extensions=[".npy"])
pm.set_data_io(load_volume, show_volume)
pm.process_data(model, batch_size=4)
pm.plot_pca()
pm.plot_data_by_index(0)
```

Copy-paste loaders for a 3D volume, a wav folder, and a text folder are in [Audio, text, and 3D](/PhenoMe/guides/experimental-samples/). Practice the same order in [Tutorial 07](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/07_experimental_samples.ipynb).

## See also

| Topic | Where |
|-------|-------|
| Installation and quick start | [Getting started](/PhenoMe/getting-started/) |
| Volumes, audio, text, and other files | [Audio, text, and 3D](/PhenoMe/guides/experimental-samples/) |
| Folder structure and metadata | [Preparing your data](/PhenoMe/guides/data-setup/), [Adding experiment info](/PhenoMe/guides/experiment-details/) |
| Visual fingerprints, channel modes, and results | [Core concepts](/PhenoMe/concepts/) |
| Full API reference | [API & data formats](/PhenoMe/advanced/), [Pipeline API](/PhenoMe/advanced/api/pipeline/) |
| Property sets and custom functions | [Select properties](/PhenoMe/guides/select-properties/), [Understanding properties](/PhenoMe/concepts/property-interpretation/) |
| Reproducibility and optimization | [Best practices](/PhenoMe/guides/best-practices/) |
