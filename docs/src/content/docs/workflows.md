---
title: "Common Workflows"
---

**Author:** [Aitor González-Marfil](https://github.com/AAitorG) (@AAitorG)

This page provides an overview of common phenotyping tasks. To minimize redundancy, full code examples and advanced implementations for these workflows are maintained directly in our interactive Jupyter Notebook tutorials.

---

## 1. Basic Analysis Workflow

The standard end-to-end workflow—from loading images and extracting visual fingerprints to computing distances and generating reports—is fully covered in our core tutorial.

➡️ **See Tutorial:** [03_core_phenotyping_workflow.ipynb](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb)

---

## 2. Drug Screening & Time-Course Analysis

Analyzing the phenotypic effects of different drug treatments or mapping changes over time requires robust metadata extraction. Whether your data is organized hierarchically (`{drug}/{concentration}/{image}.tif`) or encoded in file names (`Treated_24h`), you can handle it smoothly using our experiment details pipeline.

➡️ **See Tutorial:** [04_advanced_metadata_handling.ipynb](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_advanced_metadata_handling.ipynb)
➡️ **See Guide:** [Experiment Details (Metadata)](/PhenoMe/guides/experiment-details/)

---

## 3. Property-Based & Explainability Analysis

To connect abstract AI patterns to real biology, you can compute classic physical properties (e.g., area, eccentricity) and analyze how they correlate with your embeddings.

➡️ **See Tutorial:** [05_exploratory_analysis_and_explainability.ipynb](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb)
➡️ **See Guide:** [Property Reference](/PhenoMe/guides/property-reference/)

---

## 4. Subpopulations & Outlier Detection

Discovering hidden functional states (clusters) within a single condition or finding anomalous cells dynamically is supported through clustering and distance thresholds.

➡️ **See Tutorial:** [05_exploratory_analysis_and_explainability.ipynb](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb)

---

## 5. Temporal Images (In-Memory Exploration)

If you have an existing dataset and want to test how *new* images map onto your pre-calculated PCA/t-SNE spaces without reprocessing your entire checkpoint, PhenoMe supports temporal image processing.

```python

pheno.process_temporal_images(
    wrapper,
    image_paths=["new_data/test1.tif", "new_data/test2.tif"]
)

pheno.plot_pca(color_by='condition', hover_features=['source'])

pheno.clear_temporal_data()
```

---

## 6. Batch Processing Large Datasets

For processing huge datasets that don't fit in memory, you can enable checkpointing to continuously save extracted embeddings to disk.

```python

pheno.process_images(
    wrapper,
    checkpoint_path="results/my_large_dataset.h5",
    batch_size=128
)
```

---

## See Also

| Topic | Document |
|-------|----------|
| Installation and quick start | [Getting Started](/PhenoMe/getting-started/) |
| Folder structure and metadata | [Expected Folder Structure](/PhenoMe/getting-started/#3-expected-folder-structure), [Experiment Details](/PhenoMe/guides/experiment-details/) |
| Embeddings, channels, results | [Core Concepts](/PhenoMe/concepts/) |
| Full API reference | [API Reference Index](/PhenoMe/advanced/) · [Pipeline](/PhenoMe/advanced/api/pipeline/) |
| Property presets and custom functions | [Custom Properties](/PhenoMe/guides/custom-properties/) · [Property Reference](/PhenoMe/guides/property-reference/) |
| Reproducibility and optimization | [Best Practices](/PhenoMe/guides/best-practices/) |
