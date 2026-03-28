# PhenoMe Documentation

<p align="center">
  <img src="Logo.svg" alt="PhenoMe logo" width="240"/>
</p>

**PhenoMe** (package: `phenome`) is a modular pipeline for phenotyping analysis using deep learning embeddings. It is [model-agnostic and dataset-agnostic](concepts.md#what-is-phenotyping)—you provide your data and vision model of choice.

**Author:** [Aitor González-Marfil](https://github.com/AAitorG) (@AAitorG)

---

## Quick Links

**Getting started?** Start with [Getting Started](getting-started.md) → [Beginner Tutorial](../Notebooks/tutorials/01_beginner_interactive.ipynb)

**Looking for something specific?**

| I want to... | Go to |
|--------------|-------|
| Install and run my first analysis | [Getting Started](getting-started.md) → [Quick Example](#quick-example) |
| New to phenotyping / no ML background | [Getting Started](getting-started.md#quick-start) → [FAQ](faq.md) |
| Extract metadata from paths/CSV | [Custom Metadata](guides/custom-metadata.md) |
| Compute custom image/mask properties | [Custom Properties](guides/custom-properties.md), [Property Reference](guides/property-reference.md) |
| Use a different vision model | [Vision Model Wrappers](reference/api/model-wrapper.md) |
| Build visualizations (PCA, t-SNE, UMAP, distances) | [Visualization](reference/api/visualization.md), [Distances](reference/api/distances.md) |
| Generate HTML report | [Report Generation](reference/api/report.md) · [How to read it](guides/report-guide.md) |
| Explore embeddings interactively in Jupyter | [Interactive Explorer](reference/api/interactive.md) |
| See end-to-end workflows | [Common Workflows](examples/workflows.md) |
| Understand embeddings, channels, storage | [Core Concepts](concepts.md) |
| Troubleshoot GPU, reproducibility, speed | [Best Practices](guides/best-practices.md) |
| Extend the pipeline (plugins, presets) | [Extending](guides/extending.md) |

---

## Key Features

- **Zero-shot analysis**: Works with pre-trained models (DINOv2, CLIP, custom) with no training
- **Deep learning embeddings**: Extract global image representations for phenotypic comparison
- **Distance metrics**: Compute distances to reference groups (centroid or all-to-all)
- **Visualization**: PCA, t-SNE, UMAP, centroids, trajectories, distance distributions
- **Flexible properties**: Built-in presets or custom image/mask features
- **Reproducibility**: Checkpoint/resume for long runs, seed control, HDF5 persistence
- **Interactive explorer**: Jupyter-based dashboard for embedding exploration

---

## Documentation

### Core Documentation

| Document | What it covers |
|----------|----------------|
| [Getting Started](getting-started.md) | Installation, dependencies, quick-start code |
| [Core Concepts](concepts.md) | Embeddings, channel modes, pipeline architecture |
| [Quick Reference](quick-reference.md) | Workflow overview, common options, decision trees |
| [Common Workflows](examples/workflows.md) | End-to-end analysis examples |

### Guides

| Document | What it covers |
|----------|----------------|
| [Custom Metadata](guides/custom-metadata.md) | Path templates, CSV lookup, metadata extraction |
| [Custom Properties](guides/custom-properties.md) | Writing custom feature extractors |
| [Property Reference](guides/property-reference.md) | Built-in property presets and their definitions |
| [Best Practices](guides/best-practices.md) | GPU optimization, reproducibility, debugging |
| [Interpretability](guides/interpretability.md) | Correlating embeddings to morphological features |
| [Report Guide](guides/report-guide.md) | Understanding and interpreting HTML reports |
| [Extending](guides/extending.md) | Plugins, custom presets, extension points |
| [FAQ](faq.md) | Common questions and troubleshooting |
| [Glossary](glossary.md) | Key terminology |

### Reference

| Document | What it covers |
|----------|----------------|
| [API Reference](reference/index.md) | Full method signatures and parameters |
| [HDF5 Protocol](reference/DATABASE_PROTOCOL.md) | Checkpoint structure and file format |
| [Function Lookup](function-location-guide.md) | Find functions by name |

---

## Interactive Notebooks

| Resource | Type | What it covers |
|----------|------|----------------|
| [01_beginner_interactive.ipynb](../Notebooks/tutorials/01_beginner_interactive.ipynb) | Tutorial | Interactive widgets, no coding required |
| [02_advanced_tutorial.ipynb](../Notebooks/tutorials/02_advanced_tutorial.ipynb) | Tutorial | Custom properties, advanced analysis |
| [03_data_auditing_tutorial.ipynb](../Notebooks/tutorials/03_data_auditing_tutorial.ipynb) | Tutorial | Data auditing and quality checks |
| [example_simple_phenotyping.py](../Notebooks/examples/example_simple_phenotyping.py) | Script | Minimal workflow for quick start |
| [example_advanced_phenotyping.py](../Notebooks/examples/example_advanced_phenotyping.py) | Script | Full customization example |

---

## Quick Example

```python
from phenome import PhenoMe, load_dinov2_model

# 1. Setup
model, wrapper = load_dinov2_model()
pheno = PhenoMe(seed=42)  # seed for reproducibility

# 2. Load data
file_df = pheno.find_files("path/to/images")

# 3. Process
pheno.process_images(wrapper)

# 4. Analyze
pheno.compute_properties(property_preset='basic')
distance_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    source='embeddings'
)

# 5. Visualize
pheno.plot_pca(color_by='condition')
pheno.plot_distance_distribution(distance_results, group_by='condition')

# 6. Report
pheno.generate_report(
    output_path="report.html",
    reference_filters={'condition': 'Control'},
    include_clustering=True,
    include_image_gallery=True
)
```

---

## Quick Links

[Quick Reference](quick-reference.md) · [Function Lookup](function-location-guide.md) · [FAQ](faq.md) · [Glossary](glossary.md) · [API Reference](reference/index.md) · [Best Practices](guides/best-practices.md)

---

See [LICENSE](../LICENSE) for license information.
