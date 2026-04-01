# PhenoMe Documentation

<p align="center">
  <img src="Logo.png" alt="PhenoMe logo" width="480"/>
</p>

**PhenoMe** (package: `phenome`) is a modular framework for phenotyping analysis using deep learning embeddings. It is [model-agnostic and dataset-agnostic](concepts.md#what-is-phenotyping)—you provide your data and vision model of choice.

**Author:** [Aitor González-Marfil](https://github.com/AAitorG) (@AAitorG)

---

## Quick Start for New Users

If you are new to PhenoMe, we recommend following this path:

1.  **[Installation & Setup](getting-started.md)**: Get everything ready on your machine.
2.  **[Interactive Quickstart](../Notebooks/tutorials/01_interactive_quickstart.ipynb)**: Try it out with widgets (no coding required).
3.  **[Core Concepts](concepts.md)**: Understand how "visual fingerprints" work.
4.  **[Standard Workflow](../Notebooks/tutorials/03_core_phenotyping_workflow.ipynb)**: Run your first programmatic analysis.

---

## 📖 User Guides (Simple & Practical)

These guides focus on getting tasks done without needing deep technical knowledge.

| I want to... | Recommended Guide |
|--------------|-------------------|
| **Set up my experiment details** | [Experiment Details (Metadata)](guides/experiment-details.md) |
| **Find where a specific function lives** | [Function Location Guide](function-location-guide.md) |
| **Learn about physical properties** | [Property Reference](guides/property-reference.md) |
| **Understand the analysis reports** | [How to Read Reports](guides/report-guide.md) |
| **See common examples** | [Common Workflows](workflows.md) |
| **Find answers to common questions** | [FAQ](faq.md) |

---

## 🛠️ Advanced & Developer Reference

For users who want to customize the framework, write custom code, or understand the internals.

### Customization & Extension
- **[Custom Properties](guides/custom-properties.md)**: Extract specific features (area, intensity, etc.).
- **[Vision Model Wrappers](reference/api/model-wrapper.md)**: Use your own pre-trained models.
- **[Extending PhenoMe](guides/extending.md)**: Create plugins and custom presets.
- **[Best Practices](guides/best-practices.md)**: GPU optimization and performance tuning.

### API & Technical Reference
- **[API Index](reference/index.md)**: Complete list of classes and methods.
- **[Pipeline API](reference/api/pipeline.md)**: Details on the main `PhenoMe` class.
- **[Visualization API](reference/api/visualization.md)**: Plotting and interactive exploration.
- **[HDF5 Protocol](reference/DATABASE_PROTOCOL.md)**: How data is stored on disk.

---

## 🎓 Step-by-Step Tutorials (Notebooks)

We recommend following these in order:

1. **[01_interactive_quickstart.ipynb](../Notebooks/tutorials/01_interactive_quickstart.ipynb)** - Best for beginners.
2. **[02_inspect_your_images.ipynb](../Notebooks/tutorials/02_inspect_your_images.ipynb)** - Inspecting your images for quality.
3. **[03_core_phenotyping_workflow.ipynb](../Notebooks/tutorials/03_core_phenotyping_workflow.ipynb)** - The standard analysis workflow.
4. **[04_advanced_metadata_handling.ipynb](../Notebooks/tutorials/04_advanced_metadata_handling.ipynb)** - Handling complex experiment details.
5. **[05_exploratory_analysis_and_explainability.ipynb](../Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb)** - Clustering and AI insights.
6. **[06_extending_phenome_plugins.ipynb](../Notebooks/tutorials/06_extending_phenome_plugins.ipynb)** - Custom plugins and model wrappers.

---

## 🌟 Key Features

- **No training required**: Use "visual fingerprints" from pre-trained models immediately.
- **Easy Visualization**: Quickly see patterns with PCA, t-SNE, and UMAP.
- **Automated Reporting**: One command to generate a full analysis summary.
- **Reproducible**: Every run can be saved, resumed, and shared.

---

[Quick Reference](quick-reference.md) · [Function Lookup](function-location-guide.md) · [FAQ](faq.md) · [Glossary](glossary.md) · [API Reference](reference/index.md)
