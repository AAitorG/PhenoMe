# PhenoMe

<p align="center">
  <img src="docs/Logo.svg" alt="PhenoMe logo" width="280"/>
</p>

A modular, **dataset-agnostic** and **model-agnostic** framework for phenotyping analysis using deep learning embeddings (**visual fingerprints**) and extracted image properties.

---

## Quick Start

Analyze your images in four simple steps:

```python
from phenome import PhenoMe, load_dinov2_model

# 1. Setup: Load a vision model (DINOv2)
# device=torch.device("cuda") set device to GPU if available for faster processing
wrapper = load_dinov2_model()  # pass device as needed
pheno = PhenoMe() # pass device as needed

# 2. Load: Find your images
pheno.find_files("path/to/images")

# 3. Analyze: Extract visual fingerprints and compute properties
pheno.process_images(wrapper)
pheno.compute_properties(property_preset="basic")

# 4. Explore: Launch an interactive dashboard
pheno.create_interactive_explorer()
```

For a no-code experience, try the **[Interactive Quickstart](Notebooks/tutorials/01_interactive_quickstart.ipynb)**.

---

## Installation

```bash
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe

# Conda (Recommended)
conda env create -f environment.yml
conda activate phenome
```

**pip** (Python 3.11+): from the repo root, `pip install -e .` or `pip install -r requirements.txt` then `pip install -e .`.

**Core install** includes `ipywidgets` and `anywidget` (required for interactive explorer). For local notebooks, also install Jupyter, for example `pip install jupyterlab` or `conda install jupyter`.

**Optional dependencies** (see `requirements-optional.txt`):
- GPU-accelerated DR: `torchdr` for fast PCA/t-SNE/UMAP on CUDA.
- Memory-efficient GPU DR: `pykeops` (KeOps backend for large t-SNE/UMAP).
- UMAP on CPU (when not using TorchDR on GPU): `umap-learn`.
- Distance correlation: `dcor`.
- NIfTI image support (.nii, .nii.gz): `nibabel`.

---

## 📖 Documentation

Full documentation is available in the [`docs/`](docs/) directory.

### For New Users (Start Here)
- **[Getting Started](docs/getting-started.md)**: Easy installation and your first 4-step analysis.
- **[Interactive Tutorials](Notebooks/tutorials/)**: Hands-on learning in Jupyter notebooks.
- **[Core Concepts](docs/concepts.md)**: How "visual fingerprints" and the analysis flow work.
- **[Common Workflows](docs/workflows.md)**: Real-world examples (Drug screening, time-course, etc.).

### For Advanced Users & Developers
- **[Experiment Details (Metadata)](docs/guides/experiment-details.md)**: Linking images to your experimental context.
- **[API Reference](docs/reference/api/pipeline.md)**: Full technical documentation for all methods.
- **[Custom Properties](docs/guides/custom-properties.md)**: How to extract your own image features.
- **[Extending the Framework](docs/guides/extending.md)**: Creating plugins and custom model wrappers.

---

## Features

- **Model Agnostic**: Extract embeddings using DINOv2 or any custom PyTorch vision model.
- **Dataset Agnostic**: Works with any image dataset structure with customizable metadata extraction.
- **Analysis Toolkit**: Distance quantification, anomaly detection, property extraction, and correlation analysis.
- **Interactive Visualizations**: High-performance Plotly-based PCA, t-SNE, and UMAP plots.
- **Automated Reporting**: Generate comprehensive standalone HTML reports.

## Author

**Aitor González-Marfil** — [@AAitorG](https://github.com/AAitorG)

## Citation

```bibtex
@software{phenome,
  author = {Aitor González-Marfil},
  title = {PhenoMe: Model-Agnostic Deep Learning-Based Phenotyping Analysis},
  year = {2026},
  url = {https://github.com/AAitorG/PhenoMe}
}
```
