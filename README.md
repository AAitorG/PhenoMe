# PhenoMe

<p align="center">
  <img src="docs/Logo.svg" alt="PhenoMe logo" width="280"/>
</p>

A modular, **dataset-agnostic** and **model-agnostic** pipeline for phenotyping analysis using deep learning embeddings and extracted image properties.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Quick Start

```python
from phenome import PhenoMe, load_dinov2_model

# 1. Setup
model, wrapper = load_dinov2_model()
pheno = PhenoMe(seed=42)

# 2. Process Data
file_df = pheno.find_files("path/to/images")
pheno.process_images(wrapper)

# 3. Compute Properties and save results
pheno.compute_properties(property_preset="basic")

# 4. Visualize
pheno.create_interactive_explorer()
```

---

## Installation

```bash
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe

# Conda (Recommended)
conda env create -f environment.yml
conda activate phenome
```

**pip** (Python 3.11+): from the repo root, `pip install -e .` or `pip install -r requirements.txt` then `pip install -e .`. Use a recent `pip` and `setuptools` so editable installs work (`pip install --upgrade pip setuptools wheel` if `pip install -e .` fails).

**Core install** includes `ipywidgets` and `anywidget` (required for interactive explorer). For local notebooks, also install Jupyter, for example `pip install jupyterlab` or `conda install jupyter`.

**Optional dependencies** (see `requirements-optional.txt`):
- GPU-accelerated DR: `torchdr` for fast PCA/t-SNE/UMAP on CUDA.
- Memory-efficient GPU DR: `pykeops` (KeOps backend for large t-SNE/UMAP).
- UMAP on CPU (when not using TorchDR on GPU): `umap-learn`.
- Distance correlation: `dcor`.
- NIfTI image support (.nii, .nii.gz): `nibabel`.

---

## Documentation

Full documentation is in the [`docs/`](docs/) directory.

- **[Getting Started](docs/getting-started.md)**
- **[Core Concepts](docs/concepts.md)**
- **API Reference**:
  - [Pipeline](docs/reference/api/pipeline.md)
  - [Properties](docs/reference/api/properties.md)
  - [Distances](docs/reference/api/distances.md)
  - [Visualization](docs/reference/api/visualization.md)
  - [Report](docs/reference/api/report.md)
- **Guides**: [Custom Metadata](docs/guides/custom-metadata.md), [Custom Properties](docs/guides/custom-properties.md), [Best Practices](docs/guides/best-practices.md)
- **Examples**: [Common Workflows](docs/examples/workflows.md)

Interactive Jupyter notebooks and tutorials are available in [`Notebooks/`](Notebooks/).

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
