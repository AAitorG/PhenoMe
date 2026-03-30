# Getting Started

**Author:** [Aitor González-Marfil](https://github.com/AAitorG) (@AAitorG)

This guide covers installation, dependencies, and your first steps.

[← Back to docs](index.md)

PhenoMe is [model-agnostic and dataset-agnostic](concepts.md#what-is-phenotyping). Examples use DINOv2 by default, but any vision model works. No training required—zero-shot analysis out-of-the-box.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Basic Workflow](#basic-workflow)
5. [Expected Folder Structure](#expected-folder-structure)
6. [Image Preprocessing](#image-preprocessing)
7. [Next Steps](#next-steps)

---

## Requirements

### Python Version
- Python 3.11+

### Hardware
- **CPU**: Works on CPU, but GPU is strongly recommended
- **GPU**: NVIDIA GPU with CUDA support

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe
```

### 2. Install Dependencies

Install PyTorch first from [pytorch.org](https://pytorch.org), then:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision numpy pandas matplotlib plotly scikit-learn joblib tqdm opencv-python tifffile h5py scipy "scikit-image>=0.26" ipywidgets pillow
```

For optional features (UMAP on CPU, distance correlation, GPU-accelerated DR with TorchDR / KeOps, NIfTI image support):
```bash
pip install -r requirements-optional.txt
```

**Jupyter notebooks:** install a notebook frontend if you do not already have one, for example `pip install jupyterlab` or `pip install notebook`. Tutorial notebooks that download example data may use `requests` (`pip install requests`).

#### Dependency Reference

| Package | Purpose |
|---------|---------|
| `torch` (≥2.0) | Deep learning framework |
| `torchvision` | Image transforms |
| `numpy`, `pandas` | Numerical and tabular data |
| `matplotlib` | Static figures (some helpers) |
| `plotly` | Interactive plots, HTML reports, `FigureWidget` in the explorer |
| `scikit-learn` | PCA, t-SNE, clustering |
| `joblib` | Parallel property computation |
| `tqdm` | Progress bars |
| `opencv-python` | Image I/O (non-TIFF) |
| `tifffile` | TIFF image I/O |
| `nibabel` | NIfTI image I/O (Optional) |
| `h5py` | HDF5 checkpoint storage |
| `scipy` | Distance transform, properties |
| `scikit-image` (≥0.26) | Property presets |
| `ipywidgets` | Interactive explorer and widgets (core; pulls in `ipython`) |
| `anywidget` | Core widget protocol (required for `FigureWidget` in `plotly` 6+) |
| `pillow` | Image encoding for HTML reports (PIL) |
| `umap-learn` | UMAP when not using TorchDR on GPU (opt.) |
| `dcor` | Distance correlation (opt.) |
| `torchdr` | GPU-accelerated DR (opt.) |
| `pykeops` | Memory-efficient GPU backend (opt.) |

### 3. Verify Installation

```python
from phenome import PhenoMe, load_dinov2_model
print("Installation successful!")
```

---

## Quick Start

Minimal code to get started:

```python
from phenome import PhenoMe, load_dinov2_model

model, wrapper = load_dinov2_model()
pheno = PhenoMe(seed=42)

file_df = pheno.find_files("path/to/your/images")
pheno.process_images(wrapper)
pheno.plot_pca()
```

For folders organized by condition (e.g., `Control/`, `Drug1/`), see [Expected Folder Structure](#expected-folder-structure) and use `color_by='condition'` in `plot_pca()`.

> **Note**: See [Vision Model Wrappers](reference/api/model-wrapper.md) for other models.

---

## Basic Workflow

The typical workflow has four stages:

### 1. Setup

Initialize the pipeline with a vision model:

```python
from phenome import PhenoMe, load_dinov2_model

# DINOv2 models (smallest to largest):
# - dinov2_vits14_reg: 22M params, fastest
# - dinov2_vitb14_reg: 86M params, balanced (recommended)
# - dinov2_vitl14_reg: 300M params, better features
# - dinov2_vitg14_reg: 1.1B params, best features

model, wrapper = load_dinov2_model(model_name="dinov2_vitb14_reg")
pheno = PhenoMe(seed=42)
```

### 2. Load Data

Find images and extract metadata:

```python
# Flat folder: all images in one directory
file_df = pheno.find_files("path/to/images")

# Optional: filter by extension
file_df = pheno.find_files(
    "path/to/images",
    extensions=['.tif', '.tiff', '.png']
)
```

For condition-based folders, see [Expected Folder Structure](#expected-folder-structure).

### 3. Process Images

Extract embeddings:

```python
# Minimal
pheno.process_images(wrapper)

# With options
pheno.process_images(
    wrapper,
    batch_size=32,                          # Adjust based on GPU memory
    resize_size=224,                        # Target size for model
    channel_mode='split',                   # 'split' or 'combined'
    filters={'condition': ['Control', 'Drug1']}  # Optional filtering
)
```

### 4. Analyze and Visualize

```python
# Optional: compute properties
pheno.compute_properties(property_preset='basic')

# Compute distances to reference
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    source='embeddings',
    mode='centroid'
)

# Visualize
pheno.plot_pca(color_by='condition')
pheno.plot_distance_distribution(dist_results, group_by='condition')

# Interactive explorer in Jupyter
# pheno.create_interactive_explorer()

# Save results
pheno.save_results(output_dir="results")
```

---

## Expected Folder Structure

The pipeline is flexible—use flat or hierarchical structures. Masks are **optional** for embeddings; only required for shape-based properties.

### Minimal: Flat folder

```
data/
├── image_001.tif
├── image_002.tif
└── image_003.tif
```

```python
file_df = pheno.find_files("data")
```

### By condition (recommended)

```
data/
├── Control/
│   ├── img_001.tif
│   └── img_002.tif
├── Drug1/
│   ├── img_001.tif
│   └── img_002.tif
└── Drug2/
    └── img_001.tif
```

```python
from phenome import get_metadata_from_path

metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
file_df = pheno.find_files("data", metadata_fn=metadata_fn)
```

### With masks

```
data/
├── images/
│   ├── Control/
│   │   └── cell_001.tif
│   └── Drug1/
│       └── cell_001.tif
└── masks/
    ├── Control/
    │   └── cell_001.tif
    └── Drug1/
        └── cell_001.tif
```

See [Custom Metadata](guides/custom-metadata.md) for path templates and mask discovery.

---

## Image Preprocessing

Before processing, audit your images:

```python
audit_df = pheno.audit_data()  # Analyze dimensions and intensity ranges
```

Key parameters:
- **`resize_size`**: Target size for model input (default: 224)
- **`pad_size`**: Pad small images before resizing (optional)
- **`channel_mode`**: `'split'` (each channel separately) or `'combined'` (as RGB)

---

## Next Steps

1. **[Beginner Tutorial](../Notebooks/tutorials/01_beginner_interactive.ipynb)** - Interactive widgets, no coding required
2. **Example scripts**:
   - [example_simple_phenotyping.py](../Notebooks/examples/example_simple_phenotyping.py) - Minimal workflow
   - [example_advanced_phenotyping.py](../Notebooks/examples/example_advanced_phenotyping.py) - Full customization
3. **[Core Concepts](concepts.md)** - Embeddings, channel modes, architecture
4. **[Custom Metadata](guides/custom-metadata.md)** - Metadata extraction
5. **[Custom Properties](guides/custom-properties.md)** - Domain-specific features
6. **[Common Workflows](examples/workflows.md)** - End-to-end examples
7. **[FAQ](faq.md)** - Common questions
