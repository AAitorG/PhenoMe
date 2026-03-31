# Getting Started

This guide will help you install PhenoMe and run your first analysis.

[← Back to docs](index.md)

---

## 1. Requirements

- **Python**: 3.11 or newer.
- **Hardware**: An NVIDIA GPU is strongly recommended for faster processing, but it will also work on your CPU.

---

## 2. Installation

Open your terminal and run the following commands:

```bash
# 1. Clone the repository
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe

# 2. Install the core package and dependencies
# (We recommend installing PyTorch first from pytorch.org)
pip install -r requirements.txt
```

To verify the installation:
```python
from phenome import PhenoMe
print("PhenoMe is ready!")
```

---

## 3. Quick Start: Your First Analysis

This example shows the basic steps to analyze a folder of images.

### Step 1: Setup
Load a vision model (we use DINOv2 by default) and initialize the framework.

```python
import torch
from phenome import PhenoMe, load_dinov2_model

# Load model and initialize PhenoMe
model, wrapper = load_dinov2_model(model_name="dinov2_vitb14_reg", device=torch.device("cuda"))
pheno = PhenoMe(seed=42)
```

### Step 2: Find and Inspect Images
Tell PhenoMe where your images are and check them for any quality issues.

```python
# Find images in a folder
pheno.find_files("path/to/your/images")

# Inspect images for size and brightness issues
inspect_df = pheno.inspect_data()
```

### Step 3: Extract "Visual Fingerprints"
Extract embeddings (visual fingerprints) and compute basic properties like area and intensity.

```python
# Extract visual fingerprints using the model
pheno.process_images(wrapper)

# Compute basic image properties
pheno.compute_properties(property_preset='basic')
```

### Step 4: Visualize Results
Create a simple plot to see how your different groups (e.g., "Control" vs "Treated") look.

```python
# Visualize results with PCA (colored by condition)
pheno.plot_pca(color_by='condition')
```

---

## 4. Next Steps

Your next stop should be the tutorials, or you can dive into organizing complex experimental setups:

1.  **[Interactive Quickstart](../Notebooks/tutorials/01_interactive_quickstart.ipynb)** - Learn by doing with interactive widgets (No code required).
2.  **[Experiment Details Pipeline (Metadata)](guides/experiment-details.md)** - Learn how to link your images to treatments, concentrations, or external CSVs.
3.  **[Core Concepts](concepts.md)** - Deep dive into how the analysis works (what are embeddings?).
4.  **[Troubleshooting & FAQ](faq.md)** - What to do if something goes wrong.
5.  **[Glossary](glossary.md)** - Confused by a term? Check the glossary.
