---
title: "Quick start card"
---

One-page reference: install, minimal script, and where to read more.

---

## Install (shortest reliable path)

```bash
git clone https://github.com/AAitorG/PhenoMe.git
cd PhenoMe
conda env create -f envs/environment-cpu.yml   # or envs/environment-gpu.yml on NVIDIA Linux/Windows
conda activate phenome-cpu   # or phenome-gpu
```

**Pip-only / editable install:** see [Getting started — Installation options](getting-started.mdx#installation-options).

**Verify:**

```python
from phenome import PhenoMe
print(f"PhenoMe ({PhenoMe.__version__}) is ready!")
```

---

## Minimal analysis (4 steps)

```python
import torch
from phenome import PhenoMe, load_dinov2_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = load_dinov2_model(device=device)
pheno = PhenoMe(device=device, seed=42)

pheno.find_files("path/to/images")
pheno.process_images(wrapper)
pheno.compute_properties(property_preset="basic")  # needs masks for many features; else try "intensity"
pheno.plot_pca(color_by="condition")  # replace with a metadata column you have
```

| Step | Method | Notes |
|------|--------|--------|
| Discover images | `find_files(path, metadata_fn=...)` | Optional metadata: [Data setup](guides/data-setup.md) |
| Embeddings | `process_images(wrapper, ...)` | GPU recommended |
| Classical features | `compute_properties(property_preset=...)` | See [Property reference](guides/property-reference.md) |
| Explore | `plot_pca`, `plot_tsne`, `plot_umap`, `create_interactive_explorer` | [Visualization API](reference/api/visualization.md) |

---

## Common parameters

| Parameter | Typical values | Where |
|-----------|----------------|--------|
| `device` | `cuda:0` or `cpu` | `PhenoMe(...)`, `load_dinov2_model(...)` |
| `seed` | `42` | `PhenoMe(seed=42)` for reproducible DR |
| `property_preset` | `intensity`, `basic`, `full` | `compute_properties` |
| `batch_size` | `16`–`64` (GPU) | `process_images` if OOM, lower it |

---

## When you are stuck

| Symptom | Doc |
|---------|-----|
| Install / CUDA | [FAQ — Installation](faq.md#installation-and-setup) |
| Metadata / CSV | [Data setup](guides/data-setup.md), [FAQ — Data](faq.md#data-and-metadata) |
| Slow / OOM | [Best practices](guides/best-practices.md), [FAQ — Analysis](faq.md#analysis-and-compute) |
| Terminology | [Glossary](glossary.md) |

---

## Next steps

1. [Notebook 01 — Interactive quickstart](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/01_interactive_quickstart.ipynb)
2. [Notebook 03 — Core workflow](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/03_core_phenotyping_workflow.ipynb)
3. [API index](reference/) or [generated API](reference/api/pipeline.md) (see [Developer guide](guides/developer-guide.md#build-the-docs-site))
