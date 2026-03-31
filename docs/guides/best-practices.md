# Best Practices

How to get reproducible results, avoid common errors, and optimize performance. Useful for any analysis.

[← Documentation index](../index.md)

---

## Quick Reference

| Topic | Key tip |
|-------|---------|
| Reproducibility | `PhenoMe(seed=42)`; use `export_experiment_config()` after `save_results()` |
| DR backends | `use_gpu_for_dr=False` for strict reproducibility (sklearn/umap-learn) |
| DR GPU OOM | Install `pykeops` for memory-efficient t-SNE/UMAP; PCA uses batched ExactIncrementalPCA |
| GPU OOM | Reduce `batch_size`, use smaller model, or process in chunks |
| Portability | HDF5 stores relative paths; call `find_files` first, then `load_results`—paths are resolved automatically |
| Data inspection | Run `pheno.inspect_data()` before processing (after `find_files`) |

---

## Table of Contents

1. [Reproducibility](#reproducibility)
2. [Dimensionality Reduction Backends](#dimensionality-reduction-backends)
3. [GPU Optimization](#gpu-optimization)
4. [Memory Management](#memory-management)
5. [Data Organization](#data-organization)
6. [Error Handling](#error-handling)
7. [Performance Tips](#performance-tips)
8. [Common Issues](#common-issues)
9. [Collaboration](#collaboration)

---

## Reproducibility

### Set Random Seeds

For reproducibility, pass a seed to the framework:

```python
pheno = PhenoMe(seed=42)
```

When a seed is provided, it is used for **all** random operations:
- Determinism (PyTorch, NumPy, Python `random`)
- Clustering (GMM)
- Dimensionality reduction (t-SNE, UMAP)
- Mutual information computation
- Random sampling in reports

When `seed=None` (default), no seeds are set anywhere; runs are non-deterministic.

For manual control (e.g. before data loading), you can also call:

```python
from phenome.utils import set_determinism
set_determinism(42)
```

This ensures:
- Same PCA/t-SNE/UMAP projections (within the same backend)
- Reproducible clustering results
- Consistent train/test splits

### Save Your Configuration

**The checkpoint (HDF5) already stores configuration and results.** When you call `save_results()` or use `checkpoint_path` during `process_images()` / `compute_properties()`, the file contains:

- **Results**: embeddings, img_path, metadata, properties
- **Processing params**: channel_mode, resize_size, pad_size, force_rgb (and batch_size when set)

So the HDF5 file is self-contained for processing parameters and data. For full reproducibility, also document parameters that are *not* stored in the checkpoint:

- **seed** (set at framework init)
- **use_gpu_for_dr** (affects which DR backend is used; TorchDR vs sklearn/umap-learn)
- **reference_filters** (used for distance analysis)
- **model name** (e.g. dinov2_vitb14_reg)

```python
# After processing
pheno.save_results(output_dir="results/")  # Checkpoint stores processing_params + results

# Save the extra params not in the checkpoint
pheno.export_experiment_config(
    'results/config.json',
    reference_filters={'condition': 'Control'},
    model_name='dinov2_vitb14_reg',
)
```

When loading: call `find_files` first, then `pheno.load_results("results/phenome_results.h5")`. The framework restores configuration and resolves paths automatically.

### Version Control

Track your analysis code:
- Use git for scripts
- Record library versions
- Save environment specification

```bash
pip freeze > requirements.txt
```

### Save Intermediate Results

The HDF5 file from `save_results()` or checkpointing stores embeddings, metadata, properties, and processing_params in one place. No need to save these separately.

```python
# After processing - single file has everything
pheno.save_results(output_dir="results/")

# Optional: export properties as CSV for external analysis
df = pheno.compute_properties(property_preset="basic")
df.to_csv("results/properties.csv", index=False)

# Distances are not stored in the checkpoint - save separately if needed
dist_results = pheno.compute_reference_distances(reference_filters={'condition': 'Control'})
import pickle
with open('results/distances.pkl', 'wb') as f:
    pickle.dump(dist_results, f)
```

---

## Dimensionality Reduction Backends

> **Warning:** Results from dimensionality reduction (PCA, t-SNE, UMAP) may differ slightly between the **TorchDR + GPU** backend and the **sklearn/umap-learn + CPU** backend. For strict reproducibility across machines or over time, use a single backend consistently.

### Why Results Can Differ

| Reason | Explanation |
|--------|-------------|
| **Different implementations** | TorchDR and sklearn/umap-learn use different algorithms and optimizations. For example, UMAP in TorchDR may use KeOps, FAISS, or raw PyTorch for k-NN, while umap-learn uses a different implementation. |
| **Numerical precision** | GPU computations can differ from CPU due to floating-point accumulation order, parallel reductions, and hardware-specific rounding. Small differences can propagate in iterative methods (t-SNE, UMAP). |
| **Random initialization** | Even with the same seed, different libraries may interpret or apply the seed differently during optimization. |
| **Approximations** | GPU backends often use approximate methods (e.g., FAISS) for speed; CPU backends may use exact computations. |

### Recommendations

- **For reproducibility**: Use `use_gpu_for_dr=False` to always use sklearn/umap-learn, or document which backend and seed you used.
- **For speed**: Use TorchDR + GPU when available; the qualitative structure (clusters, trends) is typically preserved even if coordinates differ slightly.
- **For large datasets**: Install `pykeops` (`pip install pykeops`) to avoid GPU OOM with t-SNE/UMAP; PCA uses batched ExactIncrementalPCA by default.

### n_components Support

| Method | n_components | Notes |
|--------|--------------|-------|
| **PCA** | Any | No restrictions. |
| **UMAP** | Any | No restrictions. |
| **t-SNE** | Any | With sklearn (CPU): when `n_components >= 4`, uses `method='exact'` (slower, O(n²)); otherwise Barnes-Hut (faster). With TorchDR (GPU): supports any `n_components` natively. |

---

## GPU Optimization

### Choose Appropriate Model Size

| Model | GPU Memory | Use Case |
|-------|------------|----------|
| ViT-S/14 | ~2 GB | Prototyping, limited GPU |
| ViT-B/14 | ~4 GB | Good balance |
| ViT-L/14 | ~8 GB | Better features |
| ViT-g/14 | ~16 GB | Best quality |

### Adjust Batch Size

```python
# Start with smaller batch size and increase
# Monitor GPU memory with nvidia-smi

# For 8GB GPU with ViT-B/14
pheno.process_images(wrapper, batch_size=16)

# For 24GB GPU with ViT-g/14
pheno.process_images(wrapper, batch_size=64)
```

### Check GPU Availability

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Use appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Multi-GPU Processing

```python
# For multiple GPUs, specify device index
device = torch.device("cuda:0")  # First GPU
device = torch.device("cuda:1")  # Second GPU

# Note: Pipeline currently uses single GPU
# For multi-GPU, process batches on different GPUs manually
```

---

## Memory Management

### Clear GPU Memory

```python
import torch
import gc

def clear_gpu_memory():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

# Call after processing large batches
clear_gpu_memory()
```

### Process in Chunks

For very large datasets:

```python
import pandas as pd

# Discover all files, then split into chunks
file_df = pheno.find_files("path/to/images")
chunk_size = 1000
chunks = [file_df[i:i+chunk_size] for i in range(0, len(file_df), chunk_size)]

# Process each chunk (set_file_df before each process_images)
for i, chunk_df in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}")
    pheno.set_file_df(chunk_df)
    pheno.process_images(
        wrapper,
        batch_size=32,
        append=(i > 0),
    )
    clear_gpu_memory()
```

### Monitor Memory Usage

```python
import psutil
import torch

def print_memory_status():
    """Print current memory usage."""
    # CPU memory
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1e9
    print(f"CPU memory: {cpu_mem:.2f} GB")

    # GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU memory: {gpu_mem:.2f} GB (max: {gpu_max:.2f} GB)")

# Call periodically during processing
print_memory_status()
```

---

## Data Organization

### Recommended Directory Structure

```
project/
├── data/
│   ├── images/           # Raw images
│   ├── masks/            # Segmentation masks
│   └── metadata.csv      # Experimental metadata
├── results/
│   ├── phenome_results.h5
│   ├── distances.pkl
│   ├── properties.csv
│   └── config.json
├── figures/
│   ├── pca_by_condition.html
│   └── distance_distribution.html
└── scripts/
    ├── process_data.py
    └── analyze_results.py
```

### Naming Conventions

Use consistent, informative names:

```python
# Good: descriptive, consistent
images/
├── Control_30min_rep1_001.tif
├── Control_30min_rep1_002.tif
├── DrugA_10uM_30min_rep1_001.tif

# Bad: inconsistent, unclear
images/
├── img1.tif
├── control.tif
├── experiment_final_v2_new.tif
```

### Inspect Before Processing

```python
# Always inspect your data first (returns: file_path, height, width, channels, min, max, error, shape_match, ...)
inspect_df = pheno.inspect_data()

print("Image statistics:")
print(inspect_df)

# Check for multiple image shapes (height, width, channels are inspection columns)
if len(inspect_df) > 0 and inspect_df.groupby(['height', 'width', 'channels']).ngroups > 1:
    print("\nWarning: Multiple image shapes detected!")
```

---

## Error Handling

### Wrap Processing in Try-Except

```python
try:
    pheno.process_images(wrapper, batch_size=32)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU OOM - reducing batch size")
        clear_gpu_memory()
        pheno.process_images(wrapper, batch_size=16)
    else:
        raise
```

### Validate Inputs

```python
def validate_file_df(file_df):
    """Validate file DataFrame before processing."""
    assert 'file_path' in file_df.columns, "Missing 'file_path' column"

    # Check files exist
    missing = []
    for path in file_df['file_path']:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print(f"Warning: {len(missing)} files not found")
        print(f"Examples: {missing[:3]}")

    return len(missing) == 0

# Use before processing
file_df = pheno.find_files("path/to/images")
if validate_file_df(file_df):
    pheno.process_images(wrapper)
```

### Log Processing Steps

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Starting processing")
logger.info(f"Processing {len(file_df)} images")

try:
    pheno.process_images(wrapper)
    logger.info("Processing complete")
except Exception as e:
    logger.error(f"Processing failed: {e}")
    raise
```

---

## Performance Tips

### Use Appropriate num_workers

```python
import os

# Usually 2-4 workers per GPU is good
num_cpus = os.cpu_count()
num_workers = min(4, num_cpus - 1)  # Leave one CPU free

pheno.process_images(wrapper, num_workers=num_workers)
```

### Optimize Image Loading

```python
# Use appropriate image format
# - TIFF for microscopy (tifffile is optimized)
# - PNG/JPEG for natural images

# Pre-resize large images if possible
# Processing 4000x4000 images is much slower than 1000x1000
```

### Filter Early

```python
# Filter during processing, not after
pheno.process_images(
    wrapper,
    filters={'condition': ['Control', 'Drug1']},  # Process only what you need
    batch_size=32
)

# Instead of:
pheno.process_images(wrapper)  # Process everything
# Then filter results later
```

### Save and Reload

```python
# For iterative analysis, save after processing
pheno.process_images(wrapper)
pheno.save_results(output_dir="results")

# In new session, reload instead of reprocessing
pheno.find_files("path/to/images")
pheno.load_results("results/phenome_results.h5")
```

---

## Common Issues

### Issue: GPU Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size
2. Use smaller model
3. Clear GPU memory between batches
4. Process images in chunks

```python
# Solution 1: Smaller batch
pheno.process_images(wrapper, batch_size=8)

# Solution 2: Clear memory
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Issue: Slow Processing

**Symptoms:** Processing takes much longer than expected

**Solutions:**
1. Increase batch size (if GPU memory allows)
2. Increase num_workers
3. Use faster storage (SSD vs HDD)
4. Pre-filter to process only needed images

```python
# Check if using GPU
print(f"Device: {wrapper.device}")  # Should show cuda:X

# Optimize workers
pheno.process_images(wrapper, batch_size=64, num_workers=8)
```

### Issue: Inconsistent Results

**Symptoms:** Different runs give different results

**Solutions:**
1. Set random seed at the start (`PhenoMe(seed=42)` or `set_determinism(42)`)
2. Ensure deterministic mode is enabled
3. Check for non-deterministic operations

```python
# Pass seed to framework for reproducibility
pheno = PhenoMe(seed=42)

# Or call set_determinism before any processing
from phenome.utils import set_determinism
set_determinism(42)

# Verify determinism settings
print(f"cudnn.deterministic: {torch.backends.cudnn.deterministic}")
```

### Issue: Missing Properties

**Symptoms:** Some images have NaN properties (the framework warns automatically when NaNs are present)

**Solutions:**
1. Check if masks exist for all images
2. Verify requirement types match available data
3. Add error handling in property functions

```python
# Inspect the properties DataFrame for NaNs
df = pheno.compute_properties(property_preset="basic")
nan_counts = df.select_dtypes(include=['number']).isna().sum()
print("Properties with NaNs:", nan_counts[nan_counts > 0].to_dict())

# Investigate specific images with missing area
area_nan_idx = df[df['area'].isna()].index[:5].tolist()
for idx in area_nan_idx:
    info = pheno.get_image_info(idx)
    print(f"Image {idx}: {info['img_path']}")
```

### Issue: Filter Returns Empty Results

**Symptoms:** No images after filtering

**Solutions:**
1. Check available metadata keys
2. Verify filter values match data
3. Check for case sensitivity

```python
# Debug filter issues
print("Available keys:", pheno.get_available_metadata_keys())

# Check actual values
df = pheno.export_dataset_table()
print("Unique conditions:", df['condition'].unique())

# Use correct values
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'}  # Exact match required
)
```

---

## Collaboration

### Sharing Experiments

Share your analysis by saving results and generating a report:

```python
# Save results (HDF5) and generate standalone HTML report
pheno.save_results(output_dir="experiment_2026_01")
pheno.generate_report(
    output_path="experiment_2026_01/report.html",
    reference_filters={"condition": "Control"},
)

# Save config for reproducibility (seed, reference_filters not in HDF5)
pheno.export_experiment_config(
    "experiment_2026_01/config.json",
    reference_filters={"condition": "Control"},
)
```

Colleagues can load results with:

```python
pheno = PhenoMe()
pheno.find_files("/path/to/experiment_2026_01/images")
pheno.load_results("experiment_2026_01/phenome_results.h5")
```

**Directory structure:** See [Recommended Directory Structure](#recommended-directory-structure) in Data Organization. For collaboration, add `configs/` (template configs) and `environment.yml` (Conda env) at the project root.

### Optional: DVC for Data Versioning

**When to use DVC:** Large raw image datasets, checkpoint/HDF5 result files, and reproducibility of data lineage across machines.

**What the checkpoint already provides:** The HDF5 file from `save_results()` or `checkpoint_path` stores:
- `processing_params` (channel_mode, resize_size, pad_size, force_rgb)
- `version` (format version)
- embeddings, metadata, properties

You do **not** need to duplicate this configuration elsewhere. The checkpoint is self-contained for processing state.

**How DVC complements the checkpoint:** DVC handles *data and assets* (versioning, storage, sharing). The checkpoint handles *processing state* (what was computed, with which params). They work together:

```bash
# Version large image datasets
dvc add data/

# Version checkpoint/result files (large HDF5 outputs)
dvc add results/phenome_results.h5

# Commit DVC metadata to git
git add data.dvc results/phenome_results.h5.dvc
git commit -m "Add data and results"
```

**Summary:** DVC does not replace the checkpoint. Use the checkpoint for crash-safe processing and parameter traceability; use DVC for data versioning and large-file storage.

### Naming Conventions

- Use consistent metadata keys (e.g. `condition`, `replicate`, `timepoint`)
- Name config files by experiment: `drug_screen_2026.json`

---

## See Also

| Topic | Document |
|-------|----------|
| Initial setup | [Getting Started](../getting-started.md) |
| Embeddings, channels, architecture | [Core Concepts](../concepts.md) |
| End-to-end examples | [Common Workflows](../workflows.md) |
| Metadata extraction | [Experiment Details](experiment-details.md) |
| Property functions | [Custom Properties](custom-properties.md) |
| Troubleshooting | [FAQ](../faq.md) |
| Key terms | [Glossary](../glossary.md) |
| Checkpoint structure | [HDF5 Protocol](../reference/DATABASE_PROTOCOL.md) |
