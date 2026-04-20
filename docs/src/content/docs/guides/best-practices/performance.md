---
title: "Performance"
description: GPU sizing, batch size, memory management, and fixes for slow or out-of-memory runs.
sidebar:
  order: 2
---

## GPU sizing

### Adjust batch size

```python
pheno.process_images(wrapper, batch_size=16)
pheno.process_images(wrapper, batch_size=64)
```

Higher is faster but uses more memory. Drop on OOM.

### Check GPU availability

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU memory: {total:.1f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Multi-GPU selection

```python
device = torch.device("cuda:0")
device = torch.device("cuda:1")
```

Pass the same `device` to `load_dinov2_model(...)` and `PhenoMe(...)`.

## Memory management

### Clear GPU memory

```python
import gc
import torch

def clear_gpu_memory() -> None:
    """Free GPU memory between stages."""
    gc.collect()
    torch.cuda.empty_cache()

clear_gpu_memory()
```

### Process in chunks

For very large datasets:

```python
file_df = pheno.find_files("path/to/images")

chunk_size = 1000
chunks = [file_df[i:i + chunk_size] for i in range(0, len(file_df), chunk_size)]

for i, chunk_df in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}")
    pheno.set_file_df(chunk_df)
    pheno.process_images(
        wrapper,
        batch_size=32,
        append=(i > 0),
    )
    clear_gpu_memory()
```

### Monitor memory usage

```python
import psutil
import torch

def print_memory_status() -> None:
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1e9
    print(f"CPU memory: {cpu_mem:.2f} GB")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU memory: {gpu_mem:.2f} GB (max {gpu_max:.2f} GB)")
```

## Throughput tips

### Choose `num_workers`

```python
import os

num_workers = min(4, (os.cpu_count() or 2) - 1)
pheno.process_images(wrapper, num_workers=num_workers)
```

### Filter early

Do not process what you will not use later.

```python
pheno.process_images(
    wrapper,
    filters={"condition": ["Control", "Drug1"]},
    batch_size=32,
)
```

### Checkpoint and reload

```python
pheno.process_images(wrapper, checkpoint_path="results/run.h5")
pheno.save_results(output_dir="results")

pheno.find_files("path/to/images")
pheno.load_results("results/phenome_results.h5")
```

## Troubleshooting

### GPU out of memory

**Symptom**: `RuntimeError: CUDA out of memory`.

**Fixes (in order):**

1. Reduce `batch_size`.
2. Pick a smaller model (for example `vits14`).
3. `gc.collect(); torch.cuda.empty_cache()` between stages.
4. Process in chunks (see above).

```python
pheno.process_images(wrapper, batch_size=8)
```

### Slow processing

**Symptom**: jobs take much longer than expected.

**Fixes:**

1. Increase `batch_size` if GPU has headroom.
2. Increase `num_workers`.
3. Use SSD storage.
4. Pre-filter to the subset you actually need.

```python
print(f"Device: {wrapper.device}")  # should say cuda:X
pheno.process_images(wrapper, batch_size=64, num_workers=8)
```

### DR on GPU runs out of memory

Install `pykeops` so t-SNE / UMAP use memory-efficient GPU kernels:

```bash
pip install pykeops
```

PCA is already batched through `ExactIncrementalPCA`.

### Wrap processing with fallback

```python
try:
    pheno.process_images(wrapper, batch_size=32)
except RuntimeError as e:
    if "out of memory" in str(e):
        clear_gpu_memory()
        pheno.process_images(wrapper, batch_size=16)
    else:
        raise
```

## Logging for long runs

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

logger.info("Starting processing")
try:
    pheno.process_images(wrapper)
    logger.info("Processing complete")
except Exception:
    logger.exception("Processing failed")
    raise
```

## See also

- [Reproducibility](/PhenoMe/guides/best-practices/reproducibility/) - trade-offs with CPU DR
  backends.
- [Data organization](/PhenoMe/guides/best-practices/data-organization/) - avoid re-processing what
  you do not need.
- [External checkpoints](/PhenoMe/guides/external-checkpoints/) - load embeddings
  computed elsewhere.
