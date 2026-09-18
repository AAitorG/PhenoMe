---
title: "Reproducibility"
description: Seeds, deterministic mode, configuration export, and choice of dimensionality-reduction backend.
sidebar:
  order: 1
---

## Random seeds

Pass a seed to the framework:

```python
pheno = PhenoMe(seed=42)
```

When a seed is provided, it is used for **all** random operations:

- Determinism (PyTorch, NumPy, Python `random`)
- Clustering (GMM)
- Dimensionality reduction (t-SNE, UMAP)
- Mutual information computation
- Random sampling in reports

When `seed=None` (default), no seeds are set anywhere - runs are
non-deterministic.

For manual control (for example before data loading), call:

```python
from phenome.utils import set_determinism
set_determinism(42)
```

This ensures:

- Same PCA / t-SNE / UMAP projections (within the same backend).
- Reproducible clustering.
- Consistent train/test splits.

## Save your configuration

The checkpoint (HDF5) already stores configuration and results. When
`save_results()` or `checkpoint_path` are used during `process_images()` /
`compute_properties()`, the file contains:

- **Results**: embeddings, `img_path`, metadata, properties.
- **Processing params**: `channel_mode`, `resize_size`, `pad_size`,
  `force_rgb`, `l2_normalize_channels` (and `batch_size` when set).
- **Processing pipeline** (optional `/processing_pipeline`): the ordered
  printed steps (title and detail) for embeddings and, when run, properties.
  Stored as `step_01`, `step_02`, … HDF5 groups with attributes — not JSON.
  Property steps also store `kind=image`, `kind=mask`, or `kind=compute`.
  Temporal processing is session-only.

The HDF5 file is self-contained for processing parameters and data. For
full reproducibility, also call `export_experiment_config()` after the
analysis you want to document. The pipeline records method parameters
silently (model name from `load_dinov2_model`, property presets,
clustering, `reference_filters` from `compute_reference_distances`, and
so on). That call writes JSON; `save_results()` still writes only the
HDF5 file.

```python
pheno.save_results("results.h5")

pheno.export_experiment_config("results/config.json")
```

Optional overrides (`model_name=`, `reference_filters=`) still work if
you want to set values that were not captured from a method call.

To add a Methods / settings section to the HTML report (off by default):

```python
pheno.generate_report("results/report.html", include_run_settings=True)
```

When loading: call `find_files` first, then
`pheno.load_results("results.h5")`. The framework restores
configuration, the processing pipeline (base and properties), and resolves
paths automatically. Inspect it with `pheno.print_processing_pipeline()`
or `pheno.get_processing_pipeline()`, or watch INFO logs during
`process_images` / `load_results`. Warnings
mean “suspicious but processable”; they never change pixels or
parameters. Custom `preprocessing_fn` / `custom_transformations` appear
by name (and optional description) only — PhenoMe does not inspect their
source. Pass `preprocessing_description` / `transformations_description`
when you want that optional note in the INFO processing pipeline log:

```python
pheno.process_images(
    wrapper,
    preprocessing_fn=enhance_contrast,
    preprocessing_description="clip 1st/99th percentiles, then rescale to 0-1",
)
```

If embeddings and
properties are in separate files, use `pheno.load_embeddings(...)`
then `pheno.load_properties(...)` instead.

## Version control

Track your analysis code:

- Use git for scripts.
- Record library versions.
- Save the environment specification.

```bash
pip freeze > requirements-frozen.txt
```

## Save intermediate results

The HDF5 file from `save_results()` or checkpointing stores embeddings,
metadata, properties, processing params, and the base processing pipeline
in one place - no sibling config or pipeline files.

```python
pheno.save_results("results.h5")

df = pheno.compute_properties(property_preset="basic")
df.to_csv("results/properties.csv", index=False)

dist_df = pheno.compute_reference_distances(
    reference_filters={"condition": "Control"}
)
dist_df.to_csv("results/distances.csv")
```

## Dimensionality-reduction backends

:::caution
PCA, t-SNE, and UMAP results may differ slightly between the **TorchDR +
GPU** backend and the **sklearn / umap-learn + CPU** backend. For strict
reproducibility across machines or over time, use one backend consistently.
:::

### Why results can differ

| Reason | Explanation |
|--------|-------------|
| **Different implementations** | TorchDR and sklearn / umap-learn use different algorithms and optimisations. |
| **Numerical precision** | GPU floating-point accumulation order and parallel reductions can diverge slightly from CPU. |
| **Random initialisation** | Even with the same seed, libraries may apply it differently during optimisation. |
| **Approximations** | GPU backends may use approximate k-NN (e.g. FAISS, KeOps); CPU backends may use exact computations. |

### Recommendations

- **Reproducibility**: `use_gpu_for_dr=False` to always use
  sklearn / umap-learn; document the chosen backend and seed.
- **Speed**: `use_gpu_for_dr=True` when available - qualitative
  structure (clusters, trends) is typically preserved even when
  coordinates differ slightly.
- **Large datasets**: install `pykeops` to avoid GPU OOM with t-SNE and
  UMAP; PCA uses a batched `ExactIncrementalPCA` by default.

### `n_components` support

| Method | `n_components` | Notes |
|--------|----------------|-------|
| **PCA** | Any | No restrictions. |
| **UMAP** | Any | No restrictions. |
| **t-SNE** | Any | sklearn (CPU): `n_components >= 4` falls back to `method='exact'` (slower, O(n^2)); otherwise Barnes-Hut. TorchDR (GPU) supports any `n_components` natively. |

## Handling non-determinism that remains

Some non-determinism is intrinsic to the hardware stack (CUDA atomic
additions, concurrent reductions). If a bit-exact match matters:

1. Pin PyTorch, CUDA, and the DR libraries to explicit versions.
2. Set `torch.use_deterministic_algorithms(True)` (done by
   `set_determinism`) and verify no op raises.
3. Prefer the CPU DR backend.
4. Run on the same machine you report from.

## See also

- [Performance](/PhenoMe/guides/best-practices/performance/) - when to trade a little
  reproducibility for speed.
- [Collaboration](/PhenoMe/guides/best-practices/collaboration/) - sharing a fully reproducible bundle.
- [HDF5 protocol](../../../advanced/database_protocol/) - what the
  checkpoint contains.
