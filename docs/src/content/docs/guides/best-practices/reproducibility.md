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
  `force_rgb` (and `batch_size` when set).

The HDF5 file is self-contained for processing parameters and data. For
full reproducibility, also document parameters **not** stored in the
checkpoint:

- `seed` (set at framework init).
- `use_gpu_for_dr` (chooses TorchDR vs sklearn / umap-learn).
- `reference_filters` (for distance analysis).
- `model name` (for example `dinov2_vitb14_reg`).

```python
pheno.save_results("results.h5")

pheno.export_experiment_config(
    "results/config.json",
    reference_filters={"condition": "Control"},
    model_name="dinov2_vitb14_reg",
)
```

When loading: call `find_files` first, then
`pheno.load_results("results.h5")`. The framework restores
configuration and resolves paths automatically.

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
metadata, properties, and processing params in one place - no need to
save them separately.

```python
pheno.save_results("results.h5")

df = pheno.compute_properties(property_preset="basic")
df.to_csv("results/properties.csv", index=False)

dist_results = pheno.compute_reference_distances(
    reference_filters={"condition": "Control"}
)
import pickle
with open("results/distances.pkl", "wb") as f:
    pickle.dump(dist_results, f)
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
