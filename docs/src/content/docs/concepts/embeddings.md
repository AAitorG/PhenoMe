---
title: "Embeddings & Models"
description: Visual fingerprints, supported models, and the embedding lifecycle.
---

An **embedding** is a digital "fingerprint" of an image: a fixed-size vector of numbers that captures the most important visual features—textures, shapes, patterns—without requiring a human to define them.

```text
Image -> Vision model (the "brain") -> Visual fingerprint (the numbers)
```

## Supported models

Common choices:

- **DINOv2** (default): self-supervised vision transformer by Meta AI.
- **CLIP**: vision-language models.
- **Custom models**: any model can be used with the [Vision model wrappers](/PhenoMe/advanced/api/model-wrapper/).

DINOv2 transfers well to diverse microscopy images without training or fine-tuning, which is why it is the default.

### Embedding dimensions

| Model | Embedding dimension |
|-------|---------------------|
| ViT-S/14 (Small) | 384 |
| ViT-B/14 (Base) | 768 |
| ViT-L/14 (Large) | 1024 |
| ViT-g/14 (Giant) | 1536 |

For custom models see [Vision model wrappers](/PhenoMe/advanced/api/model-wrapper/).

## Embedding lifecycle: eager vs lazy

| Mode | When | Where | Access |
|------|------|-------|--------|
| **Eager** | No checkpoint used during `process_images()` | `results.embeddings` (in-memory `np.ndarray`) | Direct slice |
| **Lazy** | Checkpoint path provided or `load_results()` used | On disk (HDF5); `results.embeddings` is `None` | `get_embeddings()` loads rows on demand |

Use `get_embeddings()` for lazy-safe access. For full HDF5 storage details, see the [Database protocol](/PhenoMe/advanced/database_protocol/).

## Temporal images (in-memory)

Use `process_temporal_images()` to add new images **temporarily** to an existing analysis without re-running `process_images`. Temporal images are kept in memory only (not persisted to checkpoint) and are tagged with `metadata['source'] = 'NEW'`. Call `clear_temporal_data()` to remove them. See [Temporal images](/PhenoMe/workflows/#5-temporal-images-testing-new-data-quickly).
