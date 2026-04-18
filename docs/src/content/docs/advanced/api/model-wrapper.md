---
title: "Vision model wrappers"
description: ModelWrapper, DinoV2ModelWrapper, load_dinov2_model.
---

Auto-generated from `phenome.utils.model_wrapper`. Rebuild with `npm run prebuild` in `docs/`.

**See also:** [Pipeline](/PhenoMe/advanced/api/pipeline/)

Model wrappers for vision model embedding extraction.

## Model wrappers

### `load_dinov2_model`

```python
load_dinov2_model(
    model_name: str = 'dinov2_vitb14_reg',
    device: torch.device | None = None
) -> DinoV2ModelWrapper
```

Load a DINOv2 model and return a DinoV2ModelWrapper ready for processing.

Convenience helper for DINOv2 users. Uses get_default_device() when
device is not specified. Returns the heritage DinoV2ModelWrapper, which
subclasses ModelWrapper and is fully compatible with the pipeline.

**Args:**

- **`model_name`**: DINOv2 model identifier for torch.hub.
  Common: 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'
- **`device`**: Optional torch.device. If None, uses get_default_device().

**Returns:**

  DinoV2ModelWrapper. Use with process_images(). The underlying
  ``torch.nn.Module`` is ``wrapper.model``.
- **``**:

**Example:**

```python
>>> wrapper = load_dinov2_model()
>>> pheno.find_files("path/to/images")
>>> pheno.process_images(wrapper)

```

### `class ModelWrapper`

Base wrapper for vision model embedding extraction.

Provides only basics: model.eval(), model.to(device), torch.no_grad(), and
a sanity check that output is (B, D). Dict extraction, key lookup, and
tensor flattening are the heritage class's responsibility.

**Args:**

- **`model`**: PyTorch vision model.
- **`device`**: Optional torch.device. If None, uses get_default_device().

<div class="api-method" role="region" aria-labelledby="api-modelwrapper-extract_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-modelwrapper-extract_embeddings"><code>extract_embeddings</code></h4>
</div>

<div class="api-signature">

```python
ModelWrapper.extract_embeddings(
    self,
    tensor: torch.Tensor
) -> Tensor
```

</div>

<div class="api-body">

Extract per-image embeddings.

Delegates to _get_embeddings (heritage class) and validates output shape (B, D).

**Args:**

- **`tensor`**: Input tensor (B, C, H, W) on device.

**Returns:**

  Embedding tensor (B, D).

</div>

</div>

### `class DinoV2ModelWrapper`

Heritage wrapper for DINOv2 models from Meta AI.

Handles dict output with 'x_norm_clstoken' (default key) and squeeze (B,1,D) -> (B,D).

**Args:**

- **`model`**: DINOv2 model (from torch.hub.load('facebookresearch/dinov2', ...)).
- **`device`**: Optional torch.device. If None, uses get_default_device().
- **`embedding_output_key`**: Key for global embedding in forward_features output.
  Default 'x_norm_clstoken' (required for DINOv2).

<div class="api-method" role="region" aria-labelledby="api-dinov2modelwrapper-extract_embeddings">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-dinov2modelwrapper-extract_embeddings"><code>extract_embeddings</code></h4>
</div>

<div class="api-signature">

```python
DinoV2ModelWrapper.extract_embeddings(
    self,
    tensor: torch.Tensor
) -> Tensor
```

</div>

<div class="api-body">

Extract per-image embeddings.

Delegates to _get_embeddings (heritage class) and validates output shape (B, D).

**Args:**

- **`tensor`**: Input tensor (B, C, H, W) on device.

**Returns:**

  Embedding tensor (B, D).

</div>

</div>
