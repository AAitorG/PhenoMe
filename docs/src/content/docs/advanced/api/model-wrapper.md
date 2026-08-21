---
title: "Vision model wrappers"
description: "ModelWrapper, DinoV2ModelWrapper, load_dinov2_model."
editUrl: false
tableOfContents:
  maxHeadingLevel: 3
---

<p><span class="api-tier api-tier--public">Tier: Public API</span></p>

:::note[Auto-generated]
This page is rebuilt from docstrings in [`phenome.utils.model_wrapper`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/utils/model_wrapper.py) (module).
:::

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

**Example:**

```python
>>> wrapper = load_dinov2_model()
>>> pheno.find_files("path/to/images")
>>> pheno.process_images(wrapper)
```

### `class ModelWrapper`

Base wrapper for vision model embedding extraction.

While PhenoMe is primarily built on PyTorch, this interface is designed to be
framework-agnostic. Users can wrap models from any framework (TensorFlow, Keras,
ONNX, etc.) by implementing a custom `ModelWrapper` subclass.

**Contract for custom wrappers:**
1. The `_get_embeddings(tensor)` method must be implemented.
2. It receives a `torch.Tensor` (B, C, H, W) as input.
3. It MUST return a `torch.Tensor` (B, D) as output.

**Args:**

- **`model`**: Vision model (PyTorch module, or any other framework model).
- **`device`**: Optional torch.device. If None, uses get_default_device().

**Example (Keras):**

```python
>>> import tensorflow as tf
>>> import numpy as np
>>> class KerasWrapper(ModelWrapper):
...     def _get_embeddings(self, tensor):
...         x = np.transpose(tensor.cpu().numpy(), (0, 2, 3, 1))
...         out = self.model.predict(x, verbose=0)
...         return torch.from_numpy(out).to(self.device)
>>> model = tf.keras.applications.MobileNetV2(pooling='avg')
>>> wrapper = KerasWrapper(model)
>>> pheno.process_images(wrapper)
```

<div class="api-method" role="region" aria-labelledby="api-modelwrapper-__init__">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-modelwrapper-__init__"><code>__init__</code></h4>
</div>

<div class="api-signature">

```python
ModelWrapper(
    model: Any,
    device: torch.device | None = None
)
```

</div>

</div>

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
Even for non-PyTorch models, this method ensures the final output is a
PyTorch tensor compatible with the rest of the pipeline.

**Args:**

- **`tensor`**: Input tensor (B, C, H, W) on device.

**Returns:**

  Embedding tensor (B, D).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-modelwrapper-sync_device">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-modelwrapper-sync_device"><code>sync_device</code></h4>
</div>

<div class="api-signature">

```python
ModelWrapper.sync_device(
    self,
    device: torch.device | None = None
) -> None
```

</div>

<div class="api-body">

Move the wrapped model to *device* and set eval mode when supported.

Called from ``__init__`` and again by ``EmbeddingExtractor`` before each batch
so the model stays on the pipeline device after user changes.

**Args:**

- **`device`**: Target device. If None, uses the wrapper's current ``self.device``.

**Raises:**

- **`RuntimeError`**: If ``model.to(device)`` or ``model.eval()`` fails.

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
- **`model_name`**: Optional identifier stored for experiment export
  (for example ``dinov2_vitb14_reg``).

<div class="api-method" role="region" aria-labelledby="api-dinov2modelwrapper-__init__">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-dinov2modelwrapper-__init__"><code>__init__</code></h4>
</div>

<div class="api-signature">

```python
DinoV2ModelWrapper(
    model: Any,
    device: torch.device | None = None,
    embedding_output_key: str = 'x_norm_clstoken',
    model_name: str | None = None
)
```

</div>

</div>

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
Even for non-PyTorch models, this method ensures the final output is a
PyTorch tensor compatible with the rest of the pipeline.

**Args:**

- **`tensor`**: Input tensor (B, C, H, W) on device.

**Returns:**

  Embedding tensor (B, D).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-dinov2modelwrapper-sync_device">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-dinov2modelwrapper-sync_device"><code>sync_device</code></h4>
</div>

<div class="api-signature">

```python
DinoV2ModelWrapper.sync_device(
    self,
    device: torch.device | None = None
) -> None
```

</div>

<div class="api-body">

Move the wrapped model to *device* and set eval mode when supported.

Called from ``__init__`` and again by ``EmbeddingExtractor`` before each batch
so the model stays on the pipeline device after user changes.

**Args:**

- **`device`**: Target device. If None, uses the wrapper's current ``self.device``.

**Raises:**

- **`RuntimeError`**: If ``model.to(device)`` or ``model.eval()`` fails.

</div>

</div>
