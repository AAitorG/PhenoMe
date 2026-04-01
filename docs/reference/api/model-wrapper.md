# Vision Model Wrappers

Base and heritage classes for processing images through PyTorch vision models, handling tensor extraction and flattening.

**Related:** [Getting Started](../../getting-started.md) · [Core Concepts: Supported models](../../concepts.md#supported-models) · [Pipeline](pipeline.md)

---

### Base ModelWrapper

#### `ModelWrapper`

```python
ModelWrapper(model: torch.nn.Module, device: Optional[torch.device] = None)
```
Abstract base class. Wraps a PyTorch model and ensures output is a 2D tensor `(B, D)`.
Heritage classes must implement `_get_embeddings(tensor)`.

**Parameters:**
- **model** (*torch.nn.Module*) – PyTorch vision model.
- **device** (*Optional[torch.device]*) – Device for computation (CPU or CUDA). Defaults to `get_default_device()`.

**Methods:**
- `extract_embeddings(tensor: torch.Tensor) -> torch.Tensor`
  Passes a `(B, C, H, W)` batch to `_get_embeddings()` under `torch.no_grad()`, forcing eval mode, and validates the output shape is `(B, D)`.

---

### DinoV2ModelWrapper

#### `DinoV2ModelWrapper`

```python
DinoV2ModelWrapper(model: torch.nn.Module, device: Optional[torch.device] = None, embedding_output_key: str = 'x_norm_clstoken')
```
Wrapper for DINOv2 models. Handles dictionary extraction using `embedding_output_key` and squeezes from `(B, 1, D)` to `(B, D)`.

**Parameters:**
- **model** (*torch.nn.Module*) – DINOv2 model.
- **device** (*Optional[torch.device]*) – Execution device.
- **embedding_output_key** (*str*) – Global embedding key. Defaults to `'x_norm_clstoken'`.

**Example:**
```python
from phenome import load_dinov2_model

wrapper = load_dinov2_model('dinov2_vitb14_reg')
```

---

### Custom Model Example

To use a different vision model (e.g. ResNet, CLIP), subclass `ModelWrapper` and implement `_get_embeddings(tensor)` to return a `(B, D)` tensor:

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from phenome.utils import ModelWrapper

class ResNetModelWrapper(ModelWrapper):
    """Wrapper for ResNet backbones that produce global pooling features."""

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: (B, C, H, W) -> out: (B, D)
        out = self.model(tensor)
        if out.ndim == 4:
            out = out.flatten(1)  # (B, C, 1, 1) -> (B, C)
        elif out.ndim == 3:
            out = out.squeeze(1)
        return out

# Usage
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
wrapper = ResNetModelWrapper(model)
pheno.process_images(wrapper)
```

The only requirement is that `_get_embeddings(tensor)` returns a 2D tensor of shape `(batch_size, embedding_dim)`.
