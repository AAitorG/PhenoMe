---
title: "Custom Model Wrappers"
description: "Learn how to use non-PyTorch models (TensorFlow, Keras, ONNX) with PhenoMe."
---

While PhenoMe is built on PyTorch, its modular architecture allows you to use models from any machine learning framework. This is achieved through the `ModelWrapper` interface.

:::tip[Tutorial Reference]
For a hands-on walkthrough of custom models and advanced preprocessing, see **[Tutorial 06: Custom Models and Preprocessing](https://github.com/PhenoMe-Ecosystem/PhenoMe/blob/main/Notebooks/tutorials/06_custom_model_and_preprocessing.ipynb)**.
:::

---

## The ModelWrapper Contract

To use a custom model, you must create a subclass of `ModelWrapper` and implement the `_get_embeddings` method.

The contract is simple:
1. **Input**: A `torch.Tensor` of shape `(B, C, H, W)` on the target device.
2. **Output**: A `torch.Tensor` of shape `(B, D)` where `D` is the embedding dimension.

### Preprocessing

`custom_transformations` replaces PhenoMe's default intensity scaling, pad, resize, and ImageNet normalization. The callable receives one NumPy image `(H, W, C)` and must return a tensor `(C, H, W)`. The loader stacks those tensors and moves the batch to the pipeline device before `_get_embeddings`.

`read_image` casts every file to `np.float32`, so that is the dtype at the start of `custom_transformations`. Values stay in the file's range: about 0–255 for 8-bit images, about 0–65535 for 16-bit. `ToTensor` turns that into `torch.float32` of shape `(C, H, W)` and does not divide by 255, because the array is already floating point. `_get_embeddings` then receives `torch.float32` of shape `(B, C, H, W)`. Return `torch.float32` of shape `(B, D)`. `get_embeddings()` stores the result as `np.float32`.

For a non-PyTorch model, convert that tensor to NumPy inside the wrapper and convert the result back.

---

## Example: Keras MobileNetV2

`preprocess_input` maps pixels from 0–255 to [-1, 1]. Scale the image to [0, 1] and resize to 224 in `custom_transformations`, then call `preprocess_input` on `x * 255`. `channel_mode="combined"` feeds the model one RGB image. Both functions keep floating point: `float32` in, `float32` out.

```python
import numpy as np
import torch
import tensorflow as tf
from phenome import PhenoMe
from phenome.utils.model_wrapper import ModelWrapper

class KerasModelWrapper(ModelWrapper):
    """Wrapper for Keras/TensorFlow models."""

    def __init__(self, model, device=None):
        # Initialize the base class
        super().__init__(model, device=device)

        # Why pass 'device'?
        # 1. Base class stores it in self.device.
        # 2. Even if Keras runs on GPU, the final output must be
        #    a PyTorch tensor on the device PhenoMe expects.
        # 3. We use self.device in _get_embeddings to move the
        #    result back to the correct PyTorch device.

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        # 1. Convert PyTorch tensor (B, C, H, W) to NumPy
        x_np = tensor.detach().cpu().numpy()

        # 2. Transpose to Keras format (B, H, W, C)
        x_np = np.transpose(x_np, (0, 2, 3, 1))

        # 3. Run Keras inference
        # .predict() returns a NumPy array of shape (B, D)
        embeddings_np = self.model.predict(x_np, verbose=0)

        # 4. Convert back to PyTorch tensor on the correct device
        return torch.from_numpy(embeddings_np).to(self.device)

# --- Usage in Pipeline ---

# 1. Load a pre-trained Keras model
keras_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg'
)

# 2. Wrap the model
wrapper = KerasModelWrapper(keras_model)

# 3. Initialize PhenoMe and process images
pm = PhenoMe()
pm.find_files("path/to/images")

# The resize_size should match the model's expected input
pm.process_images(wrapper, resize_size=224)

# 4. Access results
print(f"Extracted {len(pm.results.embeddings)} embeddings")
```

---

## Example: Torchvision ResNet18

`ResNet18_Weights.IMAGENET1K_V1.transforms()` resizes the shorter side to 256, center-crops 224, and applies ImageNet normalization. It expects a `torch.float32` tensor in [0, 1], so scale the `np.float32` image first. Replacing `fc` with `Identity` returns the 512-d pooled features, still `torch.float32`.

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from phenome import PhenoMe
from phenome.utils import ModelWrapper, TypeMaxNorm

weights = ResNet18_Weights.IMAGENET1K_V1
# in:  np.float32 (H, W, C), file range
# out: torch.float32 (3, 224, 224), ImageNet-normalized
resnet_preprocess = T.Compose([
    T.ToTensor(),          # torch.float32 (C, H, W), file range unchanged
    TypeMaxNorm(),         # torch.float32 (C, H, W) in [0, 1]
    weights.transforms(),  # torch.float32 (3, 224, 224), normalized
])

class ResNet18Wrapper(ModelWrapper):
    """Wrapper for torchvision ResNet18."""

    def __init__(self, device=None):
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        super().__init__(model, device=device)

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        # in:  torch.float32 (B, 3, 224, 224), ImageNet-normalized
        # out: torch.float32 (B, 512)
        return self.model(tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = ResNet18Wrapper(device=device)

pm = PhenoMe(device=device)
pm.find_files("path/to/images")
pm.process_images(
    wrapper,
    channel_mode="combined",
    custom_transformations=resnet_preprocess,
)
print(pm.get_embeddings().shape)  # (N, 512)
```

---

## Example: Hugging Face DINOv2

Install `transformers` separately. `AutoImageProcessor` is that checkpoint's resize, crop, and normalization, so pass it as `custom_transformations`. It rescales `np.float32` values by 1/255, so map the file range onto 0–255 first. The processor returns `torch.float32` with a leading batch dimension; drop it. `pooler_output` is `torch.float32` of shape `(B, 768)`.

```python
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from phenome import PhenoMe
from phenome.utils import ModelWrapper

MODEL_ID = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
hf_model = AutoModel.from_pretrained(MODEL_ID)

def huggingface_preprocess(image: np.ndarray) -> torch.Tensor:
    # in:  np.float32 (H, W, C), file range
    # out: torch.float32 (3, 224, 224), processor-normalized
    image = np.asarray(image, dtype=np.float32)
    peak = float(image.max()) if image.size else 0.0
    if peak <= 1.0:
        image = image * 255.0          # still np.float32, now 0-255
    elif peak > 255.0:
        image = image * (255.0 / 65535.0)
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    return pixels.squeeze(0)           # (1, 3, 224, 224) -> (3, 224, 224)

class HuggingFaceWrapper(ModelWrapper):
    """Wrapper for a Hugging Face vision model."""

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        # in:  torch.float32 (B, 3, 224, 224), processor-normalized
        # out: torch.float32 (B, 768)
        return self.model(pixel_values=tensor).pooler_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = HuggingFaceWrapper(hf_model, device=device)

pm = PhenoMe(device=device)
pm.find_files("path/to/images")
pm.process_images(
    wrapper,
    channel_mode="combined",
    custom_transformations=huggingface_preprocess,
)
print(pm.get_embeddings().shape)  # (N, 768) for dinov2-base
```

---

## Common Pitfalls

### Channel Ordering
PyTorch uses `(Channels, Height, Width)`, while TensorFlow/Keras typically uses `(Height, Width, Channels)`. Always remember to transpose your arrays using `np.transpose(arr, (0, 2, 3, 1))` before passing them to a non-PyTorch model.

### Device Management
Non-PyTorch frameworks handle GPUs differently.
- **TensorFlow/Keras**: Usually manages devices globally via `tf.config.list_physical_devices()`.
- **ONNX Runtime**: Uses "Execution Providers" (e.g., `CUDAExecutionProvider`).

Ensure your framework is using the same physical device as PhenoMe to avoid unnecessary data transfers or memory conflicts.

### Normalization
PhenoMe's default stack scales intensities to [0, 1] and then applies ImageNet mean and standard deviation. Use `custom_transformations` when the model expects a different recipe, as in the examples above.

Custom callables are treated as opaque: the processing pipeline log records their name. PhenoMe does not read their source. Pass an optional `preprocessing_description` / `transformations_description` so the INFO log (and `get_processing_pipeline()`) can say what the callable does:

```python
def enhance_contrast(image):
    p1, p99 = np.percentile(image, (1, 99))
    return np.clip(image, p1, p99)

pm.process_images(
    wrapper,
    resize_size=224,
    preprocessing_fn=enhance_contrast,
    preprocessing_description="clip 1st/99th percentiles",
    # custom_transformations=my_transforms,  # optional
    # transformations_description="Resize 224 then ImageNet normalize",
)
```

The description is optional. If you omit it, the log still shows the function name with internals unknown. `print_processing_pipeline()` or `get_processing_pipeline()` shows whether ImageNet normalization is in the stack. It stays when you only pass `preprocessing_fn`. It is replaced when you pass `custom_transformations`.

---

## See Also

| Topic | Document |
|-------|----------|
| ModelWrapper API | [API Reference](/PhenoMe/advanced/api/model-wrapper/) |
| Image Processing | [Pipeline API](/PhenoMe/advanced/api/pipeline/) |
| Custom Plugins | [Plugins Guide](/PhenoMe/guides/plugins/) |
