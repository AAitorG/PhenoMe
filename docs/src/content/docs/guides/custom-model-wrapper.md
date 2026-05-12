---
title: "Custom Model Wrappers"
description: "Learn how to use non-PyTorch models (TensorFlow, Keras, ONNX) with PhenoMe."
---

While PhenoMe is built on PyTorch, its modular architecture allows you to use models from any machine learning framework. This is achieved through the `ModelWrapper` interface.

:::tip[Tutorial Reference]
For a hands-on walkthrough of custom models and advanced preprocessing, see **[Tutorial 06: Custom Models and Preprocessing](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/06_custom_model_and_preprocessing.ipynb)**.
:::

---

## The ModelWrapper Contract

To use a custom model, you must create a subclass of `ModelWrapper` and implement the `_get_embeddings` method.

The contract is simple:
1. **Input**: A `torch.Tensor` of shape `(B, C, H, W)` on the target device.
2. **Output**: A `torch.Tensor` of shape `(B, D)` where `D` is the embedding dimension.

### Preprocessing with PyTorch
Even if you are using a non-PyTorch model (like Keras), you can still leverage PyTorch's high-performance transformations. You can provide a `custom_transformations` object (e.g., `torchvision.transforms.Compose`) to `process_images()`. These transformations run on the GPU (if available) before the tensor is passed to your wrapper's `_get_embeddings` method.

For non-PyTorch models, you will typically use **numpy** as a bridge between frameworks inside the wrapper.

---

## Example: Keras MobileNetV2

This example shows a fully functional implementation of a `ModelWrapper` for a Keras model, including the necessary data transformations and pipeline integration.

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
print(f"Extracted {len(pm.embeddings)} embeddings")
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
PhenoMe's default transformations might not match what your model expects. You can provide a custom `preprocessing_fn` to `process_images()` or handle normalization inside your wrapper's `_get_embeddings` method.

---

## See Also

| Topic | Document |
|-------|----------|
| ModelWrapper API | [API Reference](/PhenoMe/advanced/api/model-wrapper/) |
| Image Processing | [Pipeline API](/PhenoMe/advanced/api/pipeline/) |
| Custom Plugins | [Plugins Guide](/PhenoMe/guides/plugins/) |
