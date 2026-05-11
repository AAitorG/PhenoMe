---
title: "Custom Model Wrappers"
description: "Learn how to use non-PyTorch models (TensorFlow, Keras, ONNX) with PhenoMe."
---

While PhenoMe is built on PyTorch, its modular architecture allows you to use models from any machine learning framework. This is achieved through the `ModelWrapper` interface.

---

## The ModelWrapper Contract

To use a custom model, you must create a subclass of `ModelWrapper` and implement the `_get_embeddings` method.

The contract is simple:
1. **Input**: A `torch.Tensor` of shape `(B, C, H, W)` on the target device.
2. **Output**: A `torch.Tensor` of shape `(B, D)` where `D` is the embedding dimension.

For non-PyTorch models, you will typically use **numpy** as a bridge between frameworks.

---

## Step-by-Step: Building a Keras Wrapper

Here is how you can wrap a Keras/TensorFlow model for use in PhenoMe.

### 1. Define the Wrapper Class

```python
import numpy as np
import torch
from phenome.utils.model_wrapper import ModelWrapper

class MyKerasWrapper(ModelWrapper):
    def __init__(self, model, device=None):
        # Initialize the base class with the model and device
        super().__init__(model, device=device)

        # Keras models don't use .to(device) like PyTorch.
        # Ensure your TensorFlow/Keras environment is configured
        # for the correct GPU/CPU before initializing.
```

### 2. Implement `_get_embeddings`

```python
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        # Step A: Convert PyTorch tensor to NumPy
        # PhenoMe provides (B, Channels, Height, Width)
        x_np = tensor.detach().cpu().numpy()

        # Step B: Adjust format for Keras (usually B, H, W, C)
        x_np = np.transpose(x_np, (0, 2, 3, 1))

        # Step C: Run inference
        # Keras .predict() returns a NumPy array
        embeddings_np = self.model.predict(x_np, verbose=0)

        # Step D: Convert back to PyTorch (B, D)
        return torch.from_numpy(embeddings_np).to(self.device)
```

### 3. Use it in the Pipeline

```python
from phenome import PhenoMe
import tensorflow as tf

# Load your Keras model
keras_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg'
)

# Wrap it
wrapper = MyKerasWrapper(keras_model)

# Run PhenoMe
pm = PhenoMe()
pm.find_files("path/to/images")
pm.process_images(wrapper, resize_size=224)
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
| Image Processing | [Pipeline Guide](/PhenoMe/guides/image-processing/) |
| Custom Plugins | [Plugins Guide](/PhenoMe/guides/plugins/) |
