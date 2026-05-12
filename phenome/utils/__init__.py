"""
@section Utilities

Utility modules for model wrappers, metadata extraction, property factories, and device management.

This package provides reusable components for building and extending phenotyping pipelines.

**Model wrapping:**
- `ModelWrapper`: Base class for custom embedding models. Override `_get_embeddings()` to use
  any pretrained model.
- `DinoV2ModelWrapper`: Wrapper for DINOv2 models (ViT-B/14, ViT-L/14, etc.).
- `load_dinov2_model`: Convenience function to load a DINOv2 model.

**Metadata extraction:**
- `default_metadata_from_path`: Extract metadata from simple file paths.
- `get_metadata_from_path`: Advanced path-based metadata extraction.
- `make_dataframe_metadata_fn`: Create metadata extractor from a DataFrame.

**Property factories (create custom cell/object properties):**
- `create_regionprops_function`: Morphological properties (area, eccentricity, solidity, etc.).
- `create_intensity_function`: Intensity statistics (mean, std, quantiles).
- `create_masked_intensity_function`: Intensity in masked regions.
- `create_texture_function`: Texture features (Haralick, LBP, etc.).
- `create_concentric_ring_function`: Ring-based intensity features.
- `create_blur_effect_function`: Blur/focus metrics.
- `create_entropy_function`: Entropy and information metrics.
- `get_preset_property_functions`: List all available preset properties.

**Image transforms:**
- `TransformBuilder`: Compose PyTorch transforms (resize, pad, normalize).
- `PadToSize`: Pad image to specified size.
- `TypeMaxNorm`: Normalize by dtype max value.
- `normalize_by_dtype_max`: Normalize array by dtype maximum.
- `scale_minmax`: Min-max scaling.
- `quantile_normalize`: Quantile-based normalization.

**Device management:**
- `get_default_device`: Get current device (CPU or GPU).
- `set_determinism`: Enable deterministic behavior (seeds, etc.).

**See Also:**
For metadata configuration objects, see `phenome.metadata`.
For plugin registration, see `phenome.plugins.register_property()`.
"""

from ..metadata import (
    DataFrameMetadata,
    DefaultMetadata,
    MetadataBase,
    PathTemplateMetadata,
)
from .device import get_default_device, set_determinism
from .metadata import (
    default_metadata_from_path,
    get_metadata_from_path,
    make_dataframe_metadata_fn,
)
from .model_wrapper import (
    DinoV2ModelWrapper,
    ModelWrapper,
    load_dinov2_model,
)
from .property_factories import (
    create_blur_effect_function,
    create_concentric_ring_function,
    create_entropy_function,
    create_intensity_function,
    create_masked_intensity_function,
    create_regionprops_function,
    create_texture_function,
    get_preset_property_functions,
)
from .transforms import (
    PadToSize,
    TransformBuilder,
    TypeMaxNorm,
    normalize_by_dtype_max,
    quantile_normalize,
    scale_minmax,
)

__all__ = [
    "DataFrameMetadata",
    "DefaultMetadata",
    "DinoV2ModelWrapper",
    "MetadataBase",
    "ModelWrapper",
    "PadToSize",
    "PathTemplateMetadata",
    "TransformBuilder",
    "TypeMaxNorm",
    "create_blur_effect_function",
    "create_concentric_ring_function",
    "create_entropy_function",
    "create_intensity_function",
    "create_masked_intensity_function",
    "create_regionprops_function",
    "create_texture_function",
    "default_metadata_from_path",
    "get_default_device",
    "get_metadata_from_path",
    "get_preset_property_functions",
    "load_dinov2_model",
    "make_dataframe_metadata_fn",
    "normalize_by_dtype_max",
    "quantile_normalize",
    "scale_minmax",
    "set_determinism",
]
