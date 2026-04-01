"""Utility modules for the phenotyping pipeline.

Provides model wrappers, metadata helpers, property factories, and device utilities.
See the top-level :mod:`phenome` module for usage.

Public API:
    - load_dinov2_model: Load DINOv2 as a DinoV2ModelWrapper.
    - ModelWrapper, DinoV2ModelWrapper: Base and DINOv2-specific wrappers.
    - default_metadata_from_path, get_metadata_from_path, make_dataframe_metadata_fn
    - create_regionprops_function, create_intensity_function, create_texture_function, ...
    - get_default_device, set_determinism
    - TransformBuilder, PadToSize, TypeMaxNorm
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
    "scale_minmax",
    "set_determinism",
]
