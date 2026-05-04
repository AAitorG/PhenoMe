"""
@section Plugins

Plugin registry and builtin extensions for phenome.

This package provides extensibility hooks for registering custom properties, metadata
extractors, and report sections. It also exports builtin property functions.

**Property registry:**
- `register_property`: Register a custom property function.
- `get_property`: Retrieve a registered property function by name.
- `list_properties`: List all registered property functions.

**Metadata registry:**
- `register_metadata_extractor`: Register a custom metadata extractor.
- `get_metadata_extractor`: Retrieve a registered metadata extractor.

**Report sections:**
- `register_report_section`: Register a custom report section.
- `get_report_sections`: Retrieve all registered report sections.

**Builtin properties:**
- `get_blob_properties`: Wrapper around scikit-image `regionprops` for blob analysis.
- `my_custom_max_intensity`: Example custom property function.

**Example: Register custom property**

```python
from phenome.plugins import register_property, get_property

def my_property(image, mask=None):
    '''Custom property: average brightness.'''
    if mask is not None:
        image = image[mask]
    return float(image.mean())

register_property("avg_brightness", my_property)

# Later, use in pipeline:
pm = PhenoMe()
prop_fn = get_property("avg_brightness")
properties = [prop_fn]
pm.compute_properties(properties)
```

**See Also:**
For property factories and presets, see `phenome.utils.property_factories`.
For built-in metadata classes, see `phenome.metadata`.
"""

from .builtin_properties import get_blob_properties, my_custom_max_intensity
from .registry import (
    get_metadata_extractor,
    get_property,
    get_report_sections,
    list_properties,
    register_metadata_extractor,
    register_property,
    register_report_section,
)

__all__ = [
    "get_blob_properties",
    "get_metadata_extractor",
    "get_property",
    "get_report_sections",
    "list_properties",
    "my_custom_max_intensity",
    "register_metadata_extractor",
    "register_property",
    "register_report_section",
]
