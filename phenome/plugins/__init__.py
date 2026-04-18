"""Plugin registry for extensibility.

Provides registries for metadata extractors, report sections, and property functions.
See the top-level `phenome` module for usage.

Public API:
    - register_property, get_property, list_properties
    - register_metadata_extractor, get_metadata_extractor
    - register_report_section, get_report_sections
    - get_blob_properties: Built-in blob-based property function
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
