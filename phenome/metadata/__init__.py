"""Metadata handling for the phenotyping pipeline.

Provides an object-oriented API for metadata extraction with configurable
column mappings, auto-generated unique IDs, and mask path resolution. See
:mod:`phenome.utils.metadata` for functional helpers.

Public API:
    - MetadataBase: Abstract base for metadata extractors.
    - DefaultMetadata: Minimal extractor (file_path, filename, id).
    - PathTemplateMetadata: Extract from path templates with capture groups.
    - DataFrameMetadata: Look up metadata from a DataFrame.
"""

from .base import MetadataBase
from .sources import DataFrameMetadata, DefaultMetadata, PathTemplateMetadata

__all__ = [
    "DataFrameMetadata",
    "DefaultMetadata",
    "MetadataBase",
    "PathTemplateMetadata",
]
