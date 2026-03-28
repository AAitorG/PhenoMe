"""I/O modules for the phenotyping pipeline.

Provides image reading, checkpoint management, and file discovery. See the
top-level :mod:`phenome` module for usage.

Public API:
    - read_image: Load image(s) from path(s) as numpy array.
    - ensure_hwc: Ensure image array is (H, W, C) format.
    - FileDiscovery: Find image files and extract metadata.
    - CheckpointManager: HDF5 checkpoint for incremental embedding/property persistence.
"""

from .checkpoint import CheckpointManager
from .discovery import FileDiscovery, ensure_hwc, read_image

__all__ = [
    "CheckpointManager",
    "FileDiscovery",
    "ensure_hwc",
    "read_image",
]
