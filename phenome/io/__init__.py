"""I/O modules for the phenotyping pipeline.

Provides image reading, checkpoint management, and file discovery. See the
top-level `phenome` module for usage.

Public API:
    - read_image: Load image(s) from path(s) as numpy array.
    - ensure_hwc: Ensure image array is (H, W, C) format.
    - FileDiscovery: Find image files and extract metadata.
    - CheckpointManager: HDF5 checkpoint for incremental embedding/property persistence.
"""

from .checkpoint import CheckpointManager
from .checkpoint_alignment import (
    compute_metadata_alignment as align_by_metadata,
)
from .checkpoint_alignment import (
    compute_path_alignment as align_by_paths,
)
from .discovery import FileDiscovery, ensure_hwc, read_image

__all__ = [
    "CheckpointManager",
    "FileDiscovery",
    "align_by_metadata",
    "align_by_paths",
    "ensure_hwc",
    "read_image",
]
