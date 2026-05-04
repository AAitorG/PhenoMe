"""
@section I/O & Discovery

I/O and file discovery modules for the phenotyping pipeline.

This package provides tools for discovering image files, reading images into standardized
formats, and managing checkpoints for resumable processing and lazy-loading of embeddings.

**Public API:**

File discovery and reading:
- `read_image`: Load image(s) from file path(s) as numpy array.
- `ensure_hwc`: Ensure image array is in (Height, Width, Channel) format.
- `FileDiscovery`: Discover image files in directories and extract metadata.

Checkpoint management:
- `CheckpointManager`: Manage HDF5 checkpoints for incremental embeddings and properties.
  Supports lazy loading (on-demand reading from disk).

Checkpoint alignment:
- `align_by_paths`: Align embeddings across datasets based on file paths.
- `align_by_metadata`: Align embeddings across datasets based on metadata columns.

**Use cases:**

- Stream large datasets that don't fit in memory (lazy checkpoints)
- Resume interrupted embedding extraction mid-process
- Align embeddings from multiple runs before comparison

**See Also:**
For usage in the main pipeline, see `phenome.PhenoMe.process_images()`.
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
