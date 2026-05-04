r"""
@section Metadata

Metadata handling for extensible and configurable file discovery.

This package provides an object-oriented API for extracting and managing metadata
from file paths, DataFrames, and custom sources. Metadata is extracted alongside
images during file discovery and used to organize results, group analysis, and
annotate visualizations.

**Metadata extractors (classes):**
- `MetadataBase`: Abstract base class for custom metadata extractors.
- `DefaultMetadata`: Minimal extractor (file path, filename, auto-generated ID).
- `PathTemplateMetadata`: Extract metadata by matching file paths against regex patterns.
- `DataFrameMetadata`: Look up metadata from a DataFrame indexed by file path.

**Quick start:**

```python
from phenome import PhenoMe, PathTemplateMetadata

# Extract batch and sample ID from paths like: batch_1/sample_42.tif
metadata_fn = PathTemplateMetadata(template=r"batch_(?P<batch>\d+)/sample_(?P<sample_id>\d+)")

pm = PhenoMe()
df = pm.find_files("images/", metadata_fn=metadata_fn)
print(df.columns)  # ['file_path', 'batch', 'sample_id']
```

**Functional helpers:**
For functional metadata extraction, see `phenome.utils.metadata`:
- `default_metadata_from_path`: Simple functional wrapper.
- `get_metadata_from_path`: Advanced functional wrapper.
- `make_dataframe_metadata_fn`: Create extractor from DataFrame.

**See Also:**
For factory functions and presets, see `phenome.utils.property_factories`.
For metadata utilities in math operations, see `phenome.core.results_metadata`.
"""

from .base import MetadataBase
from .sources import DataFrameMetadata, DefaultMetadata, PathTemplateMetadata

__all__ = [
    "DataFrameMetadata",
    "DefaultMetadata",
    "MetadataBase",
    "PathTemplateMetadata",
]
