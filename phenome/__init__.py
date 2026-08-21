"""
@section Core Pipeline

PhenoMe Package

A modular, **dataset-agnostic** and **model-agnostic** pipeline for phenotyping
analysis using deep learning embeddings.

**Overview:**
PhenoMe orchestrates the full phenotyping workflow: discover images, extract embeddings
with custom or pretrained models, compute cell/object properties, perform statistical
analysis, and generate interactive visualizations and reports.

**Public API:**

Core orchestration:
- `PhenoMe`: Main pipeline class; find images, process embeddings, compute properties, analyze.
- `PhenoMeResults`: Container for embeddings, properties, metadata, and analysis results.

Metadata helpers:
- `DefaultMetadata`: Simple key-value metadata.
- `PathTemplateMetadata`: Extract metadata from file paths using templates.
- `DataFrameMetadata`: Use a DataFrame as metadata source.
- `default_metadata_from_path`, `get_metadata_from_path`, `make_dataframe_metadata_fn`:
  Metadata factory functions.

Model loading:
- `load_dinov2_model`: Load pretrained DINOv2 model.

Property factories (create custom cell/object properties):
- `create_regionprops_function`, `create_intensity_function`,
  `create_masked_intensity_function`, `create_texture_function`,
  `create_concentric_ring_function`, `create_blur_effect_function`,
  `create_entropy_function`: Built-in property creators.
- `get_preset_property_functions`: List all available preset properties.

**Data flow:**

```
find_files()
    ↓ returns DataFrame of image paths
set_file_df() or directly via DataFrame
    ↓ store file locations
process_images(model)
    ↓ extract embeddings, populate results.embeddings
compute_properties(property_functions)
    ↓ compute morphological/biological features
compute_clustering() / detect_outliers() / ...
    ↓ statistical analysis
plot_pca() / plot_tsne() / plot_umap() / ...
    ↓ visualization
generate_report()
    ↓ HTML report
```

**Model compatibility:**
- **DINOv2:** Use `load_dinov2_model()` or wrap with `DinoV2ModelWrapper`.
- **Custom:** Subclass `ModelWrapper`, implement `_get_embeddings(images) -> (B, D)`.

**For advanced usage, import from submodules:**
- `phenome.core`: Math utilities (DR, correlation, validation, export).
- `phenome.io`: File I/O (`read_image`, `FileDiscovery`, `CheckpointManager`).
- `phenome.mixins`: Composition classes (dataset, properties, analysis, visualization).
- `phenome.plugins`: Property registry and custom property registration.
- `phenome.report`: HTML report generation.
- `phenome.utils`: Device management, transforms, metadata and property factories.

**Logging:**

Progress messages are visible at INFO level by default:

```python
import logging
logging.getLogger("phenome").setLevel(logging.WARNING)  # Suppress
```

**Example:**

Basic phenotyping workflow:

```python
from phenome import PhenoMe, load_dinov2_model

# Initialize pipeline
pm = PhenoMe(device="cuda", seed=42)

# Find images
pm.find_files("path/to/images/")

# Extract embeddings
model = load_dinov2_model(model_name="dinov2_vitb14")
pm.process_images(model, batch_size=32)

# Compute properties
pm.compute_properties(property_preset="basic")

# Analyze
pm.compute_clustering(method="kmeans", n_clusters=5)
pm.compute_pca()

# Visualize
pm.plot_pca(color_by="clustering")
```

**See Also:**
For architecture, design patterns, and extensibility, see the [Architecture guide](https://AAitorG.github.io/PhenoMe/guides/architecture/).
"""

# Initialise notebook-safe logging on first import
from importlib.metadata import PackageNotFoundError, version

from ._logging import get_logger as _get_logger
from .core import PhenoMeResults
from .metadata import (
    DataFrameMetadata,
    DefaultMetadata,
    MetadataBase,
    PathTemplateMetadata,
)
from .pipeline import PhenoMe
from .plotly_display import register_plotly_display_defaults

# Metadata functions and classes
from .utils.metadata import (
    default_metadata_from_path,
    get_metadata_from_path,
    make_dataframe_metadata_fn,
)

# Core workflow helpers
from .utils.model_wrapper import load_dinov2_model

# Property factories
from .utils.property_factories import (
    create_blur_effect_function,
    create_concentric_ring_function,
    create_entropy_function,
    create_intensity_function,
    create_masked_intensity_function,
    create_regionprops_function,
    create_texture_function,
    get_preset_property_functions,
)

try:
    __version__ = version("phenome")
except PackageNotFoundError:
    __version__ = "1.4.0"

__all__ = [
    "DataFrameMetadata",
    "DefaultMetadata",
    "MetadataBase",
    "PathTemplateMetadata",
    "PhenoMe",
    "PhenoMeResults",
    "__version__",
    # Property factories
    "create_blur_effect_function",
    "create_concentric_ring_function",
    "create_entropy_function",
    "create_intensity_function",
    "create_masked_intensity_function",
    "create_regionprops_function",
    "create_texture_function",
    # Metadata functions
    "default_metadata_from_path",
    "get_metadata_from_path",
    # Property factories
    "get_preset_property_functions",
    # Default models
    "load_dinov2_model",
    "make_dataframe_metadata_fn",
]

_get_logger(__name__)
__author__ = "Aitor González-Marfil"
__author_github__ = "AAitorG"

register_plotly_display_defaults()
