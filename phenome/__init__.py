"""PhenoMe Package

A modular, **dataset-agnostic** and **model-agnostic** pipeline for phenotyping
analysis using deep learning embeddings.

Basic usage:

    from phenome import PhenoMe, load_dinov2_model

    model, wrapper = load_dinov2_model()
    pheno = PhenoMe()  # pass seed=42 for reproducibility
    file_df = pheno.find_files("path/to/images")
    pheno.process_images(wrapper)

    # Property presets: "none", "basic", "regionprops", "intensity", "full", "full_extended"
    df = pheno.compute_properties(property_preset="basic")  # or additional_property_functions={...}

    # Distances can be computed on embeddings, phenotypic properties, or a
    # combined normalized feature space (embeddings + properties)
    results = pheno.compute_reference_distances(
        reference_filters={'condition': 'Control'},
        mode='centroid',
        source='embeddings'  # or 'properties' or 'combined'
    )
    pheno.save_results(output_dir="results")  # Directory created automatically

    # Generate interactive report
    pheno.generate_report(output_path="report.html")

    # Interactive visualization in Jupyter (no need to pass pheno):
    # explorer = pheno.create_interactive_explorer(filters={...}, hover_features=[...])

Top-level imports: PhenoMe, load_dinov2_model, default_metadata_from_path,
  get_metadata_from_path, make_dataframe_metadata_fn, create_regionprops_function,
  create_intensity_function, create_masked_intensity_function,
  create_concentric_ring_function, create_texture_function, create_blur_effect_function,
  create_entropy_function, get_preset_property_functions.

For more customization, import from submodules:
- ``phenome.utils``: ModelWrapper, DinoV2ModelWrapper, get_default_device,
  set_determinism, TransformBuilder, and the above metadata/property functions
- ``phenome.core``: filter_indices, build_metadata_columns, build_export_dataframe,
  run_dimensionality_reduction, compute_pearson_correlation, compute_spearman_correlation
- ``phenome.io``: read_image, ensure_hwc, CheckpointManager, FileDiscovery
- ``phenome.mixins``: PhenoMeDataset, collate_fn, EmbeddingExtractor,
  PhenoMeProperties, PhenoMeDistances, PhenoMeAnalysis, PhenoMeVisualization,
  PhenoMeInteractive, create_interactive_explorer
- ``phenome.mixins.interactive``: create_interactive_explorer, PhenoMeInteractive
- ``phenome.plugins``: get_blob_properties, register_property, get_property

Model Compatibility:
    - DINOv2: load_dinov2_model() or DinoV2ModelWrapper
    - Custom: subclass ModelWrapper, implement _get_embeddings() -> (B, D)

Logging
-------
Progress messages (INFO level) are visible by default in both Jupyter
notebooks and scripts.  To silence them::

    import logging
    logging.getLogger("phenome").setLevel(logging.WARNING)
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
    __version__ = "1.1.0"

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
    "get_preset_property_functions",
    # Default models
    "load_dinov2_model",
    "make_dataframe_metadata_fn",
]

_get_logger(__name__)
__author__ = "Aitor González-Marfil"
__author_github__ = "AAitorG"

register_plotly_display_defaults()
