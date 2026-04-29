"""Pipeline mixins for PhenoMe.

Provides dataset, embedding extraction, properties, distances, analysis,
visualization, and interactive capabilities. See the top-level
`phenome` module for usage.

Public API:
    - PhenoMeProperties: compute_properties, property_stats_by_group, etc.
    - PhenoMeDistances: compute_reference_distances
    - PhenoMeAnalysis: compute_clustering, detect_outliers, find_prototypes
    - PhenoMeVisualization: plot_pca, plot_tsne, plot_umap
    - PhenoMeInteractive, create_interactive_explorer: Interactive Jupyter explorer
    - PhenoMeDataset, collate_fn: Dataset for embedding extraction

For property factories (create_regionprops_function, etc.), import from
phenome or phenome.utils.

Mixin Composition Strategy:
    PhenoMe follows a mixin-based architecture where the main pipeline class
    inherits from specialized feature layers. Each mixin module is organized
    to keep the public API thin by delegating heavy orchestration and
    auxiliary logic to internal modules (e.g., `_compute.py`, `_helpers.py`).
"""

from .analysis import PhenoMeAnalysis
from .dataset import PhenoMeDataset, collate_fn
from .distances import PhenoMeDistances
from .embedding_extractor import EmbeddingExtractor
from .interactive import PhenoMeInteractive, create_interactive_explorer
from .properties import PhenoMeProperties
from .visualization import PhenoMeVisualization

__all__ = [
    "EmbeddingExtractor",
    "PhenoMeAnalysis",
    "PhenoMeDataset",
    "PhenoMeDistances",
    "PhenoMeInteractive",
    "PhenoMeProperties",
    "PhenoMeVisualization",
    "collate_fn",
    "create_interactive_explorer",
]
