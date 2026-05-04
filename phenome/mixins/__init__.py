"""
@section Mixins

Pipeline mixins for composable phenotyping functionality.

PhenoMe uses a mixin-based architecture where the main orchestration class (`PhenoMe`)
inherits from specialized feature-layer mixins. Each mixin encapsulates a specific
capability and is organized to keep the public API thin by delegating heavy logic
to internal modules (e.g., `_compute.py`, `_helpers.py`).

**Mixin classes:**
- `PhenoMeProperties`: Compute cell/object morphological and intensity properties.
- `PhenoMeAnalysis`: Statistical analysis (clustering, outlier detection, prototype selection).
- `PhenoMeDistances`: Compute reference distances and prototype distances.
- `PhenoMeVisualization`: Plotting and visualization (PCA, t-SNE, UMAP, heatmaps).
- `PhenoMeBatchCorrection`: Correct batch effects.
- `PhenoMeInteractive`: Interactive Jupyter explorer for results.

**Supporting classes:**
- `PhenoMeDataset`: PyTorch Dataset for image batching and embedding extraction.
- `EmbeddingExtractor`: Utility for orchestrating embedding extraction.
- `collate_fn`: Custom collate function for DataLoader.

**Composition in PhenoMe:**

```python
class PhenoMe(
    PhenoMeProperties,
    PhenoMeAnalysis,
    PhenoMeDistances,
    PhenoMeVisualization,
    PhenoMeBatchCorrection,
):
    # Inherits all mixin methods
    pass
```

Each mixin provides related methods as a logical group:
- `PhenoMeProperties`: `compute_properties()`, `property_stats_by_group()`, etc.
- `PhenoMeAnalysis`: `compute_clustering()`, `detect_outliers()`, `find_prototypes()`, etc.
- `PhenoMeDistances`: `compute_reference_distances()`, etc.
- `PhenoMeVisualization`: `plot_pca()`, `plot_tsne()`, `plot_umap()`, etc.
- `PhenoMeBatchCorrection`: `correct_batches()`, etc.
- `PhenoMeInteractive`: `create_interactive_explorer()`, etc.

**Internal organization:**
Most mixin modules follow this structure:
- `mixin_class.py`: Public mixin class with main methods.
- `_compute.py`: Heavy computation logic (non-public).
- `_helpers.py`: Helper functions (non-public).

This keeps the public namespace clean and makes internal implementation details easy to refactor.

**See Also:**
For architectural overview, see `phenome/ARCHITECTURE.md`.
For property factories and custom properties, see `phenome.plugins` and `phenome.utils`.
"""

from .analysis import PhenoMeAnalysis
from .batch_correction import PhenoMeBatchCorrection
from .dataset import PhenoMeDataset, collate_fn
from .distances import PhenoMeDistances
from .embedding_extractor import EmbeddingExtractor
from .interactive import PhenoMeInteractive, create_interactive_explorer
from .properties import PhenoMeProperties
from .visualization import PhenoMeVisualization

__all__ = [
    "EmbeddingExtractor",
    "PhenoMeAnalysis",
    "PhenoMeBatchCorrection",
    "PhenoMeDataset",
    "PhenoMeDistances",
    "PhenoMeInteractive",
    "PhenoMeProperties",
    "PhenoMeVisualization",
    "collate_fn",
    "create_interactive_explorer",
]
