"""
Report context for section generators.

ReportContext holds the pipeline reference, report options, and pre-computed
results. Sections receive a single context object instead of many parameters,
reducing coupling and simplifying section signatures.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.pipeline_results import PhenoMeResults
    from ..pipeline import PhenoMe


@dataclass
class ReportContext:
    """Context passed to report section generators.

    Holds pipeline reference (for compute_* methods and embeddings), report
    options (opts), and pre-computed results (dist_results, etc.) shared
    across sections.
    """

    pipeline: "PhenoMe"
    opts: dict[str, Any] = field(default_factory=dict)

    # Pre-computed results (filled by generator, used by multiple sections)
    dist_results: dict[str, Any] | None = None

    # Cached metadata (from pipeline)
    n_images: int = 0
    metadata_keys: list[str] = field(default_factory=list)
    property_keys: list[str] = field(default_factory=list)
    has_embeddings: bool = False
    has_properties: bool = False
    embedding_dim: int = 0

    @property
    def results(self) -> "PhenoMeResults":
        """Pipeline results container."""
        return self.pipeline.results

    def get_embeddings(
        self,
        indices: list[int] | np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Get embeddings (lazy-safe)."""
        return self.pipeline.get_embeddings(indices=indices)
