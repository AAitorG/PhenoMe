"""
Protocol definitions for pipeline mixin dependencies.

Use these for type hints in mixins to clarify required attributes and methods
from the parent PhenoMe without affecting runtime behavior.
Import and use in mixins: def foo(self: PhenoMeProtocol) -> ... for
IDE/type-checking only.
"""

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from .pipeline_results import PhenoMeResults


class PhenoMeProtocol(Protocol):
    """Protocol describing attributes and methods mixins expect from PhenoMe.

    Use for type hints in mixins; does not affect runtime.
    """

    # Core state
    results: "PhenoMeResults"
    device: Any
    mean: tuple
    std: tuple
    test_transforms: Any | None
    preprocessing_fn: Any | None
    _processing_params: dict[str, Any] | None
    seed: int | None
    use_gpu_for_dr: bool

    # Embedding access
    def get_embeddings(self, indices: list[int] | np.ndarray | None = None) -> np.ndarray | None:
        """Return embedding rows for ``indices`` (or all committed rows)."""
        ...

    def _normalize_embeddings_l2(self, data: np.ndarray) -> np.ndarray:
        """L2-normalize embedding rows (used for cosine-style distance)."""
        ...

    @property
    def has_embeddings(self) -> bool:
        """Whether the pipeline has embedding vectors available."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension *D*."""
        ...

    # Property / analysis (from PhenoMeProperties, PhenoMeAnalysis)
    def _get_property_matrix(
        self,
        indices: list[int] | None = None,
        property_keys: list[str] | None = None,
        normalize: bool = True,
        handle_nans: str = "filter",
    ) -> tuple:
        """Return (matrix, valid indices, column keys) for property features."""
        ...

    def get_image_info(self, idx: int, distance_results: dict | None = None) -> dict:
        """Return a summary dict for image ``idx`` (paths, metadata, optional distances)."""
        ...

    def get_available_property_keys(self) -> list[str]:
        """Return available scalar property column names."""
        ...

    def _require_file_df(self, method_name: str = "method") -> "pd.DataFrame":
        """Return the on-disk ``file_df`` or raise if missing."""
        ...
