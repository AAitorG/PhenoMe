"""Report section generators."""

from typing import Any, Protocol, runtime_checkable

from ..context import ReportContext
from .clustering import generate_clustering_section
from .correlation import generate_correlation_section
from .distance import generate_distance_section
from .export import generate_export_section
from .gallery import generate_image_gallery_section
from .interpretability import generate_interpretability_section
from .outlier import generate_outlier_section
from .overview import generate_overview_section
from .property_stats import generate_property_stats_section
from .run_settings import generate_run_settings_section
from .visualization import generate_visualization_section


@runtime_checkable
class ReportSection(Protocol):
    """Protocol for report section generator callables."""

    def __call__(self, ctx: ReportContext, **kwargs: Any) -> str: ...


__all__ = [
    "generate_clustering_section",
    "generate_correlation_section",
    "generate_distance_section",
    "generate_export_section",
    "generate_image_gallery_section",
    "generate_interpretability_section",
    "generate_outlier_section",
    "generate_overview_section",
    "generate_property_stats_section",
    "generate_run_settings_section",
    "generate_visualization_section",
]
