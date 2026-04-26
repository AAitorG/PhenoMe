"""
Visualization for the phenotyping pipeline.
Provides basic plotting and visualization methods.
"""

from typing import Any, cast

import pandas as pd
import plotly.express as px

from ..._logging import get_logger
from ...core import build_metadata_columns, get_all_metadata_keys
from ...core.pipeline_results import PhenoMeResults
from ._component_correlation_plots import _ComponentCorrelationPlotsMixin
from ._distance_plots import _DistancePlotsMixin
from ._dr_plots import _DRPlotsMixin
from ._group_enrichment_plots import _GroupEnrichmentPlotsMixin
from ._image_display import _ImageDisplayMixin
from ._interpretability_plots import _InterpretabilityPlotsMixin

logger = get_logger(__name__)


class PhenoMeVisualization(
    _DRPlotsMixin,
    _DistancePlotsMixin,
    _ImageDisplayMixin,
    _InterpretabilityPlotsMixin,
    _ComponentCorrelationPlotsMixin,
    _GroupEnrichmentPlotsMixin,
):
    """
    Pipeline class providing visualization methods for PhenoMe.

    Composed from _DRPlotsMixin (PCA, t-SNE, UMAP, centroids), _DistancePlotsMixin
    (distance distribution, correlation plots), _ImageDisplayMixin (image display),
    _InterpretabilityPlotsMixin (multivariate interpretability), and
    _ComponentCorrelationPlotsMixin (component-property correlation facets), and
    _GroupEnrichmentPlotsMixin (group Z-score enrichment facets).
    """

    # Type hints for pipeline attributes (provided by parent class)
    mean: tuple
    std: tuple
    results: PhenoMeResults
    image_transforms: object | None
    device: object

    def create_interactive_explorer(
        self,
        filters: dict | None = None,
        exclude: dict | None = None,
        hover_features: list[str] | None = None,
    ) -> Any:
        """Launch the interactive explorer for this pipeline's results."""
        from ..interactive import (
            _InteractiveExplorerProtocol,
        )
        from ..interactive import (
            create_interactive_explorer as _launch_explorer,
        )

        return _launch_explorer(
            cast(_InteractiveExplorerProtocol, self),
            filters=filters,
            exclude=exclude,
            hover_features=hover_features,
        )

    def plot_counts(self, group_by: list[str] | None = None, return_fig: bool = False) -> Any:
        """Plot count of images grouped by metadata using Plotly."""
        if not group_by:
            available = get_all_metadata_keys(self.results)
            raise ValueError(f"'group_by' must be non-empty. Available: {', '.join(available)}")

        group_by_cap = [k.capitalize() for k in group_by]
        df = pd.DataFrame(build_metadata_columns(self.results, capitalize=True, keys=group_by_cap))
        counts = df.groupby(group_by_cap, observed=True).size().reset_index(name="Count")

        if len(group_by) == 1:
            fig = px.bar(
                counts, x=group_by_cap[0], y="Count", title=f"Image Counts per {group_by_cap[0]}"
            )
        elif len(group_by) == 2:
            fig = px.bar(
                counts,
                x=group_by_cap[1],
                y="Count",
                color=group_by_cap[0],
                barmode="group",
                title=f"Image Counts per {group_by_cap[0]} and {group_by_cap[1]}",
            )
        else:
            logger.warning(
                "Grouping by %d dimensions not supported. Use 1 or 2 keys.", len(group_by)
            )
            return None

        if return_fig:
            return fig
        fig.show()
        return None
