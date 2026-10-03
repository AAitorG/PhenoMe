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
    @section Visualization
    @order 8

    Pipeline class providing visualization methods for PhenoMe.

    **Module Breakdown:**
    - `_dr_plots.py`: Dimensionality reduction visualizations (PCA, t-SNE, UMAP) and centroid plots.
    - `_distance_plots.py`: Distance distributions, reference comparison plots, and property correlations.
    - `_image_display.py`: Raw and transformed image visualization with channel-wise controls.
    - `_interpretability_plots.py`: Visualizing feature importance and drivers for embedding axes.
    - `_component_correlation_plots.py`: Faceted plots of correlations between embeddings and properties.
    - `_group_enrichment_plots.py`: Faceted Z-score enrichment plots for metadata groups.
    - `_helpers.py`: Shared utilities for Plotly and Matplotlib layout/styling.

    **Interdependencies:**
    The following private plot functions are intended for use by `PhenoMeAnalysis`:
    - `_plot_property_correlations_plotly` (from `_distance_plots.py`): Used for embedding-property correlation analysis.
    - `_display_multivariate_interpretability` (from `_interpretability_plots.py`): Used for explaining embedding axes.
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
        """Launch the interactive explorer for this pipeline's results.

        Args:
            filters: Optional metadata filters applied before the first plot.
            exclude: Optional metadata exclusions (same structure as filters).
            hover_features: Optional hover tooltip keys. Defaults to metadata keys.
        """
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

    def create_colab_interactive_explorer(
        self,
        filters: dict | None = None,
        exclude: dict | None = None,
    ) -> Any:
        """Launch the short Colab explorer for this pipeline's results.

        Embedding and Appearance controls, an SVG scatter, and click-to-image.
        Use this in Google Colab. Jupyter notebooks should use
        ``create_interactive_explorer``.

        Args:
            filters: Optional metadata filters applied when Compute runs.
            exclude: Optional metadata exclusions (same structure as filters).
        """
        from ..interactive import (
            _InteractiveExplorerProtocol,
        )
        from ..interactive import (
            create_colab_interactive_explorer as _launch_colab_explorer,
        )

        return _launch_colab_explorer(
            cast(_InteractiveExplorerProtocol, self),
            filters=filters,
            exclude=exclude,
        )

    def plot_counts(
        self, group_by: list[str] | None = None, return_fig: bool = False, plot: bool = True
    ) -> Any:
        """Plot count of images grouped by metadata using Plotly.

        Args:
            group_by: List of metadata keys to group by.
            return_fig: If True, return the Plotly figure instead of displaying it.
            plot: If True (default), create and display a Plotly plot. If False, print results as text.
        """
        if not group_by:
            available = get_all_metadata_keys(self.results)
            raise ValueError(f"'group_by' must be non-empty. Available: {', '.join(available)}")

        df = pd.DataFrame(build_metadata_columns(self.results, capitalize=True, keys=group_by))
        group_by_cap = list(df.columns)
        counts = df.groupby(group_by_cap, observed=True).size().reset_index(name="Count")

        if not plot:
            total = counts["Count"].sum()
            if total == 0:
                print(f"\nNo images found for grouping: {', '.join(group_by_cap)}\n")
                return None

            max_count = counts["Count"].max()
            counts_display = counts.copy()

            # Add percentage and a simple text-based bar chart
            counts_display["%"] = (counts_display["Count"] / total * 100).round(1).astype(str) + "%"
            counts_display["Distribution"] = counts_display["Count"].apply(
                lambda x: "█" * int(x / max_count * 20)
            )

            # Calculate optimal column widths with extra spacing
            cols = list(counts_display.columns)
            widths = {}
            for col in cols:
                data_max = counts_display[col].astype(str).str.len().max()
                widths[col] = max(len(col), data_max) + 4  # 4 spaces between columns

            total_width = sum(widths.values())
            title = f"IMAGE COUNTS: {', '.join(group_by_cap).upper()}"

            # Print formatted table
            print(f"\n{'=' * total_width}")
            print(title.center(total_width))
            print(f"{'=' * total_width}")

            # Header
            header = "".join([col.ljust(widths[col]) for col in cols])
            print(header)
            print("-" * total_width)

            # Data rows
            for _, row in counts_display.iterrows():
                row_str = "".join([str(row[col]).ljust(widths[col]) for col in cols])
                print(row_str)

            print("-" * total_width)
            footer = f"TOTAL IMAGES: {total:,}"
            print(footer.center(total_width))
            print(f"{'=' * total_width}\n")
            return None

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
