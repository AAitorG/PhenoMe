"""Distance and correlation plot methods for PhenoMeVisualization."""

from typing import Any, Literal

import pandas as pd
import plotly.express as px

from ..._logging import get_logger
from ...core import build_metadata_columns, get_all_metadata_keys

logger = get_logger(__name__)


class _DistancePlotsMixin:
    """Mixin providing distance distribution and correlation plots."""

    results: object  # PhenoMeResults, provided by parent

    def print_distance_summary(
        self,
        distance_results: dict,
        group_by: str | None = None,
        dist_range: tuple = (0, 100),
    ) -> None:
        """
        Print distance summary statistics without plotting. Safe to use when enable_plots=False.

        Args:
            distance_results: Dictionary containing 'distances' array
            group_by: Metadata key to group by
            dist_range: Tuple of (min, max) distances to include
        """
        if "distances" not in distance_results:
            logger.warning("No distance data available.")
            return
        if group_by is None:
            available = get_all_metadata_keys(self.results)
            raise ValueError(f"'group_by' must be specified. Available: {', '.join(available)}")

        group_col = group_by.capitalize()
        df = pd.DataFrame(build_metadata_columns(self.results, capitalize=True, keys=[group_col]))
        df["Distance"] = distance_results["distances"]
        df = df.dropna(subset=["Distance"])
        df = df[(df["Distance"] >= dist_range[0]) & (df["Distance"] <= dist_range[1])]
        if df.empty:
            logger.warning("No distances in range %s.", dist_range)
            return

        logger.info("DISTANCE SUMMARY STATISTICS")
        stats = df.groupby(group_col)["Distance"].agg(["mean", "std", "min", "max", "count"])
        logger.info("\n%s", stats.to_string())

    def plot_distance_distribution(
        self,
        distance_results: dict,
        group_by: str | list[str] | None = None,
        dist_range: tuple = (0, 100),
        figsize: tuple[int, int] = (10, 6),
        return_fig: bool = False,
        points: Literal["all", "outliers", False] | None = None,
    ) -> Any:
        """
        Plot distance distribution grouped by metadata using Plotly violin plots.

        Args:
            distance_results: Dictionary containing 'distances' array
            group_by: Metadata key(s) to group by. Single string or list (e.g. ['drug', 'time']).
            dist_range: Tuple of (min, max) distances to include
            figsize: Figure size (width, height)
            return_fig: Whether to return the figure
            points: Which points to show on the violin plot ('all', 'outliers', or False).
                   If None, auto-selects 'outliers' for large datasets (>5000 points)
                   and 'all' for smaller ones.
        """
        if "distances" not in distance_results:
            logger.warning("No distance data available.")
            return
        if group_by is None:
            available = get_all_metadata_keys(self.results)
            raise ValueError(f"'group_by' must be specified. Available: {', '.join(available)}")

        group_keys = [group_by] if isinstance(group_by, str) else list(group_by)
        group_keys_cap = [k.capitalize() for k in group_keys]

        df = pd.DataFrame(
            build_metadata_columns(self.results, capitalize=True, keys=group_keys_cap)
        )
        df["Distance"] = distance_results["distances"]
        df = df.dropna(subset=["Distance"])

        if len(group_keys_cap) == 1:
            group_col = group_keys_cap[0]
        else:
            group_col = "_group"
            df[group_col] = df[group_keys_cap].astype(str).agg(" | ".join, axis=1)

        df = df[(df["Distance"] >= dist_range[0]) & (df["Distance"] <= dist_range[1])]
        if df.empty:
            logger.warning("No distances in range %s.", dist_range)
            return

        if points is None:
            points = "outliers" if len(df) > 5000 else "all"

        title_suffix = group_col if len(group_keys_cap) == 1 else " | ".join(group_keys_cap)
        fig = px.violin(
            df,
            x="Distance",
            y=group_col,
            color=group_col,
            points=points,
            title=f"Distance Distribution by {title_suffix}",
        )

        fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)

        if return_fig:
            return fig

        fig.show()

        logger.info("DISTANCE SUMMARY STATISTICS")
        stats = df.groupby(group_col)["Distance"].agg(["mean", "std", "min", "max", "count"])
        logger.info("\n%s", stats.to_string())

    def plot_property_correlations(
        self,
        correlation_results: dict,
        top_k: int = 20,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Property Correlations with Embeddings",
    ) -> Any:
        """
        Plot the top correlated properties as a horizontal bar chart.

        Args:
            correlation_results: Output from compute_embedding_property_correlation()
            top_k: Number of top properties to display
            figsize: Figure size (width, height) in inches (converted to pixels for Plotly)
            title: Plot title
        """
        if not correlation_results or "summary" not in correlation_results:
            logger.warning(
                "Invalid correlation results. Run compute_embedding_property_correlation() first."
            )
            return

        df = correlation_results["summary"]
        # Drop entries with NaN aggregated correlations to avoid plotting invalid values
        df = df.dropna(subset=["aggregated_correlation"])
        if df.empty:
            logger.warning("No correlation data to plot.")
            return

        plot_df = df.head(top_k).copy()
        plot_df = plot_df.sort_values("aggregated_correlation", ascending=True)

        fig = px.bar(
            plot_df,
            x="aggregated_correlation",
            y="property",
            orientation="h",
            title=title,
            labels={
                "aggregated_correlation": f"Aggregated Correlation ({correlation_results.get('aggregation_method', 'mean_abs')})",
                "property": "Property Name",
            },
        )

        fig.update_layout(
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            xaxis_title="Correlation Strength",
            yaxis_title="Property",
        )
        fig.show()
