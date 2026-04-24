"""Distance and correlation plot methods for PhenoMeVisualization."""

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..._logging import get_logger
from ...core import build_metadata_columns, get_all_metadata_keys

logger = get_logger(__name__)


def _plot_property_correlations_plotly(
    plot_df: pd.DataFrame,
    *,
    order_by: str,
    title: str,
    figsize: tuple[int, int] = (10, 8),
    correlation_method: str | None = None,
) -> go.Figure:
    """Build a horizontal bar chart: mean of |r| per property with std error bars.

    This is a private visualization helper; callers are typically analysis helpers or mixins.
    """
    if plot_df.empty or "mean_abs" not in plot_df.columns or "std" not in plot_df.columns:
        if not plot_df.empty:
            logger.warning(
                "Property correlation frame must include 'mean_abs' and 'std'; cannot plot."
            )
        return go.Figure()

    w_px, h_px = figsize[0] * 100, figsize[1] * 100
    y_labels = plot_df["property"].astype(str).tolist()

    x_vals = np.asarray(plot_df["mean_abs"].to_numpy(), dtype=float)
    err = np.asarray(plot_df["std"].to_numpy(), dtype=float)
    err = np.where(np.isfinite(err), err, 0.0)
    x_vals = np.where(np.isfinite(x_vals), x_vals, np.nan)
    method = correlation_method or "n/a"
    # Concise description: bars = mean(|r|), error bars = SD(r), with method.
    xaxis_title = f"mean(|r|) ± SD(r) per dimension [{method}]"

    fig = go.Figure(
        data=[
            go.Bar(
                y=y_labels,
                x=x_vals,
                orientation="h",
                error_x={
                    "type": "data",
                    "array": err,
                    "visible": True,
                },
                marker_color="#3b82f6",
            )
        ],
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        title_font={"size": 12},
    )

    order_note = f" (ordered by {order_by})"
    fig.update_layout(
        title=f"{title}{order_note}" if order_by else title,
        yaxis_title="Property",
        width=w_px,
        height=h_px,
        margin={"l": 200, "r": 40, "t": 60, "b": 40},
    )
    return fig


class _DistancePlotsMixin:
    """Mixin providing distance distribution and correlation plots."""

    results: object  # PhenoMeResults, provided by parent

    @staticmethod
    def _log_distance_group_stats(group_col: str, df: pd.DataFrame) -> None:
        """Log per-group distance summary to the tool logger (plain text)."""
        logger.info("DISTANCE SUMMARY STATISTICS")
        stats = df.groupby(group_col)["Distance"].agg(["mean", "std", "min", "max", "count"])
        logger.info("\n%s", stats.to_string())

    def _prepare_distance_distribution_frame(
        self,
        distance_results: dict,
        group_by: str | list[str],
        dist_range: tuple,
    ) -> tuple[pd.DataFrame, str, str] | None:
        """Build a filtered DataFrame and grouping column for distance distribution views.

        Returns:
            (df, group_col, title_suffix) or None if there is no plottable data.
        """
        if "distances" not in distance_results:
            logger.warning("No distance data available.")
            return None

        group_keys = [group_by] if isinstance(group_by, str) else list(group_by)
        group_keys_cap = [k.capitalize() for k in group_keys]

        df = pd.DataFrame(
            build_metadata_columns(self.results, capitalize=True, keys=group_keys_cap)
        )
        df["Distance"] = distance_results["distances"]
        df = df.dropna(subset=["Distance"])

        if len(group_keys_cap) == 1:
            group_col = group_keys_cap[0]
            title_suffix = group_col
        else:
            group_col = "Group"
            title_suffix = " | ".join(group_keys_cap)
            df[group_col] = df[group_keys_cap].astype(str).agg(" | ".join, axis=1)

        df = df[(df["Distance"] >= dist_range[0]) & (df["Distance"] <= dist_range[1])]
        if df.empty:
            logger.warning("No distances in range %s.", dist_range)
            return None

        return df, group_col, title_suffix

    def _plot_distance_distribution(
        self,
        distance_results: dict,
        group_by: str | list[str],
        dist_range: tuple = (0, 100),
        figsize: tuple[int, int] = (10, 6),
        plot: bool = True,
        return_fig: bool = False,
        points: Literal["all", "outliers", False] | None = None,
    ) -> Any:
        """Visualize or summarize distance results by metadata (internal API).

        When *plot* is True, show a violin plot (or return it) without logging a text
        summary. When *plot* is False, log per-group summary statistics via
        :func:`_log_distance_group_stats` where applicable (e.g. text-only or
        *return_fig* with *plot* false).

        Args:
            distance_results: Dict containing a ``"distances"`` array.
            group_by: Metadata key(s) to group by.
            dist_range: (min, max) distance range to keep.
            figsize: Figure size in inches (width, height); scaled for Plotly.
            plot: If True, build/show a violin plot. If False, text summary only.
            return_fig: If True, return the Plotly figure and do not call ``fig.show()``.
            points: Violin point overlay: ``'all'``, ``'outliers'``, or False; None auto-selects.

        Returns:
            The Plotly figure if *return_fig* and a figure was built, else None.
        """
        prepared = self._prepare_distance_distribution_frame(distance_results, group_by, dist_range)
        if prepared is None:
            return None

        df, group_col, title_suffix = prepared

        # Text-only, no figure
        if not plot and not return_fig:
            self._log_distance_group_stats(group_col, df)
            return None

        if points is None:
            points = "outliers" if len(df) > 5000 else "all"

        fig = px.violin(
            df,
            x="Distance",
            y=group_col,
            color=group_col,
            points=points,
            title=f"Distance Distribution by {title_suffix}",
        )
        fig.update_layout(width=figsize[0] * 100, height=figsize[1] * 100)

        if not plot and return_fig:
            self._log_distance_group_stats(group_col, df)
            return fig

        if plot and return_fig:
            return fig

        if plot and not return_fig:
            fig.show()
            return None

        return None

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

        prepared = self._prepare_distance_distribution_frame(distance_results, group_by, dist_range)
        if prepared is None:
            return
        df, group_col, _ = prepared
        self._log_distance_group_stats(group_col, df)
