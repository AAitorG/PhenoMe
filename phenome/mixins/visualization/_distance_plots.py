"""Distance and correlation plot methods for PhenoMeVisualization."""

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..._logging import get_logger
from ...core import build_metadata_columns, get_all_metadata_keys
from ...utils.display_names import capitalize_preserve, format_correlation_method

logger = get_logger(__name__)


def _plot_property_correlations_plotly(
    plot_df: pd.DataFrame,
    *,
    correlations: dict[str, np.ndarray],
    order_by: str,
    title: str,
    figsize: tuple[int, int] = (10, 8),
    correlation_method: str | None = None,
) -> go.Figure:
    """Build horizontal violin plots: distribution of |r| across dimensions per property.

    This is a private visualization helper; callers are typically analysis helpers or mixins.
    """
    if plot_df.empty or not correlations:
        if not plot_df.empty and not correlations:
            logger.warning(
                "Property correlation plot needs non-empty correlations dict; cannot plot."
            )
        return go.Figure()

    rows: list[dict[str, Any]] = []
    y_ordered: list[str] = []
    for pname in plot_df["property"].astype(str):
        r = correlations.get(pname)
        if r is None:
            continue
        vc = r[np.isfinite(r)]
        if len(vc) == 0:
            continue
        display = capitalize_preserve(pname)
        y_ordered.append(display)
        for v in np.abs(vc):
            rows.append({"property": display, "abs_correlation": float(v)})

    if not rows:
        logger.warning("No finite correlation values to plot.")
        return go.Figure()

    long_df = pd.DataFrame(rows)
    method_display = (
        "n/a" if not correlation_method else format_correlation_method(correlation_method)
    )
    xaxis_title = f"Distribution of |r| across dimensions [{method_display}]"

    fig = px.violin(
        long_df,
        x="abs_correlation",
        y="property",
        color="property",
        orientation="h",
        box=True,
        points=False,
        category_orders={"property": y_ordered},
    )
    # Violin: light semi-transparent blue
    # Box: white fill with dark border for contrast
    fig.update_traces(
        fillcolor="rgba(219, 234, 254, 0.6)",  # Light blue, semi-transparent
        line={"color": "#1e40af"},  # Dark blue outline
        opacity=1.0,
        showlegend=False,
    )
    # Box plot styling for better contrast
    fig.for_each_trace(
        lambda t: t.update(
            marker={"color": "#1e40af", "size": 4},
            meanline={"color": "#0369a1", "width": 2},
            box={
                "fillcolor": "rgba(255, 255, 255, 0.8)",
                "line": {"color": "#1e40af", "width": 1.5},
            },
        )
    )

    w_px, h_px = figsize[0] * 100, figsize[1] * 100
    order_note = f" (ordered by {order_by})"
    fig.update_layout(
        title=f"{title}{order_note}" if order_by else title,
        xaxis_title=xaxis_title,
        yaxis_title="Property",
        width=w_px,
        height=h_px,
        margin={"l": 200, "r": 40, "t": 60, "b": 40},
    )
    fig.update_xaxes(title_font={"size": 12})
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
        distance_results: pd.DataFrame,
        group_by: str | list[str],
        dist_range: tuple,
    ) -> tuple[pd.DataFrame, str, str] | None:
        """Build a filtered DataFrame and grouping column for distance distribution views.

        Returns:
            (df, group_col, title_suffix) or None if there is no plottable data.
        """
        if distance_results is None or "distance" not in distance_results.columns:
            logger.warning("No distance data available.")
            return None

        group_keys = [group_by] if isinstance(group_by, str) else list(group_by)

        df = pd.DataFrame(build_metadata_columns(self.results, capitalize=True, keys=group_keys))
        group_keys_cap = list(df.columns)
        df["Distance"] = distance_results["distance"].values
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
        distance_results: pd.DataFrame,
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
            distance_results: DataFrame containing a ``"distance"`` column.
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

        if not plot:
            self._log_distance_group_stats(group_col, df)
            if not return_fig:
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

        from ._helpers import handle_figure_output

        return handle_figure_output(fig, plot, return_fig)

    def print_distance_summary(
        self,
        distance_results: pd.DataFrame,
        group_by: str | None = None,
        dist_range: tuple = (0, 100),
    ) -> None:
        """
        Print distance summary statistics without plotting. Safe to use when enable_plots=False.

        Args:
            distance_results: DataFrame containing 'distance' column
            group_by: Metadata key to group by
            dist_range: Tuple of (min, max) distances to include
        """
        if distance_results is None or "distance" not in distance_results.columns:
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


def _log_embedding_property_correlation_summary(
    summary: pd.DataFrame, order_by: str, top_k: int | None
) -> None:
    """Log a plain-text table of the top property correlation rows."""
    want = ("property", "mean_abs", "std", "max_abs", "min_abs")
    columns = [c for c in want if c in summary.columns]
    if not columns or "property" not in columns:
        logger.info("Property correlation summary is empty; nothing to log.")
        return
    view = summary[columns].copy()
    if top_k is not None:
        view = view.head(int(top_k))
    n = len(view)
    top_note = f" (showing {n} of {len(summary)} properties)" if n < len(summary) else ""
    sep = "─" * 88
    num_cols = [c for c in columns if c not in ("property",)]

    def _fmt_num_cell(v: object) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return "—"
        if not np.isfinite(x):
            return "—"
        return f"{x:.4f}"

    out = view.copy()
    out["property"] = out["property"].astype(str).map(capitalize_preserve)
    for c in num_cols:
        out[c] = out[c].map(_fmt_num_cell)
    block = out.to_string(index=False, col_space=2)
    logger.info(
        "%s\nEmbedding-property correlations  (ordered by %s)%s\n%s\n%s\n%s",
        sep,
        order_by,
        top_note,
        sep,
        block,
        sep,
    )
