"""Component-property correlation plots (faceted horizontal bar charts)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from ..._logging import get_logger

logger = get_logger(__name__)


def _log_component_correlation_results(
    summary_df: pd.DataFrame,
    component_order: list[str],
) -> None:
    """Log a plain-text summary of component-property correlations to the tool logger."""
    if summary_df.empty:
        logger.warning("No component correlation summary to log.")
        return

    banner_w = 72
    logger.info("%s", "=" * banner_w)
    logger.info("Component-property correlations")
    logger.info("%s", "=" * banner_w)

    for comp in component_order:
        sub = summary_df[summary_df["Component"] == comp].sort_values(
            "AbsCorrelation", ascending=False
        )
        if sub.empty:
            continue
        view = sub[["Property", "Correlation"]].copy()
        prop_w = max(
            len("Property"),
            int(view["Property"].astype(str).str.len().max()),
        )
        logger.info("")
        logger.info("Top properties correlated with %s:", comp)
        logger.info("")
        logger.info("  %-*s    %s", prop_w, "Property", "Correlation")
        logger.info("  %s", "-" * (prop_w + 4 + len("Correlation")))
        for _, row in view.iterrows():
            logger.info(
                "  %-*s    %+.4f",
                prop_w,
                str(row["Property"]),
                float(row["Correlation"]),
            )


def _build_component_correlation_figure(
    correlation_df: pd.DataFrame,
    component_names: list[str],
    top_k: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
) -> Figure | None:
    """Build a faceted horizontal bar chart: one row per component, bars ordered by |correlation|."""
    if correlation_df is None or correlation_df.empty:
        logger.warning("No correlation data to plot.")
        return None

    active: list[str] = []
    prepared: list[pd.Series] = []
    for comp in component_names:
        if comp not in correlation_df.columns:
            continue
        s = correlation_df[comp].dropna()
        if s.empty:
            continue
        s = s.iloc[s.abs().argsort()[::-1]]
        if top_k is not None:
            s = s.head(int(top_k))
        # Ascending |r| so the largest |r| appears at the top of the horizontal bar chart
        s_display = s.iloc[s.abs().argsort()]
        active.append(comp)
        prepared.append(s_display)

    if not prepared:
        logger.warning("No plottable component correlations.")
        return None

    w_max = 0.0
    for s_display in prepared:
        if len(s_display):
            w_max = max(w_max, float(s_display.abs().max()))
    w_max = w_max or 1e-9

    n_comp = len(active)
    max_bars = max(len(s) for s in prepared)
    row_px = max(160, 26 * max_bars + 50)
    gap_px = 40
    margin_t = 80 if title else 60
    margin_b = 56
    top_bottom_px = margin_t + margin_b
    height_px = max(280, int(row_px * n_comp + gap_px * max(0, n_comp - 1) + top_bottom_px))
    vertical_spacing = (gap_px / height_px) if n_comp > 1 else 0.02
    fig = make_subplots(
        rows=n_comp,
        cols=1,
        subplot_titles=active,
        vertical_spacing=vertical_spacing,
        shared_xaxes=True,
    )

    for i, s_display in enumerate(prepared, start=1):
        show_scale = i == n_comp
        marker: dict[str, Any] = {
            "color": s_display.values,
            "colorscale": "RdBu_r",
            "cmin": -w_max,
            "cmax": w_max,
            "showscale": show_scale,
        }
        if show_scale:
            # Anchor in full-figure paper coords so the bar spans the entire plot height
            # (not just the bottom subplot's trace box).
            marker["colorbar"] = {
                "title": "Correlation",
                "tickformat": ".2f",
                "yref": "paper",
                "y": 0.5,
                "yanchor": "middle",
                "len": 1.0,
                "lenmode": "fraction",
            }
        fig.add_trace(
            go.Bar(
                x=s_display.values,
                y=s_display.index.astype(str),
                orientation="h",
                marker=marker,
                name=active[i - 1],
                showlegend=False,
            ),
            row=i,
            col=1,
        )

    width_px = int(figsize[0] * 100)

    # Reserve left space from longest category label; fractional paper x does not track label
    # width, so we size margin in px and place the shared title with xshift in px.
    max_label_chars = max(
        (len(str(lbl)) for s in prepared for lbl in s.index.astype(str)),
        default=8,
    )
    # Tuned for a compact left strip: prior constants over-reserved space and pushed the
    # title too far left. Slightly raise char width again if long labels overlap the title.
    approx_char_px = 6.75
    tick_label_reserve_px = int(10 + approx_char_px * max_label_chars)
    title_gutter_px = 34
    # Room for |xshift| plus a little pad for the rotated word; cap avoids huge figures.
    margin_l = min(520, max(88, tick_label_reserve_px + title_gutter_px + 14))
    # Extra right margin so a full-height paper-anchored colorbar is not clipped
    margin_r = max(40, 36 + 14 * n_comp)

    fig.update_layout(
        title=title or "Component-property correlations",
        width=width_px,
        height=height_px,
        margin={"l": margin_l, "r": margin_r, "t": 80 if title else 60, "b": 56},
        showlegend=False,
    )
    # Manual horizontal space: automargin fights a stable slot for the shared y-title.
    fig.update_yaxes(automargin=False)
    # Anchor at the left edge of the plotting area (paper x=0); shift left so the title
    # sits between the figure edge and the tick labels (not centered as deep as before).
    title_center_shift_px = -(tick_label_reserve_px + title_gutter_px // 3)
    fig.add_annotation(
        text="<b>Property</b>",
        xref="paper",
        yref="paper",
        x=0,
        y=0.5,
        xshift=title_center_shift_px,
        showarrow=False,
        textangle=-90,
        xanchor="center",
        yanchor="middle",
        font={"size": 14, "color": "#1a1a1a", "family": "Arial, sans-serif"},
    )
    fig.update_xaxes(range=[-w_max * 1.08, w_max * 1.08])
    for r in range(1, n_comp):
        fig.update_xaxes(title_text="", row=r, col=1)
    fig.update_xaxes(title_text="Correlation", row=n_comp, col=1)

    return fig


class _ComponentCorrelationPlotsMixin:
    """Mixin for faceted component-correlation bar charts (internal visualization hooks)."""

    def _plot_component_correlation(
        self,
        correlation_df: pd.DataFrame,
        summary: pd.DataFrame,
        component_names: list[str],
        plot: bool = True,
        return_fig: bool = False,
        top_k: int | None = None,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
    ) -> Any:
        """Visualize or summarize component-property correlations (internal API).

        Mirrors the branching pattern used by :meth:`_DistancePlotsMixin._plot_distance_distribution`.

        Args:
            correlation_df: Properties x components correlation matrix.
            summary: Per-component top correlations (used for text logging).
            component_names: Ordered component column labels.
            plot: If True, build/show faceted bar charts when applicable.
            return_fig: If True, return the Plotly figure and do not call ``fig.show()``.
            top_k: Max properties per component facet (``None`` = all in *correlation_df*).
            figsize: Figure size in inches; width/height drive layout pixels.
            title: Optional figure title.

        Returns:
            The Plotly figure if one was built and *return_fig* is True; otherwise ``None``.
        """
        if not plot:
            _log_component_correlation_results(summary, component_order=component_names)
            if not return_fig:
                return None

        fig = _build_component_correlation_figure(
            correlation_df,
            component_names=component_names,
            top_k=top_k,
            figsize=figsize,
            title=title,
        )

        if fig is None:
            return None

        from ._helpers import handle_figure_output

        return handle_figure_output(fig, plot, return_fig)
