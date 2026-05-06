"""Group enrichment plots (faceted horizontal bar charts by Z-score)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from ..._logging import get_logger

logger = get_logger(__name__)


def _log_group_enrichment_results(
    enrichment_df: pd.DataFrame,
    group_order: list[Any],
    top_k: int | None = None,
    fixed_decimals: int = 4,
) -> None:
    """Log group enrichment as tabular plain text (column headers once; one block per group).

    When *top_k* is set, only the top *top_k* rows per group by absolute Z-score are logged.
    """
    if enrichment_df.empty:
        logger.warning("No group enrichment summary to log.")
        return

    banner_w = 72
    logger.info("%s", "=" * banner_w)
    logger.info("Group Enrichment Analysis (Z-scores)")
    logger.info("%s", "=" * banner_w)

    num_w = max(12, fixed_decimals + 6)  # room for header labels and signed .4f
    fmt = f"{{:{num_w}.{fixed_decimals}f}}"
    z_fmt = f"{{:+{num_w}.{fixed_decimals}f}}"

    # Column headers once (printed a single time above all groups)
    props = enrichment_df["Property"].astype(str)
    prop_w = max(len("Property"), int(props.str.len().max()))
    h1, h2, h3 = "Group_mean", "Z", "Pop_mean"
    w1 = max(len(h1), num_w)
    w2 = max(len(h2), num_w)
    w3 = max(len(h3), num_w)
    logger.info("")
    logger.info(
        "  %-*s  %*s  %*s  %*s",
        prop_w,
        "Property",
        w1,
        h1,
        w2,
        h2,
        w3,
        h3,
    )
    logger.info("  %s", "-" * (prop_w + w1 + w2 + w3 + 12))

    for grp in group_order:
        sub = enrichment_df[enrichment_df["Group"] == grp]
        if sub.empty:
            continue
        sub = sub.sort_values("AbsScore", ascending=False)
        if top_k is not None:
            sub = sub.head(int(top_k))
        logger.info("")
        logger.info("Group: %s", grp)
        for _, row in sub.iterrows():
            gm = float(row["Mean_Group"])
            z = float(row["Score"])
            pm = float(row["Mean_Pop"])
            logger.info(
                "  %-*s  %*s  %*s  %*s",
                prop_w,
                str(row["Property"]),
                w1,
                fmt.format(gm),
                w2,
                z_fmt.format(z),
                w3,
                fmt.format(pm),
            )


def _build_group_enrichment_figure(
    enrichment_df: pd.DataFrame,
    group_names: list[Any],
    top_k: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
) -> Figure | None:
    """Build faceted horizontal bar charts: one row per group, bars = Z-score, top-|Z| only."""
    if enrichment_df is None or enrichment_df.empty:
        logger.warning("No group enrichment data to plot.")
        return None

    active: list[Any] = []
    prepared: list[pd.Series] = []
    for grp in group_names:
        sub = enrichment_df[enrichment_df["Group"] == grp]
        if sub.empty:
            continue
        sub = sub.sort_values("AbsScore", ascending=False)
        if top_k is not None:
            sub = sub.head(int(top_k))
        s = sub.set_index("Property")["Score"].dropna()
        if s.empty:
            continue
        s_display = s.iloc[s.abs().argsort()]
        active.append(grp)
        prepared.append(s_display)

    if not prepared:
        logger.warning("No plottable group enrichment rows.")
        return None

    n_grp = len(active)
    max_bars = max(len(s) for s in prepared)
    row_px = max(160, 26 * max_bars + 50)
    gap_px = 80
    margin_t = 80 if title else 60
    margin_b = 64
    top_bottom_px = margin_t + margin_b
    height_px = max(300, int(row_px * n_grp + gap_px * max(0, n_grp - 1) + top_bottom_px))
    vertical_spacing = (gap_px / height_px) if n_grp > 1 else 0.02
    fig = make_subplots(
        rows=n_grp,
        cols=1,
        subplot_titles=[str(g) for g in active],
        vertical_spacing=vertical_spacing,
        # Independent x-axes and per-row color limits so one high-|Z| group does not wash out others.
        shared_xaxes=False,
    )

    for i, s_display in enumerate(prepared, start=1):
        w_row = float(s_display.abs().max()) if len(s_display) else 0.0
        w_row = w_row if w_row >= 1e-9 else 1e-9
        marker: dict[str, Any] = {
            "color": s_display.values,
            "colorscale": "RdBu_r",
            "cmin": -w_row,
            "cmax": w_row,
            "showscale": False,
        }
        fig.add_trace(
            go.Bar(
                x=s_display.values,
                y=s_display.index.astype(str),
                orientation="h",
                marker=marker,
                name=str(active[i - 1]),
                showlegend=False,
            ),
            row=i,
            col=1,
        )

    width_px = int(figsize[0] * 100)

    max_label_chars = max(
        (len(str(lbl)) for s in prepared for lbl in s.index.astype(str)),
        default=8,
    )
    approx_char_px = 6.75
    tick_label_reserve_px = int(10 + approx_char_px * max_label_chars)
    title_gutter_px = 34
    margin_l = min(520, max(88, tick_label_reserve_px + title_gutter_px + 14))

    fig.update_layout(
        title=title or "Group enrichment (Z-scores)",
        width=width_px,
        height=height_px,
        margin={"l": margin_l, "r": 24, "t": 80 if title else 60, "b": 64},
        showlegend=False,
    )
    fig.update_yaxes(automargin=False)
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
    for r, s_display in enumerate(prepared, start=1):
        w_row = float(s_display.abs().max()) if len(s_display) else 1e-9
        w_row = w_row if w_row >= 1e-9 else 1e-9
        fig.update_xaxes(range=[-w_row * 1.08, w_row * 1.08], row=r, col=1)
    for r in range(1, n_grp):
        fig.update_xaxes(title_text="", row=r, col=1)
    fig.update_xaxes(title_text="Z-score (per row)", row=n_grp, col=1)

    return fig


class _GroupEnrichmentPlotsMixin:
    """Mixin for group-enrichment bar charts (internal visualization hooks)."""

    def _plot_group_enrichment(
        self,
        enrichment_df: pd.DataFrame,
        group_names: list[Any] | None = None,
        plot: bool = True,
        return_fig: bool = False,
        top_k: int | None = None,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
    ) -> Any:
        """Visualize or summarize group-property Z-score enrichment (internal API).

        Mirrors the branching pattern used by :meth:`_ComponentCorrelationPlotsMixin._plot_component_correlation`.

        Args:
            enrichment_df: Rows per group-property with columns Group, Property, Score,
                Mean_Group, Mean_Pop, AbsScore.
            group_names: Order of groups for facets; default is sorted unique ``Group`` values.
            plot: If True, build/show faceted bar charts when applicable.
            return_fig: If True, return the Plotly figure and do not call ``fig.show()``.
            top_k: Max properties per group facet and per-group text block (``None`` = all rows).
            figsize: Figure size in inches; width/height drive layout pixels.
            title: Optional figure title.

        Returns:
            The Plotly figure if one was built and *return_fig* is True; otherwise ``None``.
        """
        if enrichment_df.empty:
            if not plot and not return_fig:
                _log_group_enrichment_results(enrichment_df, group_order=[])
            else:
                logger.warning("No group enrichment data to plot.")
            return None

        if group_names is None:
            group_names = sorted(enrichment_df["Group"].unique().tolist())

        if not plot and not return_fig:
            _log_group_enrichment_results(enrichment_df, group_order=group_names, top_k=top_k)
            return None

        fig = _build_group_enrichment_figure(
            enrichment_df,
            group_names=group_names,
            top_k=top_k,
            figsize=figsize,
            title=title,
        )
        if fig is None:
            if plot or return_fig:
                _log_group_enrichment_results(enrichment_df, group_order=group_names, top_k=top_k)
            return None

        _log_group_enrichment_results(enrichment_df, group_order=group_names, top_k=top_k)

        from ._helpers import handle_figure_output

        return handle_figure_output(fig, plot, return_fig)
