"""Multivariate interpretability plot methods for PhenoMeVisualization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from ..._logging import get_logger
from ...utils.display_names import format_dr_method
from ._helpers import BAR_ZERO_LINE_COLOR, BAR_ZERO_LINE_WIDTH

logger = get_logger(__name__)


def _log_multivariate_interpretability_results(
    drivers_df: pd.DataFrame,
    meta: dict[str, Any],
    top_k: int,
) -> None:
    """Log a plain-text summary of multivariate interpretability to the tool logger."""
    if drivers_df.empty:
        logger.warning("No interpretability drivers to summarize.")
        return

    r2 = float(meta.get("r2", 0.0))
    drivers = drivers_df.head(top_k).to_dict("records")

    logger.info("On-axis fit (R2, descriptive): %.2f", r2)
    logger.info("Top drivers:")
    for driver in drivers:
        logger.info("  [%+.3f] %s", float(driver["weight"]), driver["feature"])


def _build_multivariate_interpretability_figure(
    drivers_df: pd.DataFrame,
    meta: dict[str, Any],
    top_k: int,
    figsize: tuple[int, int],
) -> Figure | None:
    """Build a horizontal bar chart of top drivers ordered by absolute magnitude (largest at top)."""
    if drivers_df.empty:
        logger.warning("No interpretability results to plot.")
        return None

    resolved_model_type = meta.get("model_type", "lasso")
    if resolved_model_type not in ("lasso", "random_forest"):
        resolved_model_type = "lasso"

    r2 = float(meta.get("r2", 0.0))
    target = meta.get("target_component", "")
    dr_method = format_dr_method(str(meta.get("method", "")))
    model_name = "LASSO" if resolved_model_type == "lasso" else "Random Forest"

    df = drivers_df.head(top_k).copy()
    df["_abs"] = df["weight"].abs()
    df = df.sort_values("_abs", ascending=True).drop(columns=["_abs"])

    if resolved_model_type == "lasso":
        xaxis_title = "LASSO coefficient (standardized)"
        color_bar_title = "LASSO<br>coefficient<br>(standardized)"
        w_max = float(df["weight"].abs().max()) or 1e-9
        fig = px.bar(
            df,
            x="weight",
            y="feature",
            orientation="h",
            color="weight",
            color_continuous_scale="RdBu_r",
            range_color=(-w_max, w_max),
            labels={"weight": xaxis_title, "feature": "Phenotypic property"},
            title=(
                f"Multivariate interpretability: {dr_method} {target} ({model_name})<br>"
                f"<sup>On-axis fit (R², descriptive): {r2:.2f} | "
                f"Top {len(df)} drivers by |effect|</sup>"
            ),
        )
    else:
        xaxis_title = "Gini importance"
        color_bar_title = "Feature importance (Gini)"
        fig = px.bar(
            df,
            x="weight",
            y="feature",
            orientation="h",
            color="weight",
            color_continuous_scale="Reds",
            labels={"weight": xaxis_title, "feature": "Phenotypic property"},
            title=(
                f"Multivariate interpretability: {dr_method} {target} ({model_name})<br>"
                f"<sup>On-axis fit (R², descriptive): {r2:.2f} | "
                f"Top {len(df)} drivers by |effect|</sup>"
            ),
        )

    fig.update_traces(
        marker_line_color=BAR_ZERO_LINE_COLOR,
        marker_line_width=BAR_ZERO_LINE_WIDTH,
    )
    fig.update_layout(
        width=figsize[0] * 100 if figsize else None,
        height=figsize[1] * 100 if figsize else None,
        xaxis_title=xaxis_title,
        yaxis_title=None,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 80, "b": 40},
        coloraxis_colorbar={"title": color_bar_title},
        xaxis={
            "zeroline": True,
            "zerolinecolor": BAR_ZERO_LINE_COLOR,
            "zerolinewidth": BAR_ZERO_LINE_WIDTH,
        },
    )

    return fig


def _display_multivariate_interpretability(
    drivers_df: pd.DataFrame,
    meta: dict[str, Any],
    plot: bool = True,
    return_fig: bool = False,
    top_k: int = 10,
    figsize: tuple[int, int] = (10, 8),
) -> Figure | None:
    """Show or return multivariate interpretability visualization; optionally log a text summary.

    Args:
        drivers_df: DataFrame with columns ``feature`` and ``weight``.
        meta: Run metadata (r2, method, model_type, target_component, ...).
        plot: If True, build/show a bar chart when applicable. If False, text summary only
            (unless *return_fig* requests a figure).
        return_fig: If True, return the Plotly figure and do not call ``fig.show()``.
        top_k: Number of top drivers to include.
        figsize: Figure size (width, height); scaled for Plotly as in other mixins.

    Returns:
        The Plotly figure if one was built and *return_fig* is True; otherwise ``None``.
    """
    if not plot:
        _log_multivariate_interpretability_results(drivers_df, meta, top_k)
        if not return_fig:
            return None

    fig = _build_multivariate_interpretability_figure(
        drivers_df, meta, top_k=top_k, figsize=figsize
    )
    if fig is None:
        return None

    from ._helpers import handle_figure_output

    return handle_figure_output(fig, plot, return_fig)


class _InterpretabilityPlotsMixin:
    """Mixin providing multivariate interpretability plots (internal visualization hooks)."""

    def _plot_multivariate_interpretability(
        self,
        drivers_df: pd.DataFrame,
        meta: dict[str, Any] | None = None,
        plot: bool = True,
        return_fig: bool = False,
        top_k: int = 10,
        figsize: tuple[int, int] = (10, 8),
    ) -> Figure | None:
        """Internal API: plot or summarize multivariate interpretability results.

        Prefer :meth:`PhenoMeAnalysis.compute_multivariate_interpretability` with ``plot`` /
        ``return_fig`` for the integrated workflow.
        """
        return _display_multivariate_interpretability(
            drivers_df,
            meta or {},
            plot=plot,
            return_fig=return_fig,
            top_k=top_k,
            figsize=figsize,
        )
