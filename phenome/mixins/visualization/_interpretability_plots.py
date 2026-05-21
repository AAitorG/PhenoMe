"""Multivariate interpretability plot methods for PhenoMeVisualization."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from ..._logging import get_logger
from ...utils.display_names import format_dr_method

logger = get_logger(__name__)


def _log_multivariate_interpretability_results(results: dict[str, Any], top_k: int) -> None:
    """Log a plain-text summary of multivariate interpretability to the tool logger."""
    if not isinstance(results, dict) or not results.get("drivers"):
        logger.warning("No interpretability drivers to summarize.")
        return

    r2 = float(results.get("r2", 0.0))
    drivers = results["drivers"][:top_k]

    logger.info("Explainability Score (R2): %.2f", r2)
    logger.info("Top drivers:")
    for driver in drivers:
        logger.info("  [%+.3f] %s", float(driver["weight"]), driver["feature"])


def _build_multivariate_interpretability_figure(
    results: dict[str, Any],
    top_k: int,
    figsize: tuple[int, int],
) -> Figure | None:
    """Build a horizontal bar chart of top drivers ordered by absolute magnitude (largest at top)."""
    if not isinstance(results, dict):
        logger.warning(
            "multivariate interpretability plot: expected a dict from "
            "compute_multivariate_interpretability, got %s.",
            type(results).__name__,
        )
        return None

    resolved_model_type = results.get("model_type", "lasso")
    if resolved_model_type not in ("lasso", "random_forest"):
        resolved_model_type = "lasso"

    if not results.get("drivers"):
        logger.warning("No interpretability results to plot.")
        return None

    r2 = results["r2"]
    drivers = results["drivers"][:top_k]
    target = results["target_component"]
    dr_method = format_dr_method(str(results["method"]))
    model_name = "LASSO" if resolved_model_type == "lasso" else "Random Forest"

    df = pd.DataFrame(drivers)
    df["_abs"] = df["weight"].abs()
    # Ascending by |weight| so the largest drivers appear at the top of the horizontal chart
    df = df.sort_values("_abs", ascending=True)
    df = df.drop(columns=["_abs"])

    if resolved_model_type == "lasso":
        xaxis_title = "LASSO coefficient (standardized)"
        color_bar_title = "LASSO<br>coefficient<br>(standardized)"
        # Symmetric diverging scale: blue (negative) → white → red (positive). Plotly's RdBu
        # maps low→red; RdBu_r maps low→blue so negatives (low end of range_color) are blue.
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
                f"<sup>Explainability score (R²): {r2:.2f} | "
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
                f"<sup>Explainability score (R²): {r2:.2f} | "
                f"Top {len(df)} drivers by |effect|</sup>"
            ),
        )

    fig.update_layout(
        width=figsize[0] * 100 if figsize else None,
        height=figsize[1] * 100 if figsize else None,
        xaxis_title=xaxis_title,
        yaxis_title=None,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 80, "b": 40},
        coloraxis_colorbar={"title": color_bar_title},
    )

    fig.add_vline(x=0, line_width=1, line_color="black")
    return fig


def _interpretability_df_to_dict(
    df: pd.DataFrame,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert interpretability DataFrame (+ optional meta) to the internal dict format."""
    if df.empty and not meta:
        return {}

    results: dict[str, Any] = dict(meta or {})
    if "drivers" not in results and not df.empty:
        results["drivers"] = df[["feature", "weight"]].to_dict("records")
    elif "drivers" not in results:
        results["drivers"] = []

    return results


def _display_multivariate_interpretability(
    results: dict[str, Any] | pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]],
    plot: bool = True,
    return_fig: bool = False,
    top_k: int = 10,
    figsize: tuple[int, int] = (10, 8),
) -> Figure | None:
    """Show or return multivariate interpretability visualization; optionally log a text summary.

    Mirrors the branching pattern used by :meth:`_DistancePlotsMixin._plot_distance_distribution`.

    Args:
        results: Output from :meth:`PhenoMeAnalysis.compute_multivariate_interpretability`.
            Can be a dict (old format) or a pd.DataFrame (new format).
        plot: If True, build/show a bar chart when applicable. If False, text summary only
            (unless *return_fig* requests a figure).
        return_fig: If True, return the Plotly figure and do not call ``fig.show()``.
        top_k: Number of top drivers to include.
        figsize: Figure size (width, height); scaled for Plotly as in other mixins.

    Returns:
        The Plotly figure if one was built and *return_fig* is True; otherwise ``None``.
    """
    if isinstance(results, tuple) and len(results) >= 1:
        df_part = results[0]
        if not isinstance(df_part, pd.DataFrame):
            df_part = pd.DataFrame()
        meta_part: dict[str, Any] | None = None
        if (len(results) == 3 and isinstance(results[1], dict)) or (
            len(results) >= 2 and isinstance(results[1], dict)
        ):
            meta_part = results[1]
        results = _interpretability_df_to_dict(df_part, meta_part)
    elif isinstance(results, pd.DataFrame):
        results = _interpretability_df_to_dict(results)

    if not isinstance(results, dict):
        logger.warning(
            "multivariate interpretability display: expected a dict or DataFrame, got %s.",
            type(results).__name__,
        )
        return None

    if not plot:
        _log_multivariate_interpretability_results(results, top_k)
        if not return_fig:
            return None

    fig = _build_multivariate_interpretability_figure(results, top_k=top_k, figsize=figsize)
    if fig is None:
        return None

    from ._helpers import handle_figure_output

    return handle_figure_output(fig, plot, return_fig)


class _InterpretabilityPlotsMixin:
    """Mixin providing multivariate interpretability plots (internal visualization hooks)."""

    def _plot_multivariate_interpretability(
        self,
        results: dict[str, Any] | pd.DataFrame,
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
            results,
            plot=plot,
            return_fig=return_fig,
            top_k=top_k,
            figsize=figsize,
        )
