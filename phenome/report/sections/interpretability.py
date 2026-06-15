"""Multivariate interpretability section generator."""

from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import plotly.graph_objects as go

from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from ...utils.display_names import format_dr_method
from .._components import generate_info_box, generate_plot_container
from ..helpers import apply_dark_theme, plotly_to_html_fragment

if TYPE_CHECKING:
    from ..context import ReportContext


def _unpack_interpretability_results(
    raw_out: pd.DataFrame | tuple,
) -> tuple[pd.DataFrame, dict[str, Any], Any]:
    """Unpack compute_multivariate_interpretability with return_meta and return_fig."""
    if isinstance(raw_out, pd.DataFrame):
        return raw_out, {}, None
    if len(raw_out) == 3:
        meta = raw_out[1] if isinstance(raw_out[1], dict) else {}
        return raw_out[0], meta, raw_out[2]
    if len(raw_out) == 2:
        if isinstance(raw_out[1], dict):
            return raw_out[0], raw_out[1], None
        return raw_out[0], {}, raw_out[1]
    return pd.DataFrame(), {}, None


def generate_interpretability_section(
    ctx: "ReportContext",
    method: str = "tsne",
    component: int = 1,
    model_type: str = "lasso",
    top_k: int = 20,
) -> str:
    """Generate multivariate interpretability section.

    Args:
        ctx: Report context with pipeline and results.
        method: Dimensionality reduction method to explain.
        component: Which component to explain.
        model_type: Regression model to use ('lasso' or 'random_forest').
        top_k: Number of top driving properties to show.

    Returns:
        HTML string for the interpretability section.
    """
    pipeline = ctx.pipeline
    try:
        # Compute multivariate interpretability via the mixin
        interp_out = pipeline.compute_multivariate_interpretability(
            method=method,
            component=component,
            model_type=model_type,
            seed=getattr(pipeline, "seed", None),
            plot=False,
            return_fig=True,
            return_meta=True,
            top_k=top_k,
        )
        df, meta, fig = _unpack_interpretability_results(interp_out)
    except (ValueError, KeyError, RuntimeError, ImportError) as e:
        return generate_info_box(f"Could not compute multivariate interpretability: {e}", "warning")

    if df.empty:
        return generate_info_box(
            f"No multivariate interpretability data available for {format_dr_method(method)} component {component}.",
            "info",
        )

    if fig is None:
        return generate_info_box("Failed to generate interpretability plot.", "warning")

    plot_fig = cast(go.Figure, fig)
    apply_dark_theme(plot_fig)

    r2 = float(meta.get("r2", 0.0))
    n_samples = int(meta.get("n_samples", 0))
    model_name = "LASSO" if model_type == "lasso" else "Random Forest"

    interpretation = ""
    if r2 >= 0.7:
        interpretation = (
            f"The {model_name} model shows strong on-axis alignment ({r2:.2f}): "
            "classical features track much of the variation along this axis on the "
            "same dataset (descriptive; not out-of-sample validation)."
        )
    elif r2 >= 0.3:
        interpretation = (
            f"The {model_name} model shows moderate on-axis alignment ({r2:.2f}): "
            "classical features partially track variation along this axis."
        )
    else:
        interpretation = (
            f"The {model_name} model shows weak on-axis alignment ({r2:.2f}): "
            "classical features weakly track this axis; the embedding may encode "
            "patterns outside the current property set."
        )

    algorithm_desc = ""
    if model_type == "lasso":
        algorithm_desc = (
            "LASSO with 5-fold CV for λ selection and fold-wise scaling. "
            "R² summarizes descriptive fit to a globally defined DR axis; see documentation "
            "for interpretation limits with t-SNE/UMAP."
        )
    else:
        algorithm_desc = "Random Forest regressor. This captures non-linear relationships between features and the axis, providing Gini importance weights."

    return f"""
    <p>Multivariate interpretability explains the variability captured by deep learning embedding dimensions (like t-SNE axis 1)
    using classical phenotypic properties. This captures how multiple features work together to define the "deep" phenotype.</p>

    <div class="info-box info">
        <strong>On-axis fit (R², descriptive): {r2:.2f} ({model_name})</strong><br>
        {interpretation}
        <br><small>Computed on {n_samples} samples using {algorithm_desc}</small>
    </div>

    {generate_plot_container(plotly_to_html_fragment(plot_fig, config=PLOTLY_DISPLAY_CONFIG))}
    """
