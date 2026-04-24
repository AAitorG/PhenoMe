"""Multivariate interpretability section generator."""

from typing import TYPE_CHECKING

from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from ..components import generate_info_box, generate_plot_container
from ..helpers import apply_dark_theme, plotly_to_html_fragment

if TYPE_CHECKING:
    from ..context import ReportContext


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
        results = pipeline.compute_multivariate_interpretability(
            method=method,
            component=component,
            model_type=model_type,
            seed=getattr(pipeline, "seed", None),
            plot=True,
            return_fig=True,
            top_k=top_k,
        )
    except (ValueError, KeyError, RuntimeError, ImportError) as e:
        return generate_info_box(f"Could not compute multivariate interpretability: {e}", "warning")

    if not results or not results.get("drivers"):
        return generate_info_box(
            f"No multivariate interpretability data available for {method.upper()} component {component}.",
            "info",
        )

    fig = results.get("interpretability_fig")

    if fig is None:
        return generate_info_box("Failed to generate interpretability plot.", "warning")

    apply_dark_theme(fig)

    r2 = results["r2"]
    n_samples = results["n_samples"]
    model_name = "LASSO" if model_type == "lasso" else "Random Forest"

    interpretation = ""
    if r2 >= 0.7:
        interpretation = f"The {model_name} model shows high explainability ({r2:.2f}), suggesting that most of what the deep learning model sees along this axis is captured by classical features."
    elif r2 >= 0.3:
        interpretation = f"The {model_name} model shows moderate explainability ({r2:.2f}), suggesting that classical features partially capture the phenotypic variation in this dimension."
    else:
        interpretation = f"The {model_name} model shows low explainability ({r2:.2f}), suggesting that the neural network has found novel phenotypic patterns that classical features cannot fully describe."

    algorithm_desc = ""
    if model_type == "lasso":
        algorithm_desc = "LASSO regression with cross-validation. This identifies a sparse set of features that linearly combine to explain the axis."
    else:
        algorithm_desc = "Random Forest regressor. This captures non-linear relationships between features and the axis, providing Gini importance weights."

    return f"""
    <p>Multivariate interpretability explains the variability captured by deep learning embedding dimensions (like t-SNE axis 1)
    using classical phenotypic properties. This captures how multiple features work together to define the "deep" phenotype.</p>

    <div class="info-box info">
        <strong>Explainability Score (R²): {r2:.2f} ({model_name})</strong><br>
        {interpretation}
        <br><small>Computed on {n_samples} samples using {algorithm_desc}</small>
    </div>

    {generate_plot_container(plotly_to_html_fragment(fig, config=PLOTLY_DISPLAY_CONFIG))}
    """
