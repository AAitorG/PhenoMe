"""Multivariate interpretability plot methods for PhenoMeVisualization."""

from typing import Any

import pandas as pd
import plotly.express as px

from ..._logging import get_logger

logger = get_logger(__name__)


class _InterpretabilityPlotsMixin:
    """Mixin providing multivariate interpretability plots."""

    def plot_multivariate_interpretability(
        self,
        results: dict[str, Any],
        *,
        top_k: int = 10,
        figsize: tuple[int, int] = (10, 8),
        return_fig: bool = False,
    ) -> Any:
        """Plot the results of multivariate interpretability (LASSO or Random Forest).

        Displays a horizontal bar chart showing which classical features best explain
        the target embedding dimension.

        Args:
            results: Output dict from :meth:`compute_multivariate_interpretability`
                (must include ``drivers``, ``r2``, ``method``, ``target_component``,
                ``model_type``).
            top_k: Number of top driving features to show in the plot.
            figsize: Figure size (width, height) in pixels / 100.
            return_fig: If True, return the plotly figure object.

        Returns:
            The plotly figure object if return_fig is True, else None.
        """
        if not isinstance(results, dict):
            logger.warning(
                "plot_multivariate_interpretability: expected a dict from "
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
        dr_method = results["method"].upper()
        model_name = "LASSO" if resolved_model_type == "lasso" else "Random Forest"

        # Create DataFrame for plotting
        df = pd.DataFrame(drivers)

        if resolved_model_type == "lasso":
            df = df.sort_values(
                "weight", ascending=True
            )  # Ascending for better horizontal bar display
            # Define color based on weight sign
            df["Color"] = df["weight"].apply(
                lambda x: "Positive Correlation" if x > 0 else "Negative Correlation"
            )
            color_discrete_map = {
                "Positive Correlation": "#ef553b",
                "Negative Correlation": "#636efa",
            }
            xaxis_title = "LASSO Coefficient (Standardized)"
        else:
            df = df.sort_values(
                "weight", ascending=True
            )  # Importance is positive, sorted by magnitude
            df["Color"] = "Feature Importance"
            color_discrete_map = {"Feature Importance": "#10b981"}
            xaxis_title = "Gini Importance"

        title = (
            f"Multivariate Explanation: {dr_method} {target} ({model_name})<br>"
            f"<sup>Explainability Score (R²): {r2:.2f} | "
            f"Top {len(df)} features shown</sup>"
        )

        fig = px.bar(
            df,
            x="weight",
            y="feature",
            orientation="h",
            color="Color",
            labels={"weight": xaxis_title, "feature": "Phenotypic Property"},
            title=title,
            color_discrete_map=color_discrete_map,
        )

        fig.update_layout(
            width=figsize[0] * 100 if figsize else None,
            height=figsize[1] * 100 if figsize else None,
            xaxis_title=xaxis_title,
            yaxis_title=None,
            showlegend=(resolved_model_type == "lasso"),
            legend_title_text=None,
            margin={"l": 20, "r": 20, "t": 80, "b": 40},
        )

        # Add vertical line at 0
        fig.add_vline(x=0, line_width=1, line_color="black")

        if return_fig:
            return fig
        fig.show()
        return None
