"""Shared Plotly.js display options for notebooks and HTML reports."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

PLOTLY_DISPLAY_CONFIG: dict[str, Any] = {"displaylogo": False}


def register_plotly_display_defaults() -> None:
    """Merge `PLOTLY_DISPLAY_CONFIG` into every plotly.io renderer that holds a dict config."""
    for _name, renderer in pio.renderers.items():
        cfg = getattr(renderer, "config", None)
        if isinstance(cfg, dict):
            cfg.update(PLOTLY_DISPLAY_CONFIG)


def apply_figurewidget_display_config(fig: go.FigureWidget) -> None:
    """Hide the Plotly logo on a FigureWidget (plotly.js config is not set via pio.renderers)."""
    fig._config = {**fig._config, **PLOTLY_DISPLAY_CONFIG}
