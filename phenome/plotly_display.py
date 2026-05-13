"""Shared Plotly.js display options for notebooks and HTML reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.basewidget import BaseFigureWidget

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


# ---------------------------------------------------------------------------
# Defensive patch for a plotly FigureWidget bug.
#
# ``BaseFigureWidget._remove_overlapping_props`` (plotly >=5, still in 6.6.0)
# evaluates ``if not input_val:`` after recursing into ``_props``.  When the
# trace prop happens to be a numpy array (FigureWidget re-normalizes some
# values to ndarrays internally even when constructed from plain lists), this
# raises ``ValueError: The truth value of an array with more than one element
# is ambiguous``. The frontend then bubbles the error up from
# ``_handler_js2py_traceDeltas`` on every interactive update.
#
# We replace the truthiness check with an array-safe one so empty
# dicts/lists still get pruned (the function's intent) without exploding on
# arrays.  Installed once at import time so any code path that constructs a
# ``FigureWidget`` is protected.
# ---------------------------------------------------------------------------
def _install_figurewidget_truthiness_patch() -> None:
    """Patch ``BaseFigureWidget._remove_overlapping_props`` to be ndarray-safe."""
    if getattr(BaseFigureWidget, "_phenome_overlap_patch_installed", False):
        return

    def _is_empty(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, np.ndarray):
            # Arrays are leaf data; never treat them as "the dict has been
            # emptied" — let plotly keep them in ``_props``.
            return False
        try:
            return not val
        except (ValueError, TypeError):
            return False

    def _remove_overlapping_props(input_data, delta_data, prop_path=()):  # type: ignore[no-untyped-def]
        removed: list[tuple[Any, ...]] = []
        if isinstance(input_data, dict):
            assert isinstance(delta_data, dict)
            for p, delta_val in list(delta_data.items()):
                if isinstance(delta_val, dict) or go.Figure._is_dict_list(delta_val):
                    if p in input_data:
                        input_val = input_data[p]
                        recur_prop_path = (*prop_path, p)
                        recur_removed = _remove_overlapping_props(
                            input_val, delta_val, recur_prop_path
                        )
                        removed.extend(recur_removed)
                        if _is_empty(input_val):
                            input_data.pop(p)
                            removed.append(recur_prop_path)
                elif p in input_data and p != "uid":
                    input_data.pop(p)
                    removed.append((*prop_path, p))
        elif isinstance(input_data, list):
            assert isinstance(delta_data, list)
            for i, delta_val in enumerate(delta_data):
                if i >= len(input_data):
                    break
                input_val = input_data[i]
                if (
                    input_val is not None and isinstance(delta_val, dict)
                ) or go.Figure._is_dict_list(delta_val):
                    recur_prop_path = (*prop_path, i)
                    recur_removed = _remove_overlapping_props(input_val, delta_val, recur_prop_path)
                    removed.extend(recur_removed)
        return removed

    BaseFigureWidget._remove_overlapping_props = staticmethod(_remove_overlapping_props)
    BaseFigureWidget._phenome_overlap_patch_installed = True


_install_figurewidget_truthiness_patch()
