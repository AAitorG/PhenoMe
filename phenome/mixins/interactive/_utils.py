"""Pure utility helpers for the interactive explorer.

These helpers have no ``PhenoMeInteractive`` dependency and can be imported
anywhere inside the ``interactive`` subpackage.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ._constants import DEFER_UI_SEC


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds for status labels (adds m/h as duration grows).

    Uses tenths under 10s for responsiveness; then whole seconds; then ``Xm Ys`` / ``Xh Ym Zs``.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Event loop scheduling
# ---------------------------------------------------------------------------
def schedule_after_plotly_event_loop(fn: Callable[[], None]) -> None:
    """Run ``fn`` after the current stack unwinds when an asyncio loop exists.

    Plotly ``FigureWidget`` click handlers run inside traitlets / widget plumbing;
    updating the same widget tree, ``Output``, and matplotlib immediately can
    re-enter and freeze Jupyter or the VS Code / Cursor notebook UI. Deferring
    one tick avoids that without delaying visible feedback noticeably.

    The same applies to **ipywidgets** ``observe`` callbacks (dropdowns, sliders):
    rebuilding a ``FigureWidget`` or large ``batch_update`` synchronously while
    the widget frontend is still finishing the interaction can block the UI and
    prevent selecting other notebook cells until the sync completes.

    In VS Code / Cursor, a short delay is more reliable than ``call_soon`` for
    ensuring the frontend has finished processing the original interaction.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.call_later(DEFER_UI_SEC, fn)
        else:
            fn()
    except RuntimeError:
        # No loop running, but we might be in a thread that wants to update UI.
        # Fallback to a thread-based deferral so callers always get a best-effort
        # delay equivalent to the asyncio path.
        threading.Timer(DEFER_UI_SEC, fn).start()


# ---------------------------------------------------------------------------
# FigureWidget safety
# ---------------------------------------------------------------------------
def figurewidget_safe_figure(fig: go.Figure) -> go.Figure:
    """Return a figure whose data/layout use JSON-native types (lists, floats).

    ``plotly.express`` leaves NumPy arrays on traces; ``FigureWidget`` state sync
    in Plotly's ``basewidget`` uses ``if not value`` on nested props, which
    raises on multi-element arrays. Converting known trace fields to Python lists
    avoids a full ``pio.to_json`` / ``from_json`` round-trip (very slow on large
    point clouds) while fixing widget sync.
    """
    out = go.Figure(fig)
    for tr in out.data:
        for attr in ("x", "y", "z", "customdata", "text", "hovertext", "ids"):
            val = getattr(tr, attr, None)
            if isinstance(val, np.ndarray):
                setattr(tr, attr, val.tolist())
        m = tr.marker
        if m is not None:
            for mattr in ("color", "size", "opacity"):
                if hasattr(m, mattr):
                    mv = getattr(m, mattr)
                    if isinstance(mv, np.ndarray):
                        setattr(tr.marker, mattr, mv.tolist())
    return out


# ---------------------------------------------------------------------------
# Hover label colouring
# ---------------------------------------------------------------------------
def hover_text_color_for_bg(bg: str) -> str:
    """Readable hover text: light on dark marker colors, dark on light."""
    if not bg or not isinstance(bg, str):
        return "#f8fafc"
    s = bg.strip()
    try:
        if s.startswith("#"):
            from plotly.colors import hex_to_rgb

            r, g, b = hex_to_rgb(s)
        elif s.startswith("rgb"):
            inner = s[s.find("(") + 1 : s.rfind(")")].split(",")
            r, g, b = (int(float(inner[i].strip())) for i in range(3))
        else:
            return "#f8fafc"
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return "#0f172a" if lum > 0.62 else "#f8fafc"
    except Exception:
        return "#f8fafc"


def set_trace_hoverlabel_uniform(tr: Any, color: str) -> None:
    """Apply a single ``color`` as hover background on ``tr``."""
    tr.update(
        hoverlabel={
            "bgcolor": color,
            "bordercolor": "rgba(0,0,0,0.18)",
            "font": {"color": hover_text_color_for_bg(color), "size": 11},
        }
    )


def set_trace_hoverlabel_continuous(tr: Any, colorscale_name: str) -> None:
    """Per-point hover background sampled from the same colorscale as markers."""
    from plotly.colors import sample_colorscale

    mc = tr.marker.color
    vals = np.asarray(mc, dtype=float)
    if vals.size == 0:
        return
    valid = np.isfinite(vals)
    cmin = float(np.nanmin(vals))
    cmax = float(np.nanmax(vals))
    if cmax <= cmin:
        cmax = cmin + 1e-12
    t_arr = np.clip((vals - cmin) / (cmax - cmin), 0, 1)
    try:
        bg_list = sample_colorscale(colorscale_name, t_arr.tolist())
    except Exception:
        bg_list = ["rgba(55,55,55,0.92)"] * len(vals)
    bg_list = [bg_list[i] if valid[i] else "rgba(110,110,110,0.92)" for i in range(len(vals))]
    fc_list = [hover_text_color_for_bg(b) for b in bg_list]
    tr.update(
        hoverlabel={
            "bgcolor": bg_list,
            "bordercolor": "rgba(0,0,0,0.15)",
            "font": {"color": fc_list, "size": 11},
        }
    )


def apply_hoverlabels_matching_markers(
    fig: go.Figure,
    *,
    color_column: str | None,
    is_continuous: bool,
    colorscale_name: str,
    fallback_uniform: str,
) -> None:
    """Set trace hoverlabel colors to match marker colors (per-point bgcolor supported by Plotly)."""
    for tr in fig.data:
        ttype = getattr(tr, "type", "")
        if ttype not in ("scatter", "scattergl", "scatter3d"):
            continue
        mc = tr.marker.color
        if color_column is None:
            set_trace_hoverlabel_uniform(tr, fallback_uniform)
            continue
        if is_continuous:
            set_trace_hoverlabel_continuous(tr, colorscale_name)
            continue
        if isinstance(mc, str):
            set_trace_hoverlabel_uniform(tr, mc)
        else:
            set_trace_hoverlabel_uniform(tr, fallback_uniform)


def raw_index_from_customdata_row(cd: Any) -> int:
    """First column of a Plotly ``customdata`` row (image index in the pipeline)."""
    if cd is None:
        return -1
    if isinstance(cd, np.ndarray):
        v = np.asarray(cd).flat[0]
    elif isinstance(cd, (list, tuple)):
        if not cd:
            return -1
        v = cd[0]
    else:
        v = cd
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1
