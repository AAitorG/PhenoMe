"""Utility helpers and color option discovery for the interactive explorer.

Consolidates Pure utility functions and Color-by dropdown discovery logic
to reduce module fragmentation.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..._logging import get_logger
from ...core import get_all_metadata_keys, get_metadata_value_from_dict
from ._html_components import (
    DEFER_UI_SEC,
    DR_RANDOM_STATE,
    INTERACTIVE_COLOR_SCAN_MAX_INDICES,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds for status labels."""
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
    """Run ``fn`` on the kernel loop after the current stack unwinds.

    Widget callbacks and Colab compute threads often have no running asyncio
    loop. A ``threading.Timer`` then builds the FigureWidget off the kernel
    thread, and Colab drops the view. Hand the callback to the IPython kernel
    IOLoop with ``add_callback`` so ``call_later`` runs on the loop thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        loop.call_later(DEFER_UI_SEC, fn)
        return

    io_loop = _kernel_io_loop()
    if io_loop is not None:
        try:
            io_loop.add_callback(lambda: io_loop.call_later(DEFER_UI_SEC, fn))
        except Exception:
            logger.warning(
                "Failed to schedule a UI update on the kernel loop",
                exc_info=True,
            )
        return
    threading.Timer(DEFER_UI_SEC, fn).start()


def _kernel_io_loop() -> Any:
    """Return the IPython kernel IOLoop, or None when this process has none."""
    try:
        from IPython import get_ipython

        kernel = getattr(get_ipython(), "kernel", None)
        return getattr(kernel, "io_loop", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FigureWidget safety
# ---------------------------------------------------------------------------
def figurewidget_safe_figure(fig: go.Figure) -> go.Figure:
    """Return a figure whose data/layout use JSON-native types (lists, floats)."""
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
    """Set trace hoverlabel colors to match marker colors."""
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
    except (ValueError, TypeError):
        return -1


# ---------------------------------------------------------------------------
# Color scan helpers
# ---------------------------------------------------------------------------
def indices_for_color_scan(n_total: int, *, seed: int = DR_RANDOM_STATE) -> np.ndarray:
    """Return image indices spanning the dataset for Color-by dropdown informativeness."""
    if n_total <= 0:
        return np.array([], dtype=np.int64)
    cap = INTERACTIVE_COLOR_SCAN_MAX_INDICES
    if n_total <= cap:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=cap, replace=False)


def build_informative_color_columns(results: Any, property_keys: list[str]) -> list[str]:
    """Compute metadata/property columns eligible for the explorer Color-by dropdown."""
    meta_keys_list = get_all_metadata_keys(results)
    meta_keys_set = frozenset(meta_keys_list)
    cols = sorted(set(meta_keys_list).union(property_keys))
    informative_cols: list[str] = []
    n_total = results.n_images
    scan_ix = indices_for_color_scan(n_total)

    for col in cols:
        vals: list[Any] = []
        is_meta = col in meta_keys_set
        if is_meta:
            for i in scan_ix:
                ii = int(i)
                meta = results.metadata[ii]
                if isinstance(meta, dict):
                    v = get_metadata_value_from_dict(meta, col)
                    if v is not None:
                        vals.append(v)
        else:
            for i in scan_ix:
                ii = int(i)
                props = results.properties[ii]
                if isinstance(props, dict):
                    v = props.get(col)
                    if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v)):
                        vals.append(v)

        if not vals:
            continue

        unique_vals = set(vals)
        if len(unique_vals) <= 1:
            continue

        if (
            len(unique_vals) > n_total * 0.9
            and n_total > 10
            and col.lower() in ("id", "filename", "file_path", "path", "index")
        ):
            continue

        informative_cols.append(col)

    return sorted(informative_cols)
