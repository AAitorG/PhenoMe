"""
Interactive Visualization Module for PhenoMe.

This module provides the PhenoMeInteractive class for high-performance,
interactive exploration of large phenotypic datasets within Jupyter Notebooks.
It uses Plotly FigureWidgets and ipywidgets for a responsive experience.

Key design:
  - **Color changes** are instant (no recomputation, only visual update).
  - **Filter/Exclude** (collapsible under Embedding) restrict data before recomputation;
    same field + search + multi-select pattern as Highlight.
  - **Method/Source/Dim** or metadata filter changes trigger dimensionality reduction (expensive).
  - **Highlight mode** shows ALL data points but visually emphasises a matching
    subset (larger, brighter markers with a contrasting border) instead of hiding
    the rest.  This lets the user "find" a group in context.
  - **Multi-value highlight**: categorical fields use filter + multi-select (OR).
  - **Live stats bar** (total / highlighted / selected).
  - **Selected point**: clicking a point highlights it (larger, border) for inspection.
"""

import asyncio
import contextlib
import html
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output, display

from .._logging import get_logger
from ..core import (
    get_all_metadata_keys,
    get_metadata_value_from_dict,
    run_dimensionality_reduction,
)
from ..plotly_display import apply_figurewidget_display_config

logger = get_logger(__name__)


# Small delay so the browser finishes dropdown/slider closure and releases focus
# before we push large FigureWidget state syncs (avoids notebook/IDE cell selection
# getting "stuck" after Appearance / Highlight interactions).
_DEFER_UI_SEC = 0.1
# Typing debounce for filter/exclude/highlight search inputs
_SEARCH_DEBOUNCE_SEC = 0.25
# Click handler re-entry guard (debounce)
_CLICK_DEBOUNCE_SEC = 0.15
# Reproducible subsample when downsampling in _compute_embedding
_DR_RANDOM_STATE = 42
# Extra Plotly trace for click-to-select border (2D/3D); must match overlay click skip
_SELECTION_OVERLAY_NAME = "_phenome_sel_overlay"


def _schedule_after_plotly_event_loop(fn: Callable[[], None]) -> None:
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
            # Using call_later with a small delay is more robust in VS Code/Cursor
            # than call_soon for preventing UI focus/lockup issues.
            loop.call_later(_DEFER_UI_SEC, fn)
        else:
            fn()
    except RuntimeError:
        # No loop running, but we might be in a thread that wants to update UI.
        # If we're not on the main thread, we should try to find the main loop.
        # For simplicity, we fallback to a thread-based deferral.
        threading.Timer(_DEFER_UI_SEC, fn).start()


def _figurewidget_safe_figure(fig: go.Figure) -> go.Figure:
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


def _hover_text_color_for_bg(bg: str) -> str:
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
            r, g, b = [int(float(inner[i].strip())) for i in range(3)]
        else:
            return "#f8fafc"
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return "#0f172a" if lum > 0.62 else "#f8fafc"
    except Exception:
        return "#f8fafc"


def _set_trace_hoverlabel_uniform(tr: Any, color: str) -> None:
    tr.update(
        hoverlabel={
            "bgcolor": color,
            "bordercolor": "rgba(0,0,0,0.18)",
            "font": {"color": _hover_text_color_for_bg(color), "size": 11},
        }
    )


def _set_trace_hoverlabel_continuous(tr: Any, colorscale_name: str) -> None:
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
    fc_list = [_hover_text_color_for_bg(b) for b in bg_list]
    tr.update(
        hoverlabel={
            "bgcolor": bg_list,
            "bordercolor": "rgba(0,0,0,0.15)",
            "font": {"color": fc_list, "size": 11},
        }
    )


def _apply_hoverlabels_matching_markers(
    fig: go.Figure,
    *,
    color_column: str | None,
    is_continuous: bool,
    colorscale_name: str,
    fallback_uniform: str,
) -> None:
    """Set trace hoverlabel colors to match marker colors (Plotly supports per-point bgcolor)."""
    for tr in fig.data:
        ttype = getattr(tr, "type", "")
        if ttype not in ("scatter", "scattergl", "scatter3d"):
            continue
        mc = tr.marker.color
        if color_column is None:
            _set_trace_hoverlabel_uniform(tr, fallback_uniform)
            continue
        if is_continuous:
            _set_trace_hoverlabel_continuous(tr, colorscale_name)
            continue
        if isinstance(mc, str):
            _set_trace_hoverlabel_uniform(tr, mc)
        else:
            _set_trace_hoverlabel_uniform(tr, fallback_uniform)


class _InteractiveExplorerProtocol(Protocol):
    """Protocol for objects that can be explored interactively (pipeline or visualization mixin)."""

    results: Any
    device: Any

    def get_available_property_keys(self) -> list[str]:
        """Return available scalar property column names."""
        ...

    def _get_color_column(self, color_by: str, df: Any) -> tuple[str, bool]:
        """Resolve ``color_by`` to a DataFrame column and whether it is continuous."""
        ...

    def plot_image_by_index(
        self,
        idx: int,
        distance_results: dict | None = None,
        channels: Any | None = None,
        figsize: tuple[float, float] = (5.0, 5.0),
        title_fields: list[str] | None = None,
        show_extra_info: bool = True,
        apply_transforms: bool = True,
        downsample: int | None = None,
    ) -> None:
        """Display the image for result index ``idx`` in the notebook."""
        ...

    def image_preview_png_bytes(
        self,
        idx: int,
        distance_results: dict | None = None,
        channels: Any | None = None,
        figsize: tuple[float, float] = (5.0, 5.0),
        title_fields: list[str] | None = None,
        show_extra_info: bool = False,
        apply_transforms: bool = True,
        downsample: int | None = None,
        dpi: int = 100,
    ) -> tuple[bytes, str | None]:
        """PNG bytes and optional details text (same as ``plot_image_by_index`` extra block)."""
        ...


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
_QUALITATIVE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
]

_CONTINUOUS_SCALES = [
    "Viridis",
    "Plasma",
    "Inferno",
    "Magma",
    "Cividis",
    "Turbo",
    "Hot",
    "Jet",
    "Blues",
    "Reds",
    "YlOrRd",
]

# Shared layout for field + search + multi-select (Highlight + Embedding filter/exclude).
_INTERACTIVE_VALUES_H = 168
# Fixed widths (px) for stable ipywidgets layout (flex alone often mis-sizes in notebooks).
_INTERACTIVE_SEARCH_W_PX = 240
_INTERACTIVE_VALUES_W_PX = 320
_INTERACTIVE_APPEAR_COL_W_PX = 300
_INTERACTIVE_FIELD_MAX_W_PX = 600
# Horizontal rule between control blocks (Highlight + Filter/exclude).
_INTERACTIVE_SECTION_SEP_HTML = (
    '<div style="height:1px;background:#DDE1E6;margin:2px 0 0 0;width:100%;"></div>'
)
# Shared label gutter for Filter/Exclude/Highlight (matches Appearance sliders).
_INTERACTIVE_DESC_STYLE = {"description_width": "70px"}
# Integer columns with at most this many distinct values use categorical highlight
# (multi-select), same idea as Filter/Exclude — avoids a range slider on cluster IDs.
_HIGHLIGHT_DISCRETE_INT_MAX_UNIQUES = 512

# Fixed pixel layout for embedding plot and click-to-inspect overlay (no flex resizing).
_EMBEDDING_FIG_WIDTH_PX = 800
_EMBEDDING_FIG_HEIGHT_PX = 700
_IMAGE_OVERLAY_WIDTH_PX = 420
# Min height for the image panel (compact idle/loading; PNG expands when shown).
_IMAGE_PANEL_MIN_HEIGHT_PX = 280
# Matplotlib figsize (inches) for overlay image; ~100 DPI matches panel width minus padding.
_IMAGE_OVERLAY_FIGSIZE: tuple[float, float] = (4.0, 4.0)


def _html_embedding_placeholder_idle(w_px: int, h_px: int) -> str:
    """Full-size empty state for the embedding slot (matches figure dimensions)."""
    wh = f"{w_px}px"
    hh = f"{h_px}px"
    return (
        f'<div style="width:{wh};max-width:100%;height:{hh};min-height:{hh};box-sizing:border-box;'
        "display:flex;align-items:center;justify-content:center;"
        "background:linear-gradient(165deg,#f8fafc 0%,#eef1f6 100%);"
        'border:1px solid #d0d8e3;border-radius:8px;overflow:hidden;position:relative;">'
        '<div style="position:absolute;inset:0;opacity:0.04;pointer-events:none;'
        "background-image:radial-gradient(#4E79A7 1px,transparent 1px);"
        'background-size:18px 18px;"></div>'
        '<div style="position:relative;z-index:1;text-align:center;padding:10px 14px;max-width:320px;'
        "border:1px solid #dce3eb;border-radius:6px;background:rgba(255,255,255,0.85);"
        'box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
        '<div style="font-size:12px;font-weight:600;color:#2c3e50;">Embedding</div>'
        '<p style="margin:6px 0 0;font-size:11px;line-height:1.4;color:#5d6d7e;">'
        "Press <b>Compute</b> to build the plot. Use <b>Compute</b> again after changing "
        "method, dims, or source.</p>"
        "</div></div>"
    )


def _html_embedding_placeholder_computing(w_px: int, h_px: int) -> str:
    """Full-size state shown while DR runs (replaces idle or previous figure)."""
    wh = f"{w_px}px"
    hh = f"{h_px}px"
    return (
        "<style>"
        "@keyframes ph-bar-slide{0%{transform:translateX(-100%);}100%{transform:translateX(350%);}}"
        "</style>"
        f'<div style="width:{wh};max-width:100%;height:{hh};min-height:{hh};box-sizing:border-box;'
        "display:flex;align-items:center;justify-content:center;"
        "background:linear-gradient(165deg,#f0f3f8 0%,#e8ecf2 100%);"
        'border:1px solid #c5d0df;border-radius:8px;overflow:hidden;position:relative;">'
        '<div style="position:relative;z-index:1;text-align:center;padding:10px 14px;max-width:300px;'
        'border:1px solid #cfd8e3;border-radius:6px;background:rgba(255,255,255,0.9);">'
        '<div style="font-size:12px;font-weight:600;color:#2c3e50;">Computing…</div>'
        '<p style="margin:4px 0 0;font-size:10px;color:#7f8c8d;line-height:1.35;">'
        "Dimensionality reduction (may take a while on large data).</p>"
        '<div style="margin:10px auto 0;width:200px;max-width:100%;height:3px;border-radius:2px;'
        'background:#d5dde8;overflow:hidden;">'
        '<div style="width:38%;height:100%;border-radius:2px;'
        "background:linear-gradient(90deg,#4E79A7,#59A14F);"
        'animation:ph-bar-slide 1.25s ease-in-out infinite;"></div></div>'
        "</div></div>"
    )


def _html_image_panel_idle(w_px: int, min_h_px: int) -> str:
    """Empty state for click-to-inspect until a point is selected."""
    mh = f"{min_h_px}px"
    box_max = max(120, min_h_px - 48)
    return (
        f'<div style="width:100%;min-width:0;min-height:{mh};box-sizing:border-box;'
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        'padding:6px 6px;gap:6px;">'
        f'<div style="width:min(100%,{box_max}px);aspect-ratio:1;max-height:{min_h_px - 40}px;'
        "border:1px dashed #c5ced8;border-radius:6px;"
        'background:#f4f6f9;display:flex;align-items:center;justify-content:center;">'
        '<div style="font-size:22px;line-height:1;opacity:0.35;color:#4E79A7;">◇</div>'
        "</div>"
        '<div style="font-size:10px;color:#6b7785;text-align:center;line-height:1.35;max-width:260px;">'
        "Click the <b>embedding</b> to preview an image.</div>"
        "</div>"
    )


def _html_image_panel_loading(w_px: int, min_h_px: int) -> str:
    """Shown while ``image_preview_png_bytes`` runs on a worker thread."""
    mh = f"{min_h_px}px"
    box_max = max(120, min_h_px - 48)
    return (
        "<style>@keyframes ph-img-pulse{0%,100%{opacity:0.4;}50%{opacity:0.85;}}</style>"
        f'<div style="width:100%;min-width:0;min-height:{mh};box-sizing:border-box;'
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        'padding:6px 6px;gap:6px;">'
        f'<div style="width:min(100%,{box_max}px);aspect-ratio:1;max-height:{min_h_px - 40}px;'
        "border:1px solid #d0d8e2;border-radius:6px;"
        "background:linear-gradient(110deg,#eceff4 0%,#f5f7fa 50%,#e8ecf2 100%);"
        "background-size:200% 100%;animation:ph-img-pulse 1s ease-in-out infinite;"
        'display:flex;align-items:center;justify-content:center;">'
        '<span style="font-size:11px;font-weight:600;color:#5d6d7e;">Loading…</span>'
        "</div>"
        "</div>"
    )


class PhenoMeInteractive:
    """
    High-performance interactive explorer for PhenoMe results.

    Optimized for large datasets using WebGL and efficient data handling.
    Provides a unified interface for 2D/3D exploration with:
      - Instant colour switching (no recomputation).
      - Highlight mode (shows all points, emphasises a subset).
      - Multi-select categorical highlight and live stats bar.
      - Click-to-inspect image viewer.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        pheno_me: _InteractiveExplorerProtocol,
        filters: dict | None = None,
        exclude: dict | None = None,
        hover_features: list[str] | None = None,
        max_points: int = 1_000_000,
    ):
        """
        Initialize the interactive explorer.

        Args:
            pheno_me: An instance of PhenoMe.
            filters: Optional dictionary of metadata filters to pre-apply.
            exclude: Optional dictionary of metadata exclusions (same structure as filters).
            hover_features: Optional list of features to show on hover.
            max_points: Maximum points to plot before subsampling.
        """
        self.pheno = pheno_me
        self.initial_filters = filters
        self.initial_exclude = exclude
        self.hover_features = hover_features
        self.max_points = max_points

        # Cached computation results (set after first compute)
        self._cached_df: pd.DataFrame | None = None
        self._cached_method: str | None = None
        self._cached_source: str | None = None
        self._cached_ndims: int | None = None
        self._cached_filters: dict | None = None
        self._cached_exclude: dict | None = None
        self._cached_dr_obj: Any = None
        # Map pipeline image index -> row position in ``_cached_df`` (avoids repeated dict builds).
        self._cached_index_to_row: dict[int, int] | None = None
        # Incremented after each successful embedding compute; image-load callbacks skip stale work.
        self._compute_seq: int = 0

        # Widget state
        self.fig_widget: go.FigureWidget | None = None
        # Number of data traces from plotly express (excludes selection overlay trace).
        self._n_data_traces: int = 0
        self._current_color_by: str | None = None
        self._selected_point_index: int | None = None  # Click-to-highlight single point
        self._image_error_msg: str | None = None
        # Group highlight on/off (replaces former checkbox; synced to ``highlight_toggle`` button UI).
        self._highlight_active: bool = False
        # Full list of categorical highlight values (for search filtering)
        self._highlight_value_options_all: list[str] = []
        # Pre-compute filter/exclude value lists (search narrows SelectMultiple like Highlight)
        self._filter_value_options_all: list[str] = []
        self._exclude_value_options_all: list[str] = []

        # Background compute (DR runs in a thread so the UI stays responsive)
        self._compute_thread: threading.Thread | None = None
        self._click_lock = threading.Lock()
        self._last_click_time = 0.0
        self._debounce_timer: threading.Timer | None = None
        self._filter_search_debounce_timer: threading.Timer | None = None
        self._exclude_search_debounce_timer: threading.Timer | None = None

        # Output area: flex child can shrink so the row does not force a horizontal
        # scrollbar when the notebook is narrow (embedding figure uses a fixed width).
        self.output_area = widgets.Output(
            layout=widgets.Layout(flex="1 1 0%", min_width="0px", width="auto"),
        )
        # Plot is shown by swapping children on this VBox.  Background compute runs in a
        # thread; ``with output_area: display(...)`` relies on the kernel's parent msg_id
        # (set only on the main thread), so ``display`` from a worker never reaches the
        # widget.  Updating ``VBox.children`` syncs correctly from any thread.
        self._embedding_placeholder = widgets.HTML(
            value=_html_embedding_placeholder_idle(
                _EMBEDDING_FIG_WIDTH_PX, _EMBEDDING_FIG_HEIGHT_PX
            )
        )
        self._embedding_placeholder_computing = widgets.HTML(
            value=_html_embedding_placeholder_computing(
                _EMBEDDING_FIG_WIDTH_PX, _EMBEDDING_FIG_HEIGHT_PX
            )
        )
        _pw = f"{_EMBEDDING_FIG_WIDTH_PX}px"
        _ph = f"{_EMBEDDING_FIG_HEIGHT_PX}px"
        self._plot_slot = widgets.VBox(
            [self._embedding_placeholder],
            layout=widgets.Layout(
                width=_pw,
                min_width=_pw,
                max_width=_pw,
                min_height=_ph,
            ),
        )
        # Do not stretch to the plot column height: a tall plot + overflow_y auto
        # makes the browser show a vertical scrollbar even when the image overlay
        # fits.  align_self flex-start keeps this box only as tall as its content;
        # overflow visible avoids a scroll track when there is nothing to scroll.
        _ow = f"{_IMAGE_OVERLAY_WIDTH_PX}px"
        # Matplotlib inside ``ipywidgets.Output`` does not reliably render from Plotly
        # click callbacks (display goes to the cell / nowhere).  Use PNG +
        # ``widgets.Image`` instead, like ``_plot_slot.children`` for the embedding.
        self._img_idle_placeholder = widgets.HTML(
            value=_html_image_panel_idle(_IMAGE_OVERLAY_WIDTH_PX, _IMAGE_PANEL_MIN_HEIGHT_PX)
        )
        self._img_loading_placeholder = widgets.HTML(
            value=_html_image_panel_loading(_IMAGE_OVERLAY_WIDTH_PX, _IMAGE_PANEL_MIN_HEIGHT_PX)
        )
        _imh = f"{_IMAGE_PANEL_MIN_HEIGHT_PX}px"
        self.img_output = widgets.VBox(
            [self._img_idle_placeholder],
            layout=widgets.Layout(
                flex="0 0 auto",
                width=_ow,
                min_width=_ow,
                max_width=_ow,
                min_height=_imh,
                border="1px solid #E0E0E0",
                border_radius="8px",
                background="#FAFAFA",
                padding="8px",
                align_self="flex-start",
                overflow_x="hidden",
                overflow_y="visible",
            ),
        )

        # Build UI widgets
        self._create_widgets()

        # Populate options
        self._update_color_options()
        self._update_highlight_key_options()
        self._update_filter_exclude_options()

        # Setup callbacks
        self._setup_callbacks()

    def close(self) -> None:
        """Cancel pending debounce timers (call before discarding the explorer)."""
        for t in (
            self._debounce_timer,
            self._filter_search_debounce_timer,
            self._exclude_search_debounce_timer,
        ):
            if t is not None:
                with contextlib.suppress(Exception):
                    t.cancel()
        self._debounce_timer = None
        self._filter_search_debounce_timer = None
        self._exclude_search_debounce_timer = None

    def show(self) -> None:
        """Display the interactive dashboard.

        Only ONE widget tree is produced per cell.  Calling ``show()``
        again (or from ``create_interactive_explorer``) first clears
        any previous output so stale / duplicate widgets never appear.
        """
        self.close()
        # Clear the entire cell output first so that stale widgets from a
        # previous run (or from autoreload) are removed before we display
        # anything new.  This is the key fix for the "two menus" problem.
        clear_output(wait=True)

        self._update_color_options()
        self._update_highlight_key_options()
        self._update_filter_exclude_options()

        # --- Accordion panels: controls in rows with wrap for better horizontal use of space ---
        _row_layout = widgets.Layout(
            width="100%",
            gap="8px",
            flex_flow="row wrap",
            align_items="center",
        )
        _embed = widgets.HBox(
            [
                self.method_dropdown,
                self.dim_toggle,
                self.source_dropdown,
                self.compute_button,
            ],
            layout=_row_layout,
        )
        _embed_full = widgets.VBox(
            [_embed, self._embedding_filter_accordion],
            layout=widgets.Layout(width="100%", gap="10px"),
        )
        _appear_row_layout = widgets.Layout(
            width="100%",
            gap="10px",
            flex_flow="row wrap",
            align_items="center",
        )
        _appear_row_sliders = widgets.HBox(
            [self.point_size_slider, self.opacity_slider],
            layout=_appear_row_layout,
        )
        # Align checkbox with control column (same 70px label gutter as sliders/dropdowns)
        _appear_row_extra = widgets.HBox(
            [self.show_extra_info_checkbox],
            layout=widgets.Layout(
                width="100%",
                padding="2px 0 0 70px",
            ),
        )
        _appear = widgets.VBox(
            [
                widgets.HBox(
                    [self.color_dropdown, self.colorscale_dropdown],
                    layout=_appear_row_layout,
                ),
                _appear_row_sliders,
                _appear_row_extra,
            ],
            layout=widgets.Layout(
                width="100%",
                gap="10px",
                padding="10px 12px 12px 12px",
            ),
        )
        _hl = widgets.VBox(
            [
                self._highlight_help,
                widgets.HTML(
                    '<div style="font-size:10px;font-weight:600;color:#334155;letter-spacing:0.02em;'
                    'margin:4px 0 2px 0;padding-bottom:4px;border-bottom:1px solid #E2E8F0;">'
                    "Field and values</div>"
                ),
                widgets.HBox(
                    [self.highlight_key_dropdown],
                    layout=widgets.Layout(width="100%"),
                ),
                self._highlight_value_container,
                self._highlight_actions_sep,
                widgets.HBox(
                    [self.highlight_toggle],
                    layout=widgets.Layout(
                        width="100%",
                        gap="10px",
                        flex_flow="row wrap",
                        align_items="center",
                        padding="4px 0 0 82px",
                    ),
                ),
            ],
            layout=widgets.Layout(
                width="100%",
                gap="10px",
                padding="10px 12px 12px 12px",
            ),
        )
        accordion = widgets.Accordion(
            children=[_embed_full, _appear, _hl],
            layout=widgets.Layout(width="100%"),
        )
        accordion.set_title(0, "⊞ Embedding")
        accordion.set_title(1, "◑ Appearance")
        accordion.set_title(2, "◎ Highlight")
        accordion.selected_index = 0

        sidebar = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="background:linear-gradient(90deg,#4E79A7,#59A14F);'
                    "color:white;padding:8px 12px 8px 20px;border-radius:6px;"
                    'font-size:14px;font-weight:600;letter-spacing:0.5px;">'
                    "PhenoMe Explorer</div>"
                ),
                accordion,
                widgets.HBox(
                    [self.status_label, self.stats_bar],
                    layout=widgets.Layout(
                        flex_flow="row wrap",
                        align_items="center",
                        gap="8px",
                        width="100%",
                    ),
                ),
            ],
            layout=widgets.Layout(
                padding="12px",
                border="1px solid #E0E0E0",
                border_radius="8px",
                background="#FAFAFA",
                width="100%",
            ),
        )

        main_area = widgets.HBox(
            [self.output_area, self.img_output],
            layout=widgets.Layout(
                gap="8px",
                width="100%",
                align_items="flex-start",
            ),
        )

        self._dashboard = widgets.VBox(
            [sidebar, main_area],
            layout=widgets.Layout(gap="8px", width="100%"),
        )

        display(self._dashboard)

        self._update_stats()

        # Placeholder lives in _plot_slot; display once here (main thread) so later
        # thread-only updates can replace _plot_slot.children with the FigureWidget.
        self._plot_slot.children = (self._embedding_placeholder,)
        with self.output_area:
            clear_output(wait=True)
            display(self._plot_slot)

    def set_filters(
        self,
        filters: dict | None = None,
        exclude: dict | None = None,
    ) -> None:
        """
        Programmatically update the metadata filters and exclusions and recompute.

        Since filters/exclude change the data subset, this triggers a full recomputation.
        """
        self.initial_filters = filters
        self.initial_exclude = exclude
        self._on_compute_clicked(None)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        _style = {"description_width": "70px"}
        _dropdown_layout = widgets.Layout(width="200px")

        # Embedding controls
        self.method_dropdown = widgets.Dropdown(
            options=["PCA", "t-SNE", "UMAP"],
            value="t-SNE",
            description="Method:",
            style=_style,
            layout=_dropdown_layout,
        )
        self.dim_toggle = widgets.Dropdown(
            options=[("2D", 2), ("3D", 3)],
            value=2,
            description="Dims:",
            style=_style,
            layout=widgets.Layout(width="130px"),
        )
        self.source_dropdown = widgets.Dropdown(
            options=[
                ("Embeddings", "embeddings"),
                ("Properties", "properties"),
                ("Combined", "combined"),
            ],
            value="embeddings",
            description="Source:",
            style=_style,
            layout=_dropdown_layout,
        )

        self._create_filter_exclude_widgets()

        # Appearance controls — fixed column width so pairs align across rows
        _ac = _INTERACTIVE_APPEAR_COL_W_PX
        _appear_pair_layout = widgets.Layout(
            width=f"{_ac}px",
            min_width=f"{_ac}px",
            max_width=f"{_ac}px",
            flex="0 0 auto",
        )
        _appear_desc = {"description_width": "70px"}
        self.color_dropdown = widgets.Dropdown(
            description="Color by:",
            style=_appear_desc,
            layout=_appear_pair_layout,
        )
        self.colorscale_dropdown = widgets.Dropdown(
            options=_CONTINUOUS_SCALES,
            value="Viridis",
            description="Palette:",
            style=_appear_desc,
            layout=_appear_pair_layout,
        )
        self.point_size_slider = widgets.IntSlider(
            value=8,
            min=1,
            max=20,
            step=1,
            description="Pt size:",
            style=_style,
            layout=_appear_pair_layout,
            continuous_update=False,
        )
        self.opacity_slider = widgets.FloatSlider(
            value=0.7,
            min=0.05,
            max=1.0,
            step=0.05,
            description="Opacity:",
            style=_style,
            layout=_appear_pair_layout,
            continuous_update=False,
        )

        # Highlight controls
        self._create_highlight_widgets()

        # Compute and status
        self.compute_button = widgets.Button(
            description="Compute",
            button_style="primary",
            icon="refresh",
            layout=widgets.Layout(width="120px"),
        )
        self.status_label = widgets.HTML(
            value=self._status_html("Ready", "info"),
            layout=widgets.Layout(flex="0 1 auto"),
        )
        self.stats_bar = widgets.HTML(
            value=self._stats_bar_html(0, 0, None, None),
            layout=widgets.Layout(flex="1 1 200px"),
        )

        # Image viewer
        self.show_extra_info_checkbox = widgets.Checkbox(
            value=True,
            description="Image extra info",
            indent=False,
            layout=widgets.Layout(
                width=f"{_INTERACTIVE_FIELD_MAX_W_PX}px",
                min_width="200px",
                max_width=f"{_INTERACTIVE_FIELD_MAX_W_PX}px",
            ),
        )

    @staticmethod
    def _apply_search_to_select_multiple(
        search_value: str,
        all_options: list[str],
        select_widget: widgets.SelectMultiple,
        *,
        select_default: bool = False,
    ) -> None:
        """Narrow ``select_widget`` options by substring match (shared Highlight + filter/exclude)."""
        q = (search_value or "").strip().lower()
        old_sel = tuple(select_widget.value)
        filtered = list(all_options) if not q else [o for o in all_options if q in o.lower()]
        select_widget.options = filtered
        if select_default and filtered:
            select_widget.value = (filtered[0],)
        elif filtered:
            prev = tuple(v for v in old_sel if v in filtered)
            if prev:
                select_widget.value = prev
            elif select_default:
                select_widget.value = (filtered[0],)
            elif not old_sel:
                select_widget.value = ()
            else:
                # Prior selection(s) no longer in filtered list: fall back like Highlight.
                select_widget.value = (filtered[0],)
        else:
            select_widget.value = ()

    def _create_filter_exclude_widgets(self) -> None:
        """Create embedding filter/exclude controls (same layout pattern as Highlight)."""
        _vh = _INTERACTIVE_VALUES_H
        _sw = _INTERACTIVE_SEARCH_W_PX
        _vw = _INTERACTIVE_VALUES_W_PX
        _fmax = _INTERACTIVE_FIELD_MAX_W_PX
        _fe_pair_layout = widgets.Layout(
            width="100%",
            gap="10px",
            flex_flow="row wrap",
            align_items="flex-start",
        )
        _fe_search_layout = widgets.Layout(
            width=f"{_sw}px",
            min_width=f"{_sw}px",
            max_width=f"{_sw}px",
            flex="0 0 auto",
        )
        _fe_values_layout = widgets.Layout(
            width=f"{_vw}px",
            min_width=f"{_vw}px",
            max_width=f"{_vw}px",
            height=f"{_vh}px",
            flex="0 0 auto",
        )
        self._filter_help = widgets.HTML(
            value=(
                '<div style="font-size:11px;color:#5D6D7E;line-height:1.45;">'
                "<b>Filter</b> keeps only rows matching any selected value (OR). "
                "<b>Exclude</b> removes rows matching any selected value. "
                "Press <b>Compute</b> to apply.</div>"
            ),
            layout=widgets.Layout(width="100%"),
        )
        self.filter_key_dropdown = widgets.Dropdown(
            description="Field:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(width="100%", min_width="0", max_width=f"{_fmax}px"),
        )
        self.filter_search_input = widgets.Text(
            value="",
            description="Search:",
            placeholder="Search values…",
            style=_INTERACTIVE_DESC_STYLE,
            layout=_fe_search_layout,
        )
        self.filter_value_select = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Values:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=_fe_values_layout,
        )
        self._filter_row_cat = widgets.HBox(
            [self.filter_search_input, self.filter_value_select],
            layout=_fe_pair_layout,
        )
        self._filter_include_block = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="font-size:10px;font-weight:600;color:#334155;letter-spacing:0.02em;'
                    'margin:0 0 2px 0;padding-bottom:4px;border-bottom:1px solid #E2E8F0;">'
                    "Include (filter)</div>"
                ),
                widgets.HBox(
                    [self.filter_key_dropdown],
                    layout=widgets.Layout(width="100%"),
                ),
                self._filter_row_cat,
            ],
            layout=widgets.Layout(width="100%", gap="8px"),
        )
        self.exclude_key_dropdown = widgets.Dropdown(
            description="Field:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(width="100%", min_width="0", max_width=f"{_fmax}px"),
        )
        self.exclude_search_input = widgets.Text(
            value="",
            description="Search:",
            placeholder="Search values…",
            style=_INTERACTIVE_DESC_STYLE,
            layout=_fe_search_layout,
        )
        self.exclude_value_select = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Values:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=_fe_values_layout,
        )
        self._exclude_row_cat = widgets.HBox(
            [self.exclude_search_input, self.exclude_value_select],
            layout=_fe_pair_layout,
        )
        self._filter_exclude_block = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="font-size:10px;font-weight:600;color:#334155;letter-spacing:0.02em;'
                    'margin:10px 0 2px 0;padding-bottom:4px;border-bottom:1px solid #E2E8F0;">'
                    "Exclude</div>"
                ),
                widgets.HBox(
                    [self.exclude_key_dropdown],
                    layout=widgets.Layout(width="100%"),
                ),
                self._exclude_row_cat,
            ],
            layout=widgets.Layout(width="100%", gap="8px"),
        )
        self._filter_exclude_actions_sep = widgets.HTML(
            value=_INTERACTIVE_SECTION_SEP_HTML,
            layout=widgets.Layout(width="100%"),
        )
        self.filter_clear_btn = widgets.Button(
            description="Clear filters",
            button_style="warning",
            icon="times",
            layout=widgets.Layout(width="130px"),
        )
        self._filter_exclude_panel = widgets.VBox(
            [
                self._filter_help,
                self._filter_include_block,
                self._filter_exclude_block,
                self._filter_exclude_actions_sep,
                widgets.HBox(
                    [self.filter_clear_btn],
                    layout=widgets.Layout(
                        width="100%",
                        padding="4px 0 0 70px",
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%", gap="10px", padding="10px 12px 12px 12px"),
        )
        self._embedding_filter_accordion = widgets.Accordion(
            children=[self._filter_exclude_panel],
            layout=widgets.Layout(width="100%"),
        )
        self._embedding_filter_accordion.set_title(0, "▽ Filter / exclude")
        self._embedding_filter_accordion.selected_index = None

    def _create_highlight_widgets(self) -> None:
        """Create highlight/filter widgets."""
        _values_h = _INTERACTIVE_VALUES_H
        _sw = _INTERACTIVE_SEARCH_W_PX
        _vw = _INTERACTIVE_VALUES_W_PX
        _fmax = _INTERACTIVE_FIELD_MAX_W_PX
        _hl_pair_layout = widgets.Layout(
            width="100%",
            gap="10px",
            flex_flow="row wrap",
            align_items="flex-start",
        )
        _hl_search_layout = widgets.Layout(
            width=f"{_sw}px",
            min_width=f"{_sw}px",
            max_width=f"{_sw}px",
            flex="0 0 auto",
        )
        _hl_values_layout = widgets.Layout(
            width=f"{_vw}px",
            min_width=f"{_vw}px",
            max_width=f"{_vw}px",
            height=f"{_values_h}px",
            flex="0 0 auto",
        )
        self._highlight_help = widgets.HTML(
            value=(
                '<div style="font-size:11px;color:#5D6D7E;line-height:1.45;">'
                "<b>Highlight</b> keeps every point visible and emphasises matches. "
                "Categorical: pick one or more values (OR). Numeric: set a range.</div>"
            ),
            layout=widgets.Layout(width="100%"),
        )
        self.highlight_key_dropdown = widgets.Dropdown(
            description="Field:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(width="100%", min_width="0", max_width=f"{_fmax}px"),
        )
        self.highlight_search_input = widgets.Text(
            value="",
            description="Search:",
            placeholder="Search values…",
            style=_INTERACTIVE_DESC_STYLE,
            layout=_hl_search_layout,
        )
        self.highlight_value_select = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Values:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=_hl_values_layout,
        )
        self.highlight_range_slider = widgets.FloatRangeSlider(
            value=(0.0, 1.0),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Range:",
            style=_INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(
                width=f"{_fmax}px",
                min_width="280px",
                max_width=f"{_fmax}px",
                flex="0 0 auto",
                display="none",
            ),
            continuous_update=False,
            readout=True,
            readout_format=".2f",
        )
        self._highlight_row_cat = widgets.HBox(
            [self.highlight_search_input, self.highlight_value_select],
            layout=_hl_pair_layout,
        )
        self._highlight_value_container = widgets.VBox(
            [self._highlight_row_cat, self.highlight_range_slider],
            layout=widgets.Layout(width="100%", gap="10px"),
        )
        self._highlight_field_is_numeric: bool = False

        self._highlight_actions_sep = widgets.HTML(
            value=_INTERACTIVE_SECTION_SEP_HTML,
            layout=widgets.Layout(width="100%"),
        )

        # Custom colors (``button_style`` would fight theme); light blue = start, light red = stop.
        self._highlight_btn_style_blue = widgets.ButtonStyle(
            button_color="#BFDBFE",
            text_color="#1E3A8A",
        )
        self._highlight_btn_style_stop = widgets.ButtonStyle(
            button_color="#FECACA",
            text_color="#991B1B",
        )
        self.highlight_toggle = widgets.Button(
            description="Highlight",
            tooltip="Apply group highlight on the plot",
            layout=widgets.Layout(width="108px", min_width="100px", max_width="118px"),
            style=self._highlight_btn_style_blue,
        )
        self.highlight_toggle.button_style = ""
        self._sync_highlight_button_appearance()

    def _setup_callbacks(self) -> None:
        """Setup widget callbacks."""
        self.compute_button.on_click(self._on_compute_clicked)
        self.color_dropdown.observe(self._on_color_changed, names="value")
        self.colorscale_dropdown.observe(self._on_colorscale_changed, names="value")
        self.filter_key_dropdown.observe(self._on_filter_key_changed, names="value")
        self.filter_search_input.observe(self._on_filter_search_changed, names="value")
        self.filter_value_select.observe(self._on_filter_value_changed, names="value")
        self.exclude_key_dropdown.observe(self._on_exclude_key_changed, names="value")
        self.exclude_search_input.observe(self._on_exclude_search_changed, names="value")
        self.exclude_value_select.observe(self._on_exclude_value_changed, names="value")
        self.filter_clear_btn.on_click(self._on_filter_clear_clicked)
        self.highlight_key_dropdown.observe(self._on_highlight_key_changed, names="value")
        self.highlight_search_input.observe(self._on_highlight_search_changed, names="value")
        self.highlight_value_select.observe(self._on_highlight_value_changed, names="value")
        self.highlight_range_slider.observe(self._on_highlight_value_changed, names="value")
        self.highlight_toggle.on_click(self._on_highlight_click)
        self.point_size_slider.observe(self._on_marker_style_changed, names="value")
        self.opacity_slider.observe(self._on_marker_style_changed, names="value")

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _status_html(msg: str, level: str = "info") -> str:
        """Return styled HTML status label.

        Args:
            msg: Status message text.
            level: Status level: 'info', 'ok', 'warn', or 'err'.

        Returns:
            HTML string for status label.
        """
        msg_safe = html.escape(str(msg))
        icons = {"info": "&#x2139;", "ok": "✓", "warn": "⚠", "err": "✕"}
        bg = {"info": "#EBF5FB", "ok": "#EAFAF1", "warn": "#FEF9E7", "err": "#FDEDEC"}
        border = {"info": "#3498db", "ok": "#27ae60", "warn": "#f39c12", "err": "#e74c3c"}
        color = {"info": "#1a5276", "ok": "#1e8449", "warn": "#9a7b0a", "err": "#922b21"}
        icon = icons.get(level, "&#x2139;")
        b = bg.get(level, "#F4F4F4")
        br = border.get(level, "#555")
        c = color.get(level, "#333")
        return (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f"padding:3px 10px;border-radius:12px;font-size:11px;"
            f"background:{b};border:1px solid {br};color:{c};"
            f'">{icon} {msg_safe}</span>'
        )

    @staticmethod
    def _stats_bar_html(
        n_total: int, n_highlighted: int, selected: int | None, error_msg: str | None = None
    ) -> str:
        """HTML for the live stats bar (totals / highlight / selection)."""
        sel = "—" if selected is None else html.escape(str(selected))
        err_span = (
            f'<span style="flex-basis:100%;color:#e74c3c;font-weight:600;">'
            f"{html.escape(error_msg)}</span>"
            if error_msg
            else ""
        )
        sep_after_sel = (
            '<div style="flex-basis:100%;height:1px;background:#E9ECEF;margin:4px 0 0 0;"></div>'
            if error_msg
            else ""
        )
        return (
            '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;'
            "padding:5px 10px;background:#F8F9FA;border-radius:6px;"
            'font-size:11px;color:#444;border:1px solid #E9ECEF;">'
            f"<span><b>Total</b> {n_total:,}</span>"
            '<span style="color:#ccc;">|</span>'
            f"<span><b>Highlighted</b> {n_highlighted:,}</span>"
            '<span style="color:#ccc;">|</span>'
            f"<span><b>Selected id</b> {sel}</span>"
            '<span style="color:#ccc;">|</span>'
            f"{sep_after_sel}"
            f"{err_span}"
            "</div>"
        )

    # ------------------------------------------------------------------
    # Option builders
    # ------------------------------------------------------------------
    def _update_color_options(self) -> None:
        """Populate color dropdown from available metadata and properties."""
        if not hasattr(self.pheno, "results") or not self.pheno.results:
            self.color_dropdown.options = []
            return

        cols: list[str] = []
        cols.extend(get_all_metadata_keys(self.pheno.results))
        if hasattr(self.pheno, "get_available_property_keys"):
            cols.extend(self.pheno.get_available_property_keys())
        cols = sorted(set(cols))
        self.color_dropdown.options = cols

        # Set sensible default
        for default in ["drug", "cluster", "treatment", "condition", "time"]:
            if default in cols:
                self.color_dropdown.value = default
                return
        if cols:
            self.color_dropdown.value = cols[0]

    def _update_highlight_key_options(self) -> None:
        """Populate highlight field dropdown from cached DataFrame or pipeline."""
        if self._cached_df is not None:
            # Use DataFrame columns (excludes coordinate cols)
            coord_prefixes = ("Component ",)
            exclude = {"Index", "Image"}
            opts = [
                c
                for c in self._cached_df.columns
                if c not in exclude and not any(c.startswith(p) for p in coord_prefixes)
            ]
        else:
            opts = list(self.color_dropdown.options)

        self.highlight_key_dropdown.options = sorted(opts) if opts else []
        if opts:
            self.highlight_key_dropdown.value = opts[0]

    def _update_filter_exclude_options(self) -> None:
        """Populate filter and exclude field dropdowns from available metadata."""
        if not hasattr(self.pheno, "results") or not self.pheno.results:
            self.filter_key_dropdown.options = []
            self.exclude_key_dropdown.options = []
            return

        cols: list[str] = []
        cols.extend(get_all_metadata_keys(self.pheno.results))
        cols = sorted(set(cols))

        self.filter_key_dropdown.options = cols
        self.exclude_key_dropdown.options = cols

        if cols:
            self.filter_key_dropdown.value = cols[0]
            self.exclude_key_dropdown.value = cols[0]
        self._update_filter_value_options(select_default=False)
        self._update_exclude_value_options(select_default=False)

    def _unique_metadata_values_for_key(self, key: str | None) -> list[str]:
        """Sorted unique string values for a metadata key across ``pheno.results``.

        ``PhenoMeResults`` stores ``metadata`` as a list of per-image dicts; we walk
        that list once and use :func:`get_metadata_value_from_dict` per row (avoids
        re-fetching the metadata list on every index as :func:`get_metadata_value` does).
        """
        if key is None or not hasattr(self.pheno, "results") or not self.pheno.results:
            return []
        results = self.pheno.results
        if hasattr(results, "metadata"):
            metadata_list = results.metadata
        else:
            metadata_list = list(results.get("metadata", []))
        all_vals: list[str] = []
        for meta in metadata_list:
            if isinstance(meta, dict):
                v = get_metadata_value_from_dict(meta, key)
                if v is not None:
                    all_vals.append(str(v))
        return sorted(set(all_vals), key=str)

    def _sync_embed_filters_from_widgets(self) -> None:
        """Set ``initial_filters`` / ``initial_exclude`` from current filter widgets."""
        fk = self.filter_key_dropdown.value
        fv = list(self.filter_value_select.value)
        self.initial_filters = None if fk is None or not fv else {fk: fv}
        ek = self.exclude_key_dropdown.value
        ev = list(self.exclude_value_select.value)
        self.initial_exclude = None if ek is None or not ev else {ek: ev}

    def _apply_filter_search_filter(self, *, select_default: bool = False) -> None:
        """Filter ``filter_value_select`` by ``filter_search_input``; sync pipeline dict."""
        self._apply_search_to_select_multiple(
            self.filter_search_input.value,
            self._filter_value_options_all,
            self.filter_value_select,
            select_default=select_default,
        )
        self._sync_embed_filters_from_widgets()

    def _apply_exclude_search_filter(self, *, select_default: bool = False) -> None:
        """Filter ``exclude_value_select`` by ``exclude_search_input``; sync pipeline dict."""
        self._apply_search_to_select_multiple(
            self.exclude_search_input.value,
            self._exclude_value_options_all,
            self.exclude_value_select,
            select_default=select_default,
        )
        self._sync_embed_filters_from_widgets()

    def _update_slot_value_options(
        self,
        key_dropdown: widgets.Dropdown,
        value_select: widgets.SelectMultiple,
        search_input: widgets.Text,
        options_attr: str,
        apply_search_fn: Callable[..., None],
        *,
        select_default: bool = True,
    ) -> None:
        """Populate filter or exclude value list for the selected metadata key."""
        key = key_dropdown.value
        if key is None or not hasattr(self.pheno, "results") or not self.pheno.results:
            setattr(self, options_attr, [])
            value_select.options = []
            value_select.value = ()
            search_input.value = ""
            self._sync_embed_filters_from_widgets()
            return

        try:
            unique_vals = self._unique_metadata_values_for_key(key)
            setattr(self, options_attr, unique_vals)
            search_input.value = ""
            apply_search_fn(select_default=select_default)
        except (AttributeError, KeyError, TypeError):
            setattr(self, options_attr, [])
            value_select.options = []
            value_select.value = ()
            self._sync_embed_filters_from_widgets()

    def _update_filter_value_options(self, *, select_default: bool = True) -> None:
        """Populate filter value list for the selected key (same pattern as Highlight)."""
        self._update_slot_value_options(
            self.filter_key_dropdown,
            self.filter_value_select,
            self.filter_search_input,
            "_filter_value_options_all",
            self._apply_filter_search_filter,
            select_default=select_default,
        )

    def _update_exclude_value_options(self, *, select_default: bool = True) -> None:
        """Populate exclude value list for the selected key."""
        self._update_slot_value_options(
            self.exclude_key_dropdown,
            self.exclude_value_select,
            self.exclude_search_input,
            "_exclude_value_options_all",
            self._apply_exclude_search_filter,
            select_default=select_default,
        )

    def _show_dropdown(self) -> None:
        """Show categorical search + multi-select, hide the range slider."""
        self._highlight_row_cat.layout.display = ""
        self.highlight_search_input.layout.display = ""
        self.highlight_value_select.layout.display = ""
        self.highlight_range_slider.layout.display = "none"

    def _show_range_slider(self) -> None:
        """Show the range slider, hide categorical widgets."""
        self._highlight_row_cat.layout.display = "none"
        self.highlight_range_slider.layout.display = ""

    def _update_highlight_value_options(self) -> None:
        """Populate the value widget for the currently selected highlight key.

        - **Categorical columns** → show the dropdown with unique values.
        - **Integer columns with few distinct values** (e.g. cluster id) → categorical
          multi-select, aligned with Filter/Exclude.
        - **Continuous numeric columns** → range slider spanning [min, max].
        """
        key = self.highlight_key_dropdown.value
        if key is None or self._cached_df is None or key not in self._cached_df.columns:
            self._highlight_value_options_all = []
            self.highlight_value_select.options = []
            self.highlight_value_select.value = ()
            self.highlight_search_input.value = ""
            self._highlight_field_is_numeric = False
            self._show_dropdown()
            return

        col = self._cached_df[key]

        # Treat boolean columns as categorical
        is_bool = pd.api.types.is_bool_dtype(col)
        is_numeric = pd.api.types.is_numeric_dtype(col) and not is_bool
        n_unique_non_na = int(col.dropna().nunique()) if is_numeric else 0
        # Cluster-like integer columns: multi-select, not a range slider (Filter/Exclude
        # already expose these as discrete string tokens).
        discrete_int_categorical = (
            is_numeric
            and pd.api.types.is_integer_dtype(col)
            and n_unique_non_na <= _HIGHLIGHT_DISCRETE_INT_MAX_UNIQUES
        )

        if is_numeric and not discrete_int_categorical:
            self._highlight_field_is_numeric = True
            self._show_range_slider()
            self.highlight_search_input.value = ""
            col_clean = col.dropna()
            if len(col_clean) == 0:
                return
            col_min = float(col_clean.min())
            col_max = float(col_clean.max())
            # Guard against min == max (constant column)
            if col_min == col_max:
                col_max = col_min + 1.0

            # Choose a sensible step size (~200 steps across the range)
            span = col_max - col_min
            step = round(span / 200, 6) if span > 0 else 0.01

            # Determine readout format based on magnitude (more precision for small values)
            if span >= 100:
                readout_fmt = ".1f"
            elif span >= 1:
                readout_fmt = ".2f"
            elif span >= 0.01:
                readout_fmt = ".4f"
            else:
                readout_fmt = ".6f"

            # Update slider bounds.  Order matters because traitlets
            # enforces min <= value <= max at every step.  Widen first,
            # then narrow, so the intermediate state is always valid.
            slider = self.highlight_range_slider
            slider.min = min(col_min, slider.min)
            slider.max = max(col_max, slider.max)
            slider.step = step
            slider.value = (col_min, col_max)
            slider.min = col_min
            slider.max = col_max
            slider.readout_format = readout_fmt
        else:
            self._highlight_field_is_numeric = False
            self._show_dropdown()
            unique_vals = sorted(col.dropna().unique(), key=str)
            self._highlight_value_options_all = [str(v) for v in unique_vals]
            self.highlight_search_input.value = ""
            self._apply_highlight_search_filter(select_default=True)

    def _apply_highlight_search_filter(self, *, select_default: bool = False) -> None:
        """Filter ``highlight_value_select`` options by ``highlight_search_input``."""
        self._apply_search_to_select_multiple(
            self.highlight_search_input.value,
            self._highlight_value_options_all,
            self.highlight_value_select,
            select_default=select_default,
        )

    def _on_highlight_search_changed(self, _change: Any) -> None:
        """Live-filter categorical highlight values (debounced)."""
        if self._highlight_field_is_numeric:
            return

        if self._debounce_timer is not None:
            self._debounce_timer.cancel()

        def _do_search() -> None:
            self._apply_highlight_search_filter(select_default=False)
            if self._highlight_active:
                # Apply highlight styling from main thread event loop
                _schedule_after_plotly_event_loop(self._apply_highlight)
                _schedule_after_plotly_event_loop(self._update_highlight_status)

        self._debounce_timer = threading.Timer(_SEARCH_DEBOUNCE_SEC, _do_search)
        self._debounce_timer.start()

    # ------------------------------------------------------------------
    # Compute (expensive - dimensionality reduction)
    # ------------------------------------------------------------------
    def _compute_embedding(self) -> float:
        """Run dimensionality reduction and populate self._cached_df."""
        method_label = self.method_dropdown.value  # 'PCA', 't-SNE', 'UMAP'
        method_key = method_label.lower().replace("-", "")  # 'pca', 'tsne', 'umap'
        n_dims = self.dim_toggle.value
        source = self.source_dropdown.value

        t0 = time.time()
        df, _feature_type, dr_obj = run_dimensionality_reduction(
            self.pheno,
            method=method_key,
            n_components=n_dims,
            source=source,
            filters=self.initial_filters,
            exclude=self.initial_exclude,
            device=self.pheno.device,
            use_gpu=getattr(self.pheno, "use_gpu_for_dr", True),
        )
        elapsed = time.time() - t0

        if df is None:
            raise ValueError("Dimensionality reduction returned no data.")

        if len(df) > self.max_points:
            df = df.sample(n=self.max_points, random_state=_DR_RANDOM_STATE).copy()

        self._cached_df = df
        self._cached_method = method_label
        self._cached_source = source
        self._cached_ndims = n_dims
        self._cached_filters = self.initial_filters
        self._cached_exclude = self.initial_exclude
        self._cached_dr_obj = dr_obj

        # Refresh highlight options now that we have a new DataFrame
        self._update_highlight_key_options()
        self._update_highlight_value_options()

        self._refresh_index_row_map()
        self._compute_seq += 1

        return elapsed

    def _refresh_index_row_map(self) -> None:
        """Build ``Index`` column -> row index map for O(1) lookups (highlight / overlay)."""
        df = self._cached_df
        if df is None or "Index" not in df.columns:
            self._cached_index_to_row = None
            return
        idx_arr = df["Index"].to_numpy()
        # First occurrence wins (same as ``df.loc[df['Index']==i].iloc[0]``).
        self._cached_index_to_row = {}
        for j, v in enumerate(idx_arr):
            iv = int(v)
            if iv not in self._cached_index_to_row:
                self._cached_index_to_row[iv] = j

    # ------------------------------------------------------------------
    # Figure building (uses cached DF - fast)
    # ------------------------------------------------------------------
    def _coord_columns(self) -> tuple[str, str, str | None]:
        """Return (x_col, y_col, z_col) based on cached dims. All methods use Component 1/2/3."""
        x, y = "Component 1", "Component 2"
        z = "Component 3" if self._cached_ndims == 3 else None
        return x, y, z

    def _get_data_traces(self) -> list[Any]:
        """Return plotly express data traces only (exclude selection overlay)."""
        if self.fig_widget is None or self._n_data_traces <= 0:
            return []
        return list(self.fig_widget.data[: self._n_data_traces])

    def _selection_overlay_trace(self) -> Any | None:
        """The extra trace used for click-to-select border (2D or 3D)."""
        if self.fig_widget is None or self._n_data_traces <= 0:
            return None
        if len(self.fig_widget.data) <= self._n_data_traces:
            return None
        return self.fig_widget.data[self._n_data_traces]

    def _highlight_marker_size(self) -> int:
        """Marker size for selection overlay / highlight emphasis vs base point size."""
        ns = self.point_size_slider.value
        return max(ns + 5, int(ns * 2))

    def _make_selection_overlay_trace(self, is_3d: bool) -> go.Scatter3d | go.Scatter:
        """Single extra trace for click-to-select border (WebGL traces lack per-point line width)."""
        hs = self._highlight_marker_size()
        marker = {
            "size": hs,
            "color": "rgba(0,0,0,0)",
            "opacity": 1.0,
            "line": {"color": "black", "width": 2},
        }
        if is_3d:
            return go.Scatter3d(
                name=_SELECTION_OVERLAY_NAME,
                x=[],
                y=[],
                z=[],
                mode="markers",
                showlegend=False,
                hoverinfo="skip",
                visible=False,
                marker=marker,
                customdata=[],
            )
        return go.Scatter(
            name=_SELECTION_OVERLAY_NAME,
            x=[],
            y=[],
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            visible=False,
            marker=marker,
            customdata=[],
        )

    def _apply_uniform_data_traces(self) -> None:
        """Uniform marker size/opacity on all data traces (no highlight, no selection on data)."""
        n_tr = len(self._get_data_traces())
        if n_tr == 0:
            return
        normal_size = self.point_size_slider.value
        normal_opacity = self.opacity_slider.value
        self._apply_marker_style_to_traces(
            [normal_size] * n_tr,
            [normal_opacity] * n_tr,
            [0] * n_tr,
        )

    def _build_figure(self) -> None:
        """Build a new FigureWidget from the cached DataFrame."""
        df = self._cached_df
        if df is None:
            return
        if len(df) == 0:
            # No data - show empty placeholder and clear selection
            self._selected_point_index = None
            self._n_data_traces = 0
            self.fig_widget = go.FigureWidget(
                layout=go.Layout(
                    template="plotly_white",
                    autosize=False,
                    width=_EMBEDDING_FIG_WIDTH_PX,
                    height=_EMBEDDING_FIG_HEIGHT_PX,
                    title={
                        "text": "<b>No data</b>",
                        "x": 0.5,
                        "xanchor": "center",
                        "y": 0.98,
                        "yanchor": "top",
                        "font": {
                            "size": 17,
                            "family": "Inter, Helvetica Neue, Arial, sans-serif",
                            "color": "#141428",
                        },
                    },
                    paper_bgcolor="white",
                    plot_bgcolor="#FAFCFE",
                    xaxis={"visible": False},
                    yaxis={"visible": False},
                    annotations=[{"text": "Compute embedding to see data", "showarrow": False}],
                    margin={"l": 50, "r": 50, "t": 72, "b": 50},
                )
            )
            apply_figurewidget_display_config(self.fig_widget)
            return
        x_col, y_col, z_col = self._coord_columns()
        color_by = self.color_dropdown.value

        # Determine colour column
        color_column: str | None = None
        is_continuous = False
        if color_by:
            try:
                color_column, is_continuous = self.pheno._get_color_column(color_by, df)
            except ValueError:
                color_column = None

        # Hover columns
        coord_cols = {x_col, y_col, z_col}
        hover_cols = ["Index"]
        if "Image" in df.columns:
            hover_cols.append("Image")
        if self.hover_features:
            for feat in self.hover_features:
                col = next((c for c in df.columns if c.lower() == feat.lower()), None)
                if col and col not in coord_cols and col not in hover_cols:
                    hover_cols.append(col)
        elif color_column and color_column not in hover_cols:
            hover_cols.append(color_column)

        kwargs: dict[str, Any] = {
            "data_frame": df,
            "x": x_col,
            "y": y_col,
            "hover_data": hover_cols,
            # Ensure pipeline image index is always the first customdata column for clicks.
            "custom_data": ["Index"],
            "width": _EMBEDDING_FIG_WIDTH_PX,
            "height": _EMBEDDING_FIG_HEIGHT_PX,
        }
        if z_col:
            kwargs["z"] = z_col
        # 2D defaults to WebGL (``scattergl``) for performance; ``_resolve_click_point_index``
        # falls back to nearest x/y when ``point_inds`` is empty.

        if color_column:
            kwargs["color"] = color_column
            if is_continuous:
                kwargs["color_continuous_scale"] = self.colorscale_dropdown.value
            else:
                kwargs["category_orders"] = {
                    color_column: sorted(df[color_column].dropna().unique(), key=str)
                }
                kwargs["color_discrete_sequence"] = _QUALITATIVE_PALETTE

        fig = px.scatter_3d(**kwargs) if z_col else px.scatter(**kwargs)

        # Labelled hover (customdata column order matches ``hover_cols`` / hover_data)
        if hover_cols:
            ht_parts = [f"{col}=%{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]
            fig.update_traces(hovertemplate="<br>".join(ht_parts) + "<extra></extra>")

        # Marker styling (no outline in baseline — highlight/selection add borders)
        marker_size = self.point_size_slider.value
        opacity = self.opacity_slider.value
        fig.update_traces(marker={"size": marker_size, "opacity": opacity, "line": {"width": 0}})
        if not color_column:
            fig.update_traces(marker_color=_QUALITATIVE_PALETTE[0])

        _apply_hoverlabels_matching_markers(
            fig,
            color_column=color_column,
            is_continuous=is_continuous,
            colorscale_name=self.colorscale_dropdown.value,
            fallback_uniform=_QUALITATIVE_PALETTE[0],
        )

        _font = {"family": "Inter, Helvetica Neue, Arial, sans-serif", "size": 12}
        _axis_2d = {
            "showline": False,
            "zeroline": False,
            "gridcolor": "#EBEBEB",
            "ticks": "",
        }
        # Per-trace hoverlabel bgcolor is set in `_apply_hoverlabels_matching_markers`
        # so each point matches its marker color; layout only sets shared defaults.
        _hoverlabel = {
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        }
        _legend = {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.14,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(248,250,252,0.92)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
            "font": {"size": 10, "family": _font["family"], "color": "#334155"},
            "itemsizing": "constant",
            "tracegroupgap": 8,
            "itemwidth": 30,
        }
        if color_column and not is_continuous:
            _legend["title"] = {
                "text": color_column,
                "side": "top center",
                "font": {"size": 11, "family": _font["family"], "color": "#334155"},
            }

        source_label = self._cached_source if self._cached_source else "?"
        title_main = f"{self._cached_method} ({source_label})"
        title_html = (
            f'<span style="font-size:17px;font-weight:600;letter-spacing:-0.02em;color:#141428;">'
            f"{html.escape(title_main)}</span>"
        )
        if color_by:
            title_html += (
                f'<br><span style="font-size:12px;font-weight:500;color:#64748b;">'
                f'Colored by <span style="color:#334155;">{html.escape(str(color_by))}</span></span>'
            )
        _title_layout = {
            "text": title_html,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"t": 6},
            "font": {"family": _font["family"], "size": 14, "color": "#141428"},
        }

        # Axis titles (PCA: show explained variance)
        x_title, y_title, z_title = x_col, y_col, z_col
        if self._cached_method == "PCA" and self._cached_dr_obj is not None:
            pca = self._cached_dr_obj
            x_title = f"Component 1 ({pca.explained_variance_ratio_[0]:.1%})"
            y_title = f"Component 2 ({pca.explained_variance_ratio_[1]:.1%})"
            if z_col:
                z_title = f"Component 3 ({pca.explained_variance_ratio_[2]:.1%})"

        if z_col:
            _scene_axis = {
                "showbackground": True,
                "backgroundcolor": "#FAFCFE",
                "gridcolor": "#E8ECF0",
                "showline": False,
                "zeroline": False,
                "ticks": "",
            }
            fig.update_layout(
                template="plotly_white",
                font=_font,
                paper_bgcolor="white",
                hovermode="closest",
                autosize=False,
                title=_title_layout,
                margin={"l": 50, "r": 50, "t": 56, "b": 124},
                hoverlabel=_hoverlabel,
                legend=_legend,
                scene={
                    "bgcolor": "#FAFCFE",
                    "xaxis": {**_scene_axis, "title": x_title},
                    "yaxis": {**_scene_axis, "title": y_title},
                    "zaxis": {**_scene_axis, "title": z_title},
                },
            )
        else:
            fig.update_layout(
                template="plotly_white",
                font=_font,
                paper_bgcolor="white",
                plot_bgcolor="#FAFCFE",
                hovermode="closest",
                autosize=False,
                xaxis={**_axis_2d, "title": x_title},
                yaxis={**_axis_2d, "title": y_title},
                title=_title_layout,
                margin={"l": 50, "r": 50, "t": 56, "b": 124},
                hoverlabel=_hoverlabel,
                legend=_legend,
            )

        # Continuous colour: horizontal colorbar under the plot (shared coloraxis for 2D/3D)
        if is_continuous and color_column:
            fig.update_layout(
                coloraxis={
                    "colorbar": {
                        "orientation": "h",
                        "yanchor": "top",
                        "y": -0.10,
                        "xanchor": "center",
                        "x": 0.5,
                        "lenmode": "fraction",
                        "len": 0.72,
                        "thickness": 16,
                        "title": {"side": "bottom", "font": {"size": 12, "color": "#334155"}},
                        "tickfont": {"size": 10, "color": "#64748b"},
                        "ticklen": 5,
                        "tickcolor": "#cbd5e1",
                        "outlinewidth": 1,
                        "outlinecolor": "#e2e8f0",
                    }
                },
                margin_b=max(fig.layout.margin.b or 124, 152),
            )

        fig.update_layout(
            autosize=False,
            width=_EMBEDDING_FIG_WIDTH_PX,
            height=_EMBEDDING_FIG_HEIGHT_PX,
        )
        # Convert to FigureWidget (strip numpy from trace/layout for widget sync)
        self.fig_widget = go.FigureWidget(_figurewidget_safe_figure(fig))
        apply_figurewidget_display_config(self.fig_widget)
        self._n_data_traces = len(self.fig_widget.data)
        self._current_color_by = color_by

        # Selection overlay: scattergl/scatter3d do not support per-point line width; use a
        # dedicated trace so only one point shows a border (see module docstring).
        self.fig_widget.add_trace(self._make_selection_overlay_trace(is_3d=bool(z_col)))

        # Apply mode-specific styling
        if self._highlight_active:
            self._apply_highlight()

        # Selected point highlight (click-to-select)
        if self._selected_point_index is not None:
            self._apply_selected_point_highlight()

        # Click callback for image inspection and point selection
        if self.fig_widget is not None:
            for trace in self.fig_widget.data:
                trace.on_click(self._on_figure_click)

    # ------------------------------------------------------------------
    # Instant colour update (no recomputation)
    # ------------------------------------------------------------------
    def _try_patch_continuous_recolor(self, color_column: str) -> bool:
        """Update continuous coloring in-place (preserves zoom/pan).

        Plotly express uses one data trace + ``layout.coloraxis`` for continuous
        colour. Categorical plots use one trace per category; those must still
        go through :meth:`_build_figure`.

        Returns:
            True if the figure was updated without a full rebuild.
        """
        if self.fig_widget is None or self._cached_df is None:
            return False
        df = self._cached_df
        if color_column not in df.columns:
            return False
        traces = self._get_data_traces()
        if len(traces) != 1:
            return False
        tr = traces[0]
        vals = df[color_column]
        try:
            with self.fig_widget.batch_update():
                tr.marker.color = vals
                tr.marker.coloraxis = "coloraxis"
                self.fig_widget.layout.coloraxis.colorscale = self.colorscale_dropdown.value
                cbar = getattr(self.fig_widget.layout.coloraxis, "colorbar", None)
                if cbar is not None and getattr(cbar, "title", None) is not None:
                    cbar.title.text = color_column
                _set_trace_hoverlabel_continuous(tr, self.colorscale_dropdown.value)
        except Exception:
            logger.debug(
                "Continuous recolor patch failed; falling back to full rebuild.",
                exc_info=True,
            )
            return False
        return True

    def _recolor_figure(self) -> None:
        """Re-colour the existing figure without recomputation."""
        if self.fig_widget is None or self._cached_df is None:
            return

        color_by = self.color_dropdown.value
        df = self._cached_df

        color_column: str | None = None
        is_continuous = False
        if color_by:
            try:
                color_column, is_continuous = self.pheno._get_color_column(color_by, df)
            except ValueError:
                color_column = None

        if color_column is None:
            # Single colour fallback
            u = _QUALITATIVE_PALETTE[0]
            with self.fig_widget.batch_update():
                for trace in self._get_data_traces():
                    trace.marker.color = u
                    _set_trace_hoverlabel_uniform(trace, u)
            self._current_color_by = color_by
            return

        if is_continuous and self._try_patch_continuous_recolor(color_column):
            self._current_color_by = color_by
            return

        # Categorical (multi-trace) or patch failed: rebuild legend / traces.
        self._build_figure()
        self._display_figure()
        self._current_color_by = color_by

    # ------------------------------------------------------------------
    # Highlight logic
    # ------------------------------------------------------------------
    @staticmethod
    def _raw_index_from_customdata_row(cd: Any) -> int:
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

    def _resolve_click_point_index(self, trace: Any, points: Any) -> int | None:
        """Marker index within ``trace`` for a click (handles empty ``point_inds`` on WebGL).

        Expected shapes:
            ``trace.x`` / ``trace.y``: length ``n_points`` coordinate arrays.
            ``points.xs`` / ``points.ys``: click coordinates from the frontend (at least one point).
        """
        if points is None:
            return None
        if hasattr(points, "point_inds") and points.point_inds:
            return int(points.point_inds[0])

        ttype = getattr(trace, "type", "")
        # ``scatter3d`` also uses WebGL; ``point_inds`` may be empty like ``scattergl``.
        if ttype not in ("scatter", "scattergl", "scatter3d"):
            return None

        # Extract click coords from points object immediately (points may be transient)
        if not hasattr(points, "xs") or not points.xs or not hasattr(points, "ys") or not points.ys:
            return None

        try:
            x0, y0 = float(points.xs[0]), float(points.ys[0])
            # Use cached coordinates if possible (faster than fetching from trace on every click)
            tx = np.asarray(trace.x, dtype=float)
            ty = np.asarray(trace.y, dtype=float)
            if tx.size == 0 or ty.size == 0 or tx.shape != ty.shape:
                return None
            # 3D: nearest point in x,y,z when click provides z (WebGL may omit point_inds)
            if ttype == "scatter3d":
                tz = np.asarray(getattr(trace, "z", None), dtype=float)
                if tz.size == tx.size and hasattr(points, "zs") and points.zs:
                    z0 = float(points.zs[0])
                    d = (tx - x0) ** 2 + (ty - y0) ** 2 + (tz - z0) ** 2
                    return int(np.argmin(d))
            return int(np.argmin((tx - x0) ** 2 + (ty - y0) ** 2))
        except (TypeError, ValueError, IndexError, AttributeError):
            return None

    def _on_figure_click(self, trace: Any, points: Any, _state: Any) -> None:
        """Handle FigureWidget clicks - scheduled via event loop to avoid hangs."""
        # Fast exit if busy or too soon (simple debouncing)
        t_now = time.time()
        if t_now - self._last_click_time < _CLICK_DEBOUNCE_SEC:
            return
        if not self._click_lock.acquire(blocking=False):
            return

        scheduled = False
        try:
            # Skip clicks on the selection overlay trace itself
            if getattr(trace, "name", "") == _SELECTION_OVERLAY_NAME:
                return

            point_idx = self._resolve_click_point_index(trace, points)
            if point_idx is None:
                return

            try:
                if trace.customdata is None or point_idx >= len(trace.customdata):
                    return
                cdata = trace.customdata[point_idx]
                idx = self._raw_index_from_customdata_row(cdata)
                if idx < 0:
                    return
            except (TypeError, ValueError, IndexError):
                return

            # Update click time for debouncing
            self._last_click_time = t_now

            def _apply_click_safe() -> None:
                try:
                    self._image_error_msg = None
                    # Toggle selected point: same click = deselect, different = select
                    if self._selected_point_index == idx:
                        self._selected_point_index = None
                    else:
                        self._selected_point_index = idx

                    # Update plot styling
                    self._apply_selected_point_highlight()

                    if self._selected_point_index is None:
                        self._set_image_panel_idle()
                        self._update_stats()
                        return

                    sel_idx = self._selected_point_index
                    self.img_output.children = (self._img_loading_placeholder,)
                    seq_at_load = self._compute_seq

                    def _load_and_show_image_task() -> None:
                        try:
                            # Track which point this thread is loading for
                            loading_idx = sel_idx
                            png, details_text = self.pheno.image_preview_png_bytes(
                                loading_idx,
                                apply_transforms=False,
                                figsize=_IMAGE_OVERLAY_FIGSIZE,
                                downsample=max(360, _IMAGE_OVERLAY_WIDTH_PX * 2),
                                dpi=100,
                                show_extra_info=self.show_extra_info_checkbox.value,
                            )

                            # Create widgets on main thread loop via _schedule_after_plotly_event_loop
                            def _update_ui() -> None:
                                # Skip if embedding was recomputed or selection moved
                                if self._compute_seq != seq_at_load:
                                    return
                                if self._selected_point_index != loading_idx:
                                    return

                                preview = widgets.Image(
                                    value=png,
                                    format="png",
                                    layout=widgets.Layout(
                                        width="100%", max_width="100%", height="auto"
                                    ),
                                )
                                if details_text:
                                    extra = widgets.HTML(
                                        value=(
                                            '<div style="margin-top:8px;font-size:13px;line-height:1.45;'
                                            "color:#333;max-height:260px;overflow-y:auto;border-top:1px solid #E0E0E0;"
                                            'padding-top:6px;">'
                                            '<pre style="margin:0;font-size:13px;white-space:pre-wrap;word-break:break-word;">'
                                            f"{html.escape(details_text)}</pre></div>"
                                        ),
                                        layout=widgets.Layout(width="100%"),
                                    )
                                    self.img_output.children = (preview, extra)
                                else:
                                    self.img_output.children = (preview,)
                                self._update_stats()

                            _schedule_after_plotly_event_loop(_update_ui)
                        except Exception as e:
                            # Bind for the deferred callback: `e` is cleared when the except block ends.
                            image_load_err = e

                            def _update_err() -> None:
                                if self._compute_seq != seq_at_load:
                                    return
                                if self._selected_point_index != sel_idx:
                                    return
                                self._set_image_panel_idle()
                                logger.error(
                                    "Error displaying image for index %s: %s",
                                    sel_idx,
                                    image_load_err,
                                )
                                self._image_error_msg = (
                                    f"⚠ Error loading image: {type(image_load_err).__name__}"
                                )
                                self._update_stats()

                            _schedule_after_plotly_event_loop(_update_err)

                    # Offload image loading to a background thread
                    threading.Thread(target=_load_and_show_image_task, daemon=True).start()

                finally:
                    # Always release the lock via the event loop task
                    self._click_lock.release()

            # Schedule UI work after Plotly's event handling
            _schedule_after_plotly_event_loop(_apply_click_safe)
            scheduled = True

        except Exception:
            raise
        finally:
            if not scheduled:
                self._click_lock.release()

    def _get_per_trace_masks(self, row_mask: np.ndarray) -> list[np.ndarray]:
        """Map a boolean mask over cached DataFrame rows to per-trace boolean arrays."""
        df = self._cached_df
        if df is None or self.fig_widget is None:
            return []
        if self._cached_index_to_row is None:
            self._refresh_index_row_map()
        df_row_map = self._cached_index_to_row
        if df_row_map is None:
            return []
        trace_masks: list[np.ndarray] = []
        for trace in self._get_data_traces():
            n_trace = len(trace.x) if hasattr(trace, "x") and trace.x is not None else 0
            if n_trace == 0:
                trace_masks.append(np.zeros(0, dtype=bool))
                continue
            if trace.customdata is not None and len(trace.customdata) > 0:
                try:
                    trace_indices = [
                        self._raw_index_from_customdata_row(cd) for cd in trace.customdata
                    ]
                    tm = np.array(
                        [
                            bool(row_mask[df_row_map[ti]]) if ti in df_row_map else False
                            for ti in trace_indices
                        ]
                    )
                except (TypeError, ValueError, KeyError, IndexError):
                    tm = np.zeros(n_trace, dtype=bool)
            else:
                tm = np.zeros(n_trace, dtype=bool)
            trace_masks.append(tm)
        return trace_masks

    def _apply_marker_style_to_traces(
        self,
        per_trace_sizes: list[Any],
        per_trace_opacity: list[Any],
        per_trace_line_width: list[Any],
    ) -> None:
        """Apply marker size, opacity, and line width per data trace (scalar or per-point list).

        Does not modify the selection overlay trace; use :meth:`_selection_overlay_trace` for that.
        """
        if self.fig_widget is None:
            return
        traces = self._get_data_traces()
        if len(per_trace_sizes) != len(traces):
            logger.warning(
                "Marker style list length %s != data trace count %s",
                len(per_trace_sizes),
                len(traces),
            )
            return
        with self.fig_widget.batch_update():
            for trace, sz, op, lw in zip(
                traces, per_trace_sizes, per_trace_opacity, per_trace_line_width, strict=True
            ):
                trace.marker.size = sz
                trace.marker.opacity = op
                trace.marker.line = {"width": lw, "color": "black"}

    def _update_stats(self) -> None:
        """Refresh the stats bar (total / highlighted / selected index)."""
        if self._cached_df is None:
            self.stats_bar.value = self._stats_bar_html(
                0, 0, self._selected_point_index, getattr(self, "_image_error_msg", None)
            )
            return
        n_total = len(self._cached_df)
        n_hl = 0
        if self._highlight_active:
            mask = self._build_highlight_mask()
            if mask is not None:
                n_hl = int(mask.sum())
        self.stats_bar.value = self._stats_bar_html(
            n_total, n_hl, self._selected_point_index, getattr(self, "_image_error_msg", None)
        )

    def _build_highlight_mask(self) -> np.ndarray | None:
        """Return a boolean mask over the cached DataFrame rows, or None.

        Shape:
            ``(n_rows,)`` bool, aligned with ``self._cached_df`` row order.
        """
        key = self.highlight_key_dropdown.value
        if key is None or self._cached_df is None or key not in self._cached_df.columns:
            return None

        col_vals = self._cached_df[key]

        if self._highlight_field_is_numeric:
            # Range-based highlight
            lo, hi = self.highlight_range_slider.value
            numeric_vals = pd.to_numeric(col_vals, errors="coerce")
            return ((numeric_vals >= lo) & (numeric_vals <= hi)).values
        # Multi-value categorical (OR); widget tokens are strings — align column values as str once.
        vals = self.highlight_value_select.value
        if not vals:
            return np.zeros(len(col_vals), dtype=bool)
        val_list = list(vals)
        if pd.api.types.is_string_dtype(col_vals.dtype):
            return col_vals.isin(val_list).values
        if isinstance(col_vals.dtype, pd.CategoricalDtype):
            return col_vals.astype(str).isin(val_list).values
        return col_vals.astype(str).isin(val_list).values

    def _apply_highlight(self) -> None:
        """Apply or remove highlight styling based on widget state.

        Uses per-point sizes (supported by scattergl/scatter3d) and **trace-level** opacity
        (per-point opacity is not reliable on WebGL traces). No borders on data traces.
        """
        if self.fig_widget is None or self._cached_df is None:
            self._update_stats()
            return

        data_traces = self._get_data_traces()
        n_tr = len(data_traces)

        if not self._highlight_active:
            normal_size = self.point_size_slider.value
            normal_opacity = self.opacity_slider.value
            self._apply_marker_style_to_traces(
                [normal_size] * n_tr,
                [normal_opacity] * n_tr,
                [0] * n_tr,
            )
            self._update_stats()
            return

        highlight_mask = self._build_highlight_mask()
        if highlight_mask is None:
            self._update_stats()
            return

        trace_masks = self._get_per_trace_masks(highlight_mask)
        normal_size = self.point_size_slider.value
        highlight_size = self._highlight_marker_size()
        normal_opacity = self.opacity_slider.value
        dim_opacity = max(0.08, normal_opacity * 0.15)

        sizes_l: list[Any] = []
        op_l: list[Any] = []
        lw_l: list[Any] = []
        for _trace, tm in zip(data_traces, trace_masks, strict=True):
            n_trace = len(tm)
            if n_trace == 0:
                sizes_l.append(normal_size)
                op_l.append(normal_opacity)
                lw_l.append(0)
                continue
            has_any = bool(np.any(tm))
            sizes = np.where(tm, highlight_size, normal_size).tolist()
            # Scalar per trace: dim whole trace if it has no highlighted points.
            op: Any = normal_opacity if has_any else dim_opacity
            lw = 0
            sizes_l.append(sizes)
            op_l.append(op)
            lw_l.append(lw)
        self._apply_marker_style_to_traces(sizes_l, op_l, lw_l)
        self._update_stats()

    def _apply_selected_point_highlight(self) -> None:
        """Show selection via overlay trace only; data traces use highlight or uniform style."""
        if self.fig_widget is None or self._cached_df is None:
            self._update_stats()
            return

        ov = self._selection_overlay_trace()
        x_col, y_col, z_col = self._coord_columns()

        # Do **not** wrap the block below in batch_update while calling
        # ``_apply_highlight`` / ``_apply_uniform_data_traces``: those paths use
        # their own ``batch_update`` on the same FigureWidget. Nested
        # ``batch_update`` contexts break FigureWidget sync after a figure rebuild
        # (e.g. recompute with a new method), leaving data traces empty in the UI.

        if self._selected_point_index is not None:
            m = self._cached_index_to_row
            if m is None or self._selected_point_index not in m:
                self._selected_point_index = None

        # Apply base styling to all data traces (each helper uses one batch_update).
        if self._highlight_active:
            self._apply_highlight()
        else:
            self._apply_uniform_data_traces()

        if ov is not None:
            with self.fig_widget.batch_update():
                if self._selected_point_index is None:
                    ov.visible = False
                    ov.x = []
                    ov.y = []
                    if getattr(ov, "type", "") == "scatter3d":
                        ov.z = []
                    ov.customdata = []
                else:
                    df = self._cached_df
                    m = self._cached_index_to_row
                    if m is None or self._selected_point_index not in m:
                        self._selected_point_index = None
                        ov.visible = False
                        ov.x = []
                        ov.y = []
                        if getattr(ov, "type", "") == "scatter3d":
                            ov.z = []
                        ov.customdata = []
                    else:
                        row = df.iloc[m[self._selected_point_index]]
                        highlight_size = self._highlight_marker_size()
                        ov.marker.size = highlight_size
                        ov.marker.line = {"color": "black", "width": 2}
                        if z_col:
                            ov.x = [float(row[x_col])]
                            ov.y = [float(row[y_col])]
                            ov.z = [float(row[z_col])]
                        else:
                            ov.x = [float(row[x_col])]
                            ov.y = [float(row[y_col])]
                        ov.customdata = [[self._selected_point_index]]
                        ov.visible = True

        self._update_stats()

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------
    def _on_compute_clicked(self, _btn: Any) -> None:
        """Handle Compute button click - runs DR and rebuilds figure.

        DR runs in a daemon thread so the UI stays responsive; a ticker updates
        elapsed time in the status label.  The embedding plot is mounted by
        updating ``_plot_slot.children`` so it still appears when compute
        finishes on that thread (see module docstring / Output widget threading
        notes).  Library warnings from DR may still print to the notebook or
        kernel log.
        """
        if self._compute_thread is not None and self._compute_thread.is_alive():
            return

        compute_done = threading.Event()
        t_start = time.time()
        self.compute_button.disabled = True
        self.status_label.value = self._status_html("Computing… 0s", "warn")
        # Full-size computing state so the plot area is not a blank gap while DR runs.
        self._plot_slot.children = (self._embedding_placeholder_computing,)

        def _tick() -> None:
            while True:
                if compute_done.wait(timeout=0.5):
                    break
                elapsed = time.time() - t_start
                self.status_label.value = self._status_html(f"Computing… {elapsed:.0f}s", "warn")

        def _run() -> None:
            ticker = threading.Thread(target=_tick, daemon=True)
            ticker.start()
            elapsed = 0.0
            try:
                # Do not use ``with self.output_area`` here: compute runs on a worker
                # thread where Output's capture context has no valid kernel parent.
                elapsed = self._compute_embedding()

                # Rebuilding and displaying the figure should happen via the event loop
                # to ensure it doesn't conflict with any active interaction.
                def _display_new_fig() -> None:
                    try:
                        # Lock during figure replacement to avoid race conditions with clicks
                        with self._click_lock:
                            self._build_figure()
                            self._display_figure()
                    except Exception as e:
                        self.status_label.value = self._status_html(f"Display error: {e}", "err")

                _schedule_after_plotly_event_loop(_display_new_fig)

                n = len(self._cached_df) if self._cached_df is not None else 0
                self.status_label.value = self._status_html(
                    f"Done — {n:,} points in {elapsed:.1f}s", "ok"
                )
            except Exception as e:
                self.status_label.value = self._status_html(f"Error: {e}", "err")
                import traceback

                self._plot_slot.children = (self._embedding_placeholder,)
                self.output_area.append_stderr(traceback.format_exc() + "\n")
            finally:
                compute_done.set()
                self.compute_button.disabled = False
                self._update_stats()

        self._compute_thread = threading.Thread(target=_run, daemon=True)
        self._compute_thread.start()

    def _on_color_changed(self, change: Any) -> None:
        """Instant colour change (no recomputation)."""
        if self._cached_df is None:
            return
        self.status_label.value = self._status_html("Updating colours…", "info")

        def _do() -> None:
            try:
                self._recolor_figure()
                if self._highlight_active:
                    self._apply_highlight()
                elif self._selected_point_index is not None:
                    self._apply_selected_point_highlight()
                self.status_label.value = self._status_html("Colour updated", "ok")
                self._update_stats()
            except Exception as e:
                self.status_label.value = self._status_html(f"Colour error: {e}", "err")

        _schedule_after_plotly_event_loop(_do)

    def _on_colorscale_changed(self, change: Any) -> None:
        """Instant colourscale change for continuous variables."""
        if self._cached_df is None:
            return

        def _do() -> None:
            try:
                self._recolor_figure()
                if self._highlight_active:
                    self._apply_highlight()
                elif self._selected_point_index is not None:
                    self._apply_selected_point_highlight()
                self._update_stats()
            except Exception as e:
                self.status_label.value = self._status_html(f"Colourscale error: {e}", "err")

        _schedule_after_plotly_event_loop(_do)

    def _on_highlight_key_changed(self, change: Any) -> None:
        """Update available values when highlight field changes."""
        self._update_highlight_value_options()
        if not (self._highlight_active and self.fig_widget is not None):
            return

        def _do() -> None:
            self._apply_highlight()
            self._update_highlight_status()

        _schedule_after_plotly_event_loop(_do)

    def _sync_highlight_button_appearance(self) -> None:
        """Sync highlight button label/style with :attr:`_highlight_active`."""
        btn = self.highlight_toggle
        btn.button_style = ""
        if self._highlight_active:
            btn.description = "Stop"
            btn.tooltip = "Stop group highlight"
            btn.style = self._highlight_btn_style_stop
        else:
            btn.description = "Highlight"
            btn.tooltip = "Apply group highlight on the plot"
            btn.style = self._highlight_btn_style_blue

    def _on_highlight_value_changed(self, change: Any) -> None:
        """Re-apply highlight when value changes (if highlight is active)."""
        if not self._highlight_active:
            return

        def _do() -> None:
            self._apply_highlight()
            self._update_highlight_status()

        _schedule_after_plotly_event_loop(_do)

    def _on_highlight_click(self, _btn: Any) -> None:
        """Toggle group highlight on/off via button."""
        self._highlight_active = not self._highlight_active
        self._sync_highlight_button_appearance()
        new_on = self._highlight_active

        def _do() -> None:
            self._apply_highlight()
            if new_on:
                self._update_highlight_status()
            else:
                self.status_label.value = self._status_html("Highlight off", "ok")
                self._update_stats()

        _schedule_after_plotly_event_loop(_do)

    def _update_highlight_status(self) -> None:
        """Show how many points are highlighted in the status bar."""
        self._update_stats()
        mask = self._build_highlight_mask()
        if mask is not None:
            n_hl = int(mask.sum())
            n_total = len(mask)
            self.status_label.value = self._status_html(
                f"Highlighted {n_hl:,} / {n_total:,} points", "ok"
            )
        else:
            self.status_label.value = self._status_html("Highlight on", "ok")

    def _set_image_panel_idle(self) -> None:
        """Reset the click-to-inspect panel to the placeholder (no PNG)."""
        self.img_output.children = (self._img_idle_placeholder,)

    def _debounce_search(self, timer_attr: str, apply_fn: Callable[[], None]) -> None:
        """Cancel any pending timer and schedule ``apply_fn`` after :data:`_SEARCH_DEBOUNCE_SEC`."""
        timer: threading.Timer | None = getattr(self, timer_attr)
        if timer is not None:
            timer.cancel()
        new_timer = threading.Timer(_SEARCH_DEBOUNCE_SEC, apply_fn)
        setattr(self, timer_attr, new_timer)
        new_timer.start()

    def _on_slot_value_changed(
        self,
        label: str,
        key_dropdown: widgets.Dropdown,
        value_select: widgets.SelectMultiple,
    ) -> None:
        """Sync pipeline filter/exclude dict and update status (shared by filter + exclude)."""
        self._sync_embed_filters_from_widgets()
        key = key_dropdown.value
        values = list(value_select.value)
        self.status_label.value = self._status_html(
            f"{label}: {key}={values if values else 'None'}", "info"
        )

    def _on_filter_key_changed(self, change: Any) -> None:
        """Update available filter values when filter field changes."""
        self._update_filter_value_options()

    def _on_filter_value_changed(self, change: Any) -> None:
        """Sync pipeline filter dict when filter values change."""
        self._on_slot_value_changed("Filter", self.filter_key_dropdown, self.filter_value_select)

    def _on_exclude_key_changed(self, change: Any) -> None:
        """Update available exclude values when exclude field changes."""
        self._update_exclude_value_options()

    def _on_exclude_value_changed(self, change: Any) -> None:
        """Sync pipeline exclude dict when exclude values change."""
        self._on_slot_value_changed("Exclude", self.exclude_key_dropdown, self.exclude_value_select)

    def _on_filter_search_changed(self, _change: Any) -> None:
        """Live-filter filter values (debounced)."""
        self._debounce_search(
            "_filter_search_debounce_timer",
            lambda: self._apply_filter_search_filter(select_default=False),
        )

    def _on_exclude_search_changed(self, _change: Any) -> None:
        """Live-filter exclude values (debounced)."""
        self._debounce_search(
            "_exclude_search_debounce_timer",
            lambda: self._apply_exclude_search_filter(select_default=False),
        )

    def _on_filter_clear_clicked(self, _btn: Any) -> None:
        """Clear filter and exclude selections (recompute still requires Compute)."""
        self.filter_search_input.value = ""
        self.exclude_search_input.value = ""
        self.filter_value_select.value = ()
        self.exclude_value_select.value = ()
        self._update_filter_value_options(select_default=False)
        self._update_exclude_value_options(select_default=False)
        self.status_label.value = self._status_html("Filters cleared", "info")

    def _on_marker_style_changed(self, change: Any) -> None:
        """Adjust marker size / opacity without recomputation."""
        if self.fig_widget is None:
            return

        def _do() -> None:
            fig = self.fig_widget
            if fig is None:
                return
            if self._highlight_active:
                self._apply_highlight()
            else:
                if self._selected_point_index is not None:
                    self._apply_selected_point_highlight()
                else:
                    new_size = self.point_size_slider.value
                    new_opacity = self.opacity_slider.value
                    with fig.batch_update():
                        for trace in self._get_data_traces():
                            trace.marker.size = new_size
                            trace.marker.opacity = new_opacity
            if self._selected_point_index is not None and self._highlight_active:
                ov = self._selection_overlay_trace()
                if ov is not None:
                    hs = self._highlight_marker_size()
                    with fig.batch_update():
                        ov.marker.size = hs
            self._update_stats()

        _schedule_after_plotly_event_loop(_do)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _display_figure(self) -> None:
        """Show the FigureWidget in the plot slot (safe from background compute threads)."""
        if self.fig_widget is not None:
            self._plot_slot.children = (self.fig_widget,)
        else:
            self._plot_slot.children = (self._embedding_placeholder,)
        self._update_stats()


def create_interactive_explorer(
    pheno_me: _InteractiveExplorerProtocol,
    filters: dict[str, Any] | None = None,
    exclude: dict[str, Any] | None = None,
    hover_features: list[str] | None = None,
) -> PhenoMeInteractive:
    """Launch an interactive explorer for phenotyping results in Jupyter.

    Creates a PhenoMeInteractive instance and displays it. Use in Jupyter notebooks
    to explore embeddings via PCA/t-SNE/UMAP with instant color switching,
    highlight mode, and click-to-inspect image viewing.

    Args:
        pheno_me: Processed PhenoMe instance with embeddings
            and optional properties. Must have run process_images() first.
        filters: Optional metadata filters to restrict which images are shown.
            Dict mapping metadata keys to allowed values or lists of values.
            Example: {'condition': 'Control', 'time': ['24h', '48h']}.
        exclude: Optional metadata exclusions (same structure as filters).
        hover_features: Optional list of metadata or property keys to show in
            hover tooltips. If None, uses metadata keys from the pipeline.

    Returns:
        PhenoMeInteractive: The explorer instance. Call .show() again to re-display.

    Example:
        >>> from phenome import PhenoMe, load_dinov2_model
        >>> model, wrapper = load_dinov2_model()
        >>> pheno = PhenoMe(seed=42)
        >>> pheno.find_files("data/")
        >>> pheno.process_images(wrapper)
        >>> explorer = pheno.create_interactive_explorer(
        ...     filters={'condition': 'Treatment'},
        ...     hover_features=['drug', 'area', 'eccentricity'],
        ... )
    """
    explorer = PhenoMeInteractive(
        pheno_me,
        filters=filters,
        exclude=exclude,
        hover_features=hover_features,
    )
    explorer.show()
    return explorer
