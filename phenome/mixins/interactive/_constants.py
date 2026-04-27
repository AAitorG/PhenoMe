"""Module-level constants for the interactive explorer.

Kept in a dedicated module so callables in ``_utils`` / ``_html`` / ``explorer``
share a single source of truth for timings, pixel sizes, and layout tokens.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Timing / debounce
# ---------------------------------------------------------------------------

# Small delay so the browser finishes dropdown/slider closure and releases focus
# before we push large FigureWidget state syncs (avoids notebook/IDE cell selection
# getting "stuck" after Appearance / Highlight interactions).
DEFER_UI_SEC = 0.1
# Typing debounce for filter/exclude/highlight search inputs
SEARCH_DEBOUNCE_SEC = 0.25
# Click handler re-entry guard (debounce)
CLICK_DEBOUNCE_SEC = 0.15
# Reproducible subsample when downsampling in ``_compute_embedding``.
DR_RANDOM_STATE = 42
# Max dataset rows sampled when judging Color-by metadata/property cardinality
INTERACTIVE_COLOR_SCAN_MAX_INDICES = 3000


# ---------------------------------------------------------------------------
# Trace / plot primitives
# ---------------------------------------------------------------------------

# Extra Plotly trace for click-to-select border (2D/3D); must match overlay click skip.
SELECTION_OVERLAY_NAME = "_phenome_sel_overlay"
# Extra Plotly trace slot for box/lasso (2D only); selection drawn via Plotly dimming, not this trace.
MULTI_SELECT_OVERLAY_NAME = "_phenome_multi_sel_overlay"
# Extra Plotly trace for group-highlight halo (2D only).
HIGHLIGHT_OVERLAY_NAME = "_phenome_highlight_overlay"
# 2D only: thick orange ring for “which thumbnail” traceback (distinct from plot-click black).
GRID_THUMB_OVERLAY_NAME = "_phenome_grid_thumb_overlay"

# Halo fill for group-highlight (translucent; click-to-select uses a ring on a separate trace).
HALO_COLOR_HIGHLIGHT = "rgba(236, 72, 153, 0.40)"  # pink — group highlight
HALO_SIZE_MULTIPLIER = 2.4  # halo diameter vs base point size


# ---------------------------------------------------------------------------
# Color scales (continuous)
# ---------------------------------------------------------------------------

CONTINUOUS_SCALES = [
    "Viridis",
    "Plasma",
    "Inferno",
    "Magma",
    "Cividis",
    "RdYlBu",
    "Coolwarm",
    "Spectral",
    "RdBu",
    "BrBG",
    "Blues",
    "Greens",
    "Reds",
    "Purples",
    "YlOrRd",
]


# ---------------------------------------------------------------------------
# Shared layout tokens (ipywidgets layout is more reliable with fixed widths)
# ---------------------------------------------------------------------------

INTERACTIVE_VALUES_H = 168
INTERACTIVE_SEARCH_W_PX = 240
INTERACTIVE_VALUES_W_PX = 320
INTERACTIVE_APPEAR_COL_W_PX = 300
INTERACTIVE_FIELD_MAX_W_PX = 600

INTERACTIVE_SECTION_SEP_HTML = (
    '<div style="height:1px;background:#DDE1E6;margin:2px 0 0 0;width:100%;"></div>'
)
INTERACTIVE_DESC_STYLE = {"description_width": "70px"}

# Integer columns with at most this many distinct values use categorical highlight
# (multi-select), same idea as Filter/Exclude — avoids a range slider on cluster IDs.
HIGHLIGHT_DISCRETE_INT_MAX_UNIQUES = 512


# ---------------------------------------------------------------------------
# Figure pixel sizing
# ---------------------------------------------------------------------------

EMBEDDING_FIG_WIDTH_PX = 800
EMBEDDING_FIG_HEIGHT_PX = 700
IMAGE_OVERLAY_WIDTH_PX = 500
# Min height for the image panel (compact idle/loading; PNG expands when shown).
IMAGE_PANEL_MIN_HEIGHT_PX = 380
# Box/lasso multi-select: thumbnail grid (anywidget) — small PNGs, parallel decode.
THUMBNAIL_SIZE_PX = 80
THUMBNAIL_GRID_MAX_IMAGES = 100
THUMBNAIL_DOWNSAMPLE = 96
# Matplotlib figsize (inches) for overlay image; ~100 DPI matches panel width minus padding.
IMAGE_OVERLAY_FIGSIZE: tuple[float, float] = (6.0, 6.0)
