"""HTML components, layout constants, and anywidget primitives for the interactive explorer.

Consolidates UI fragments, pixel sizes, and the thumbnail grid widget into a
single module for better navigability.
"""

from __future__ import annotations

import html

import anywidget
import traitlets as t

# ---------------------------------------------------------------------------
# Timing / debounce
# ---------------------------------------------------------------------------
DEFER_UI_SEC = 0.1
SEARCH_DEBOUNCE_SEC = 0.25
CLICK_DEBOUNCE_SEC = 0.15
SELECTION_MERGE_WINDOW_SEC = 0.2
DR_RANDOM_STATE = 42
INTERACTIVE_COLOR_SCAN_MAX_INDICES = 3000


# ---------------------------------------------------------------------------
# Trace / plot primitives
# ---------------------------------------------------------------------------
SELECTION_OVERLAY_NAME = "_phenome_sel_overlay"
MULTI_SELECT_OVERLAY_NAME = "_phenome_multi_sel_overlay"
HIGHLIGHT_OVERLAY_NAME = "_phenome_highlight_overlay"
GRID_THUMB_OVERLAY_NAME = "_phenome_grid_thumb_overlay"

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
# Shared layout tokens
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
HIGHLIGHT_DISCRETE_INT_MAX_UNIQUES = 512


# ---------------------------------------------------------------------------
# Figure pixel sizing
# ---------------------------------------------------------------------------
EMBEDDING_FIG_WIDTH_PX = 800
EMBEDDING_FIG_HEIGHT_PX = 700
IMAGE_OVERLAY_WIDTH_PX = 500
IMAGE_PANEL_MIN_HEIGHT_PX = 380
THUMBNAIL_SIZE_PX = 80
THUMBNAIL_GRID_MAX_IMAGES = 100
THUMBNAIL_DOWNSAMPLE = 96
IMAGE_OVERLAY_FIGSIZE: tuple[float, float] = (6.0, 6.0)


# ---------------------------------------------------------------------------
# Status chip and stats bar
# ---------------------------------------------------------------------------
def status_html(msg: str, level: str = "info") -> str:
    """Return styled HTML for the single-line status label."""
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


def stats_bar_html(
    n_total: int,
    n_highlighted: int,
    selected: int | None,
    error_msg: str | None = None,
    *,
    n_multi_selected: int = 0,
) -> str:
    """HTML for the live stats bar."""
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
    multi_span = (
        f'<span style="color:#ccc;">|</span><span><b>Lasso/Box</b> {n_multi_selected:,}</span>'
        if n_multi_selected
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
        f"{multi_span}"
        f"{sep_after_sel}"
        f"{err_span}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Embedding placeholders
# ---------------------------------------------------------------------------
def embedding_placeholder_idle(w_px: int, h_px: int) -> str:
    """Full-size empty state for the embedding slot."""
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


def embedding_placeholder_computing(w_px: int, h_px: int) -> str:
    """Full-size state shown while DR runs."""
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


# ---------------------------------------------------------------------------
# Image panel placeholders
# ---------------------------------------------------------------------------
def image_panel_idle(w_px: int, min_h_px: int) -> str:
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


def image_panel_loading(w_px: int, min_h_px: int) -> str:
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


# ---------------------------------------------------------------------------
# Thumbnail Grid AnyWidget
# ---------------------------------------------------------------------------
_ESM = r"""
function render(view) {
  const el = view.el;
  const model = view.model;
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexWrap = "wrap";
  wrap.style.alignContent = "flex-start";
  wrap.style.gap = "4px";
  wrap.style.maxHeight = "min(60vh, 620px)";
  wrap.style.overflowY = "auto";
  wrap.style.overflowX = "hidden";
  wrap.style.width = "100%";
  el.appendChild(wrap);

  function paint() {
    const b64s = model.get("images_b64");
    const idxs = model.get("indices");
    const sz = model.get("thumb_size_px");
    const focus = model.get("grid_focus_index");
    wrap.innerHTML = "";
    for (let i = 0; i < b64s.length; i++) {
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + b64s[i];
      img.width = sz;
      img.height = sz;
      img.style.objectFit = "contain";
      img.style.cursor = "pointer";
      img.style.boxSizing = "border-box";
      const ix = idxs[i];
      const isFocus = focus === ix;
      img.style.border = isFocus
        ? "3px solid #f97316"
        : "1px solid #ccc";
      img.style.boxShadow = isFocus ? "0 0 0 1px rgba(249, 115, 22, 0.35)" : "none";
      img.style.borderRadius = "3px";
      const DBL = 280;
      let clickTimer = null;
      img.addEventListener("click", () => {
        if (clickTimer) {
          clearTimeout(clickTimer);
        }
        clickTimer = setTimeout(() => {
          clickTimer = null;
          model.set("thumb_strike_index", ix);
          model.set("thumb_strike", model.get("thumb_strike") + 1);
          model.save_changes();
        }, DBL);
      });
      img.addEventListener("dblclick", (e) => {
        e.preventDefault();
        if (clickTimer) {
          clearTimeout(clickTimer);
          clickTimer = null;
        }
        model.set("thumb_dbl_strike_index", ix);
        model.set("thumb_dbl_strike", model.get("thumb_dbl_strike") + 1);
        model.save_changes();
      });
      wrap.appendChild(img);
    }
  }

  paint();
  model.on("change:images_b64", paint);
  model.on("change:indices", paint);
  model.on("change:thumb_size_px", paint);
  model.on("change:grid_focus_index", paint);
}
export default { render };
"""


class ThumbnailGrid(anywidget.AnyWidget):
    """Renders a scrollable pack of small PNGs."""

    _esm = _ESM
    images_b64 = t.List(t.Unicode(), default_value=[]).tag(sync=True)
    indices = t.List(t.Integer(), default_value=[]).tag(sync=True)
    thumb_size_px = t.CInt(default_value=THUMBNAIL_SIZE_PX).tag(sync=True)
    thumb_strike = t.CInt(0).tag(sync=True)
    thumb_strike_index = t.CInt(-1).tag(sync=True)
    thumb_dbl_strike = t.CInt(0).tag(sync=True)
    thumb_dbl_strike_index = t.CInt(-1).tag(sync=True)
    grid_focus_index = t.CInt(-1).tag(sync=True)
