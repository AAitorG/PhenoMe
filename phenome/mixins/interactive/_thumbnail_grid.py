"""Clickable flex-wrap thumbnail grid for box/lasso selection (anywidget + front end)."""

from __future__ import annotations

import anywidget
import traitlets as t

from ._constants import THUMBNAIL_SIZE_PX

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
    """Renders a scrollable pack of small PNGs; single-click vs double-click are debounced in JS."""

    _esm = _ESM
    images_b64 = t.List(t.Unicode(), default_value=[]).tag(sync=True)
    indices = t.List(t.Integer(), default_value=[]).tag(sync=True)
    thumb_size_px = t.CInt(default_value=THUMBNAIL_SIZE_PX).tag(sync=True)
    # Incremented on every thumbnail single click so re-selecting the same index still notifies.
    thumb_strike = t.CInt(0).tag(sync=True)
    thumb_strike_index = t.CInt(-1).tag(sync=True)
    # Double-click: open full single-image view from the explorer.
    thumb_dbl_strike = t.CInt(0).tag(sync=True)
    thumb_dbl_strike_index = t.CInt(-1).tag(sync=True)
    # Pipeline index of the last thumbnail chosen for plot traceback; -1 = none.
    grid_focus_index = t.CInt(-1).tag(sync=True)
