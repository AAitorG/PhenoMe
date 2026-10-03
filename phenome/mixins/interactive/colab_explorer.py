"""Small interactive explorer for Google Colab.

Colab only syncs widget updates made while handling the browser request that
triggered them. This explorer therefore computes the embedding and loads the
clicked image inside those callbacks, and it skips the Jupyter explorer's
accordion, highlight, selection, and animated placeholders.
"""

from __future__ import annotations

import contextlib
import html
from typing import Any, Literal, cast

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

from ..._logging import get_logger
from ...core import run_dimensionality_reduction
from ...plotly_display import apply_figurewidget_display_config
from ._html_components import CONTINUOUS_SCALES, status_html
from ._protocol import _InteractiveExplorerProtocol
from ._utils import (
    build_informative_color_columns,
    figurewidget_safe_figure,
    raw_index_from_customdata_row,
)

logger = get_logger(__name__)

# Same proportions as the first Colab figure (540 x 460).
_FIG_WIDTH_PX = 780
_FIG_HEIGHT_PX = 665
_IMAGE_MAX_PX = 330
_IMAGE_PANEL_PX = 400
_RING_NAME = "_phenome_sel_overlay"
_COLOR_DEFAULTS = ("drug", "drug_name", "cluster", "treatment", "condition", "time")


def _enable_colab_widgets() -> None:
    """Turn on the widget manager Colab needs before a FigureWidget is shown."""
    try:
        from google.colab import output

        output.enable_custom_widget_manager()
        output.no_vertical_scroll()
    except Exception:
        logger.warning("Colab widget setup failed", exc_info=True)


def _section_label(text: str) -> widgets.HTML:
    """Plain section title. No accordion: Colab's widget manager cannot draw one."""
    safe = html.escape(text)
    return widgets.HTML(
        value=(
            f'<div style="font-size:13px;font-weight:600;color:#1e293b;'
            f'margin:2px 0 4px 0;">{safe}</div>'
        )
    )


def _idle_plot_html() -> str:
    """Static placeholder shown until Compute finishes."""
    return (
        f'<div style="width:{_FIG_WIDTH_PX}px;height:{_FIG_HEIGHT_PX}px;box-sizing:border-box;'
        "display:flex;align-items:center;justify-content:center;"
        'border:1px solid #d0d8e3;background:#f8fafc;">'
        '<div style="text-align:center;font-size:12px;color:#5d6d7e;max-width:280px;">'
        "<b>Embedding</b><br>Press <b>Compute</b> to build the plot."
        "</div></div>"
    )


def _idle_image_html() -> str:
    """Static placeholder for the side image until a point is clicked."""
    inner = _FIG_HEIGHT_PX - 16
    return (
        f'<div style="height:{inner}px;display:flex;align-items:center;justify-content:center;'
        'text-align:center;font-size:12px;color:#5d6d7e;padding:12px;box-sizing:border-box;">'
        "Click a point to show its image.</div>"
    )


class ColabInteractiveExplorer:
    """Embedding scatter with appearance controls and click-to-image for Colab.

    The menu is Embedding (method, dimensions, source, Compute) and Appearance
    (color, palette, point size, opacity, image extra info). Filters and
    exclusions passed to the constructor are applied when Compute runs.
    """

    def __init__(
        self,
        pheno_me: _InteractiveExplorerProtocol,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
    ) -> None:
        """Build the controls. Call :meth:`show` to display them.

        Args:
            pheno_me: Processed PhenoMe instance. Must have run ``process_images()``.
            filters: Metadata filters applied at compute time.
            exclude: Metadata exclusions applied at compute time.
        """
        self.pheno = pheno_me
        self.filters = dict(filters) if filters else None
        self.exclude = dict(exclude) if exclude else None

        self._df: pd.DataFrame | None = None
        self._method_label: str | None = None
        self._source: str | None = None
        self._ndims: int | None = None
        self._dr_obj: Any = None
        self._fig: go.FigureWidget | None = None
        self._n_data_traces = 0
        self._index_to_row: dict[int, int] = {}
        self._selected: int | None = None
        self._suspend_appearance = False
        self._displayed: widgets.Widget | None = None

        style = {"description_width": "70px"}
        dropdown = widgets.Layout(width="200px")
        self.method_dropdown = widgets.Dropdown(
            options=["PCA", "t-SNE", "UMAP"],
            value="t-SNE",
            description="Method:",
            style=style,
            layout=dropdown,
        )
        self.dim_toggle = widgets.Dropdown(
            options=[("2D", 2), ("3D", 3)],
            value=2,
            description="Dims:",
            style=style,
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
            style=style,
            layout=dropdown,
        )
        self.compute_button = widgets.Button(
            description="Compute",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )
        pair = widgets.Layout(width="220px")
        slider = widgets.Layout(width="380px")
        self.color_dropdown = widgets.Dropdown(
            description="Color by:",
            style=style,
            layout=pair,
        )
        self.colorscale_dropdown = widgets.Dropdown(
            options=CONTINUOUS_SCALES,
            value="Viridis",
            description="Palette:",
            style=style,
            layout=pair,
        )
        self.point_size_slider = widgets.IntSlider(
            value=7,
            min=1,
            max=20,
            step=1,
            description="Pt size:",
            style=style,
            layout=slider,
            continuous_update=False,
        )
        self.opacity_slider = widgets.FloatSlider(
            value=0.7,
            min=0.05,
            max=1.0,
            step=0.05,
            description="Opacity:",
            style=style,
            layout=slider,
            continuous_update=False,
        )
        self.show_extra_info = widgets.Checkbox(
            value=True,
            description="Image extra info",
            indent=False,
            layout=widgets.Layout(margin="0 0 0 18px"),
        )
        self.status_label = widgets.HTML(value=status_html("Ready", "info"))
        self._plot_slot = widgets.VBox(
            [widgets.HTML(value=_idle_plot_html())],
            layout=widgets.Layout(width=f"{_FIG_WIDTH_PX}px", min_height=f"{_FIG_HEIGHT_PX}px"),
        )
        self.img_output = widgets.VBox(
            [widgets.HTML(value=_idle_image_html())],
            layout=widgets.Layout(
                width=f"{_IMAGE_PANEL_PX}px",
                min_width=f"{_IMAGE_PANEL_PX}px",
                height=f"{_FIG_HEIGHT_PX}px",
                min_height=f"{_FIG_HEIGHT_PX}px",
                max_height=f"{_FIG_HEIGHT_PX}px",
                border="1px solid #E0E0E0",
                padding="8px",
                align_items="center",
                overflow_y="hidden",
            ),
        )
        self._dashboard = self._build_dashboard()
        self._update_color_options()
        self.compute_button.on_click(self._on_compute_clicked)
        self.color_dropdown.observe(self._on_color_changed, names="value")
        self.colorscale_dropdown.observe(self._on_color_changed, names="value")
        self.point_size_slider.observe(self._on_marker_style_changed, names="value")
        self.opacity_slider.observe(self._on_marker_style_changed, names="value")
        self.show_extra_info.observe(self._on_extra_info_changed, names="value")

    def show(self) -> None:
        """Display the dashboard.

        A later call shows the same widget again. The previous tree is hidden
        rather than cleared, because Colab can erase a widget displayed right
        after ``clear_output``.
        """
        _enable_colab_widgets()
        if self._displayed is not None and self._displayed is not self._dashboard:
            self._displayed.layout.display = "none"
            with contextlib.suppress(Exception):
                self._displayed.close()
        self._dashboard.layout.display = "flex"
        if self._displayed is not self._dashboard:
            display(self._dashboard)
            self._displayed = self._dashboard

    def _build_dashboard(self) -> widgets.VBox:
        """Stack the two control groups above the plot and the image."""
        row = widgets.Layout(width="100%", flex_flow="row wrap", align_items="center")
        embedding = widgets.VBox(
            [
                _section_label("Embedding"),
                widgets.HBox(
                    [
                        self.method_dropdown,
                        self.dim_toggle,
                        self.source_dropdown,
                        self.compute_button,
                    ],
                    layout=row,
                ),
            ]
        )
        appearance = widgets.VBox(
            [
                _section_label("Appearance"),
                widgets.HBox(
                    [self.color_dropdown, self.colorscale_dropdown, self.show_extra_info],
                    layout=row,
                ),
                widgets.HBox([self.point_size_slider, self.opacity_slider], layout=row),
            ]
        )
        sidebar = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="background:#4E79A7;color:white;padding:8px 12px;'
                    'font-size:14px;font-weight:600;">PhenoMe Explorer</div>'
                ),
                embedding,
                appearance,
                self.status_label,
            ],
            layout=widgets.Layout(
                width="100%",
                border="1px solid #E0E0E0",
                padding="8px",
            ),
        )
        main = widgets.HBox(
            [self._plot_slot, self.img_output],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )
        return widgets.VBox([sidebar, main], layout=widgets.Layout(width="100%"))

    def _update_color_options(self) -> None:
        """Fill the color dropdown from metadata and property columns."""
        results = getattr(self.pheno, "results", None)
        if not results:
            self.color_dropdown.options = []
            return
        prev = self.color_dropdown.value
        prop_keys: list[str] = []
        getter = getattr(self.pheno, "get_available_property_keys", None)
        if getter is not None:
            prop_keys = list(getter())
        cols = build_informative_color_columns(results, prop_keys)
        self._suspend_appearance = True
        try:
            self.color_dropdown.options = cols
            if cols and prev in cols:
                self.color_dropdown.value = prev
            elif cols:
                self.color_dropdown.value = next(
                    (name for name in _COLOR_DEFAULTS if name in cols),
                    cols[0],
                )
        finally:
            self._suspend_appearance = False

    def _on_compute_clicked(self, _btn: Any) -> None:
        """Run dimensionality reduction and mount the figure before returning."""
        self.compute_button.disabled = True
        self.status_label.value = status_html("Computing…", "warn")
        self._selected = None
        try:
            self._compute_embedding()
            self._update_color_options()
            self._mount_figure()
            n = 0 if self._df is None else len(self._df)
            self.status_label.value = status_html(f"Done — {n:,} points", "ok")
        except Exception as exc:
            logger.warning("Colab explorer compute failed", exc_info=True)
            self.status_label.value = status_html(f"Error: {exc}", "err")
            self._plot_slot.children = (widgets.HTML(value=_idle_plot_html()),)
        finally:
            self.compute_button.disabled = False

    def _compute_embedding(self) -> None:
        """Store the reduced coordinates for the current Embedding controls."""
        method_label = str(self.method_dropdown.value)
        method_key = cast(
            Literal["pca", "tsne", "umap"],
            method_label.lower().replace("-", ""),
        )
        n_dims = int(self.dim_toggle.value)
        source = str(self.source_dropdown.value)
        try:
            df, _feature_type, dr_obj = run_dimensionality_reduction(
                self.pheno,
                method=method_key,
                n_components=n_dims,
                source=source,
                filters=self.filters,
                exclude=self.exclude,
                device=self.pheno.device,
                use_gpu=getattr(self.pheno, "use_gpu_for_dr", True),
            )
        except ImportError as exc:
            message = str(exc).lower()
            if "umap" in message:
                raise ImportError(
                    "UMAP is not installed. Please install it with: pip install umap-learn"
                ) from exc
            if "torchdr" in message:
                raise ImportError(
                    "TorchDR is not installed. Please install it with: pip install torchdr"
                ) from exc
            raise
        if df is None or len(df) == 0:
            raise ValueError("Dimensionality reduction returned no data.")
        self._df = df
        self._method_label = method_label
        self._source = source
        self._ndims = n_dims
        self._dr_obj = dr_obj
        self._index_to_row = {}
        if "Index" in df.columns:
            for row_i, value in enumerate(df["Index"].to_numpy()):
                key = int(value)
                if key not in self._index_to_row:
                    self._index_to_row[key] = row_i

    def _mount_figure(self) -> None:
        """Build the scatter and put it directly in the plot column."""
        self._fig = self._build_figure()
        self._plot_slot.children = (self._fig,)
        self._apply_ring()

    def _build_figure(self) -> go.FigureWidget:
        """SVG scatter in 2D, scatter3d in 3D, plus one click-ring trace."""
        df = self._df
        if df is None:
            raise ValueError("Compute an embedding before building the plot.")
        x_col, y_col, z_col = self._coord_columns()
        color_by = self.color_dropdown.value
        color_column: str | None = None
        is_continuous = False
        if color_by:
            try:
                color_column, is_continuous = self.pheno._get_color_column(str(color_by), df)
            except (ValueError, AttributeError):
                color_column = None

        kwargs: dict[str, Any] = {
            "data_frame": df,
            "x": x_col,
            "y": y_col,
            "width": _FIG_WIDTH_PX,
            "height": _FIG_HEIGHT_PX,
        }
        if "Index" in df.columns:
            kwargs["hover_data"] = ["Index"]
        if z_col:
            kwargs["z"] = z_col
        else:
            kwargs["render_mode"] = "svg"
        if color_column:
            kwargs["color"] = color_column
            if is_continuous:
                kwargs["color_continuous_scale"] = self.colorscale_dropdown.value

        fig = px.scatter_3d(**kwargs) if z_col else px.scatter(**kwargs)
        size = int(self.point_size_slider.value)
        opacity = float(self.opacity_slider.value)
        fig.update_traces(marker={"size": size, "opacity": opacity, "line": {"width": 0}})
        if not color_column:
            fig.update_traces(marker_color="#636EFA")

        x_title, y_title, z_title = x_col, y_col, z_col
        if self._method_label == "PCA" and self._dr_obj is not None:
            ratios = getattr(self._dr_obj, "explained_variance_ratio_", None)
            if ratios is not None and len(ratios) >= 2:
                x_title = f"Component 1 ({ratios[0]:.1%})"
                y_title = f"Component 2 ({ratios[1]:.1%})"
                if z_col and len(ratios) >= 3:
                    z_title = f"Component 3 ({ratios[2]:.1%})"

        title = self._method_label or "Embedding"
        if self._source:
            title = f"{title} ({self._source})"
        # Legend and colorbar sit under the axes so they do not steal plot width.
        margin_b = 48
        layout: dict[str, Any] = {
            "template": "plotly_white",
            "title": {"text": title, "x": 0.5, "xanchor": "center"},
            "autosize": False,
            "width": _FIG_WIDTH_PX,
            "height": _FIG_HEIGHT_PX,
            "hovermode": "closest",
        }
        if color_column and not is_continuous:
            layout["legend"] = {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.16,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 11},
                "itemsizing": "constant",
            }
            margin_b = 88
        elif color_column and is_continuous:
            layout["coloraxis"] = {
                "colorbar": {
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.18,
                    "xanchor": "center",
                    "x": 0.5,
                    "len": 0.55,
                    "thickness": 12,
                    "title": {"text": color_column, "side": "top"},
                }
            }
            margin_b = 88
        layout["margin"] = {"l": 48, "r": 16, "t": 48, "b": margin_b}
        if z_col:
            layout["scene"] = {
                "xaxis": {"title": x_title},
                "yaxis": {"title": y_title},
                "zaxis": {"title": z_title},
                "aspectmode": "data",
            }
        else:
            layout["xaxis"] = {"title": x_title}
            layout["yaxis"] = {"title": y_title, "scaleanchor": "x", "scaleratio": 1}
        fig.update_layout(**layout)

        widget = go.FigureWidget(figurewidget_safe_figure(fig))
        apply_figurewidget_display_config(widget)
        self._n_data_traces = len(widget.data)
        widget.add_trace(self._ring_trace(is_3d=bool(z_col), size=size))
        for trace in list(widget.data)[: self._n_data_traces]:
            trace.on_click(self._on_figure_click)
        return widget

    def _coord_columns(self) -> tuple[str, str, str | None]:
        """Return Component column names for the cached dimensionality."""
        z_col = "Component 3" if self._ndims == 3 else None
        return "Component 1", "Component 2", z_col

    def _ring_trace(self, *, is_3d: bool, size: int) -> go.Scatter | go.Scatter3d:
        """Hollow marker drawn on the clicked point."""
        marker = {
            "size": max(size + 5, int(size * 2)),
            "color": "rgba(0,0,0,0)",
            "line": {"color": "black", "width": 2},
        }
        common: dict[str, Any] = {
            "name": _RING_NAME,
            "mode": "markers",
            "showlegend": False,
            "hoverinfo": "skip",
            "visible": False,
            "marker": marker,
        }
        if is_3d:
            return go.Scatter3d(x=[], y=[], z=[], **common)
        return go.Scatter(x=[], y=[], **common)

    def _on_color_changed(self, _change: Any) -> None:
        """Recolor by rebuilding the figure inside the dropdown callback."""
        if self._suspend_appearance or self._df is None or self._fig is None:
            return
        try:
            self._mount_figure()
            self.status_label.value = status_html("Colour updated", "ok")
        except Exception as exc:
            self.status_label.value = status_html(f"Colour error: {exc}", "err")

    def _on_marker_style_changed(self, _change: Any) -> None:
        """Update point size and opacity on the traces already on screen."""
        if self._suspend_appearance or self._fig is None:
            return
        size = int(self.point_size_slider.value)
        opacity = float(self.opacity_slider.value)
        ring_size = max(size + 5, int(size * 2))
        with self._fig.batch_update():
            for trace in list(self._fig.data)[: self._n_data_traces]:
                trace.marker.size = size
                trace.marker.opacity = opacity
            self._fig.data[-1].marker.size = ring_size

    def _on_extra_info_changed(self, _change: Any) -> None:
        """Reload the open image when the extra-info checkbox changes."""
        if self._selected is not None:
            self._show_image(self._selected)

    def _on_figure_click(self, trace: Any, points: Any, _state: Any) -> None:
        """Show the clicked image and a ring before this callback returns."""
        if getattr(trace, "name", "") == _RING_NAME:
            return
        idx = self._index_from_click(trace, points)
        if idx < 0:
            return
        if self._selected == idx:
            self._selected = None
            self._apply_ring()
            self.img_output.children = (widgets.HTML(value=_idle_image_html()),)
            return
        self._selected = idx
        self._apply_ring()
        self._show_image(idx)

    def _index_from_click(self, trace: Any, points: Any) -> int:
        """Pipeline image index for a FigureWidget click, or -1."""
        if points is None:
            return -1
        point_idx: int | None = None
        if getattr(points, "point_inds", None):
            point_idx = int(points.point_inds[0])
        elif getattr(points, "xs", None) and getattr(points, "ys", None):
            point_idx = _nearest_point(trace, points)
        if point_idx is None:
            return -1
        custom = getattr(trace, "customdata", None)
        if custom is None or point_idx >= len(custom):
            return -1
        return raw_index_from_customdata_row(custom[point_idx])

    def _apply_ring(self) -> None:
        """Move the hollow ring to the selected point, or hide it."""
        fig = self._fig
        df = self._df
        if fig is None or df is None or not fig.data:
            return
        ring = fig.data[-1]
        selected = self._selected
        if selected is None or selected not in self._index_to_row:
            ring.visible = False
            ring.x = []
            ring.y = []
            if self._ndims == 3:
                ring.z = []
            return
        row = df.iloc[self._index_to_row[selected]]
        x_col, y_col, z_col = self._coord_columns()
        ring.x = [float(row[x_col])]
        ring.y = [float(row[y_col])]
        if z_col:
            ring.z = [float(row[z_col])]
        ring.visible = True

    def _show_image(self, idx: int) -> None:
        """Load one preview PNG and place it in the side panel."""
        try:
            png, details, title = self.pheno.image_preview_png_bytes(
                idx,
                apply_transforms=False,
                downsample=_IMAGE_MAX_PX * 2,
                show_extra_info=bool(self.show_extra_info.value),
            )
        except Exception as exc:
            logger.warning("Colab explorer image load failed for index %s", idx, exc_info=True)
            self.status_label.value = status_html(f"Error loading image: {exc}", "err")
            self.img_output.children = (widgets.HTML(value=_idle_image_html()),)
            return
        children: list[widgets.Widget] = []
        if title:
            children.append(
                widgets.HTML(
                    value=(
                        "<div style='text-align:center;font-weight:bold;font-size:13px;"
                        f"margin-bottom:6px;'>{html.escape(str(title))}</div>"
                    ),
                    layout=widgets.Layout(width="100%"),
                )
            )
        disp_w, disp_h = _fitted_image_px(png)
        children.append(
            widgets.Image(
                value=png,
                format="png",
                width=f"{disp_w}px",
                height=f"{disp_h}px",
                layout=widgets.Layout(width=f"{disp_w}px", height=f"{disp_h}px"),
            )
        )
        if details:
            details_h = _details_scroll_px(has_title=bool(title), image_h=disp_h)
            children.append(
                widgets.HTML(
                    value=(
                        f'<div style="margin-top:8px;height:{details_h}px;overflow-y:auto;'
                        "width:100%;box-sizing:border-box;border-top:1px solid #E0E0E0;"
                        'padding-top:6px;">'
                        '<pre style="margin:0;font-size:12px;white-space:pre-wrap;'
                        f'word-break:break-word;">{html.escape(str(details))}</pre></div>'
                    ),
                    layout=widgets.Layout(width="100%", height=f"{details_h + 8}px"),
                )
            )
        self.img_output.children = tuple(children)
        self.status_label.value = status_html(f"Image {idx}", "ok")


def _details_scroll_px(*, has_title: bool, image_h: int) -> int:
    """Height left in the image panel for the scrolling extra-info block.

    The panel is as tall as the plot. Title and image take the top; the rest
    scrolls.
    """
    used = 16 + image_h + 14
    if has_title:
        used += 28
    return max(72, _FIG_HEIGHT_PX - used)


def _png_size(png: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG header, or None when the bytes are not a PNG."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _fitted_image_px(png: bytes) -> tuple[int, int]:
    """Scale a preview so its long side is ``_IMAGE_MAX_PX``, keeping its aspect ratio."""
    size = _png_size(png)
    if size is None:
        return _IMAGE_MAX_PX, _IMAGE_MAX_PX
    width, height = size
    scale = _IMAGE_MAX_PX / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _nearest_point(trace: Any, points: Any) -> int | None:
    """Marker index closest to a click when ``point_inds`` is empty."""
    try:
        x0 = float(points.xs[0])
        y0 = float(points.ys[0])
        tx = np.asarray(trace.x, dtype=float)
        ty = np.asarray(trace.y, dtype=float)
    except (TypeError, ValueError, IndexError):
        return None
    if tx.size == 0 or tx.shape != ty.shape:
        return None
    if getattr(trace, "type", "") == "scatter3d" and getattr(points, "zs", None):
        tz = np.asarray(getattr(trace, "z", None), dtype=float)
        if tz.size == tx.size:
            z0 = float(points.zs[0])
            dist = (tx - x0) ** 2 + (ty - y0) ** 2 + (tz - z0) ** 2
            return int(np.argmin(dist))
    return int(np.argmin((tx - x0) ** 2 + (ty - y0) ** 2))


def create_colab_interactive_explorer(
    pheno_me: _InteractiveExplorerProtocol,
    filters: dict[str, Any] | None = None,
    exclude: dict[str, Any] | None = None,
) -> ColabInteractiveExplorer:
    """Launch a short interactive explorer that works in Google Colab.

    Shows Embedding and Appearance controls, an SVG scatter of the embedding,
    and the image for a clicked point. Highlight, box/lasso selection, and
    filter editors are not included. ``filters`` and ``exclude`` still restrict
    which images are reduced when Compute is pressed.

    Widget updates run inside the button and click callbacks. Colab does not
    show updates made from a background thread.

    Args:
        pheno_me: Processed PhenoMe instance with embeddings. Must have run
            ``process_images()`` first.
        filters: Optional metadata filters. Dict mapping metadata keys to
            allowed values or lists of values.
        exclude: Optional metadata exclusions (same structure as filters).

    Returns:
        The explorer. Call ``.show()`` again to re-display it.

    Example:
        >>> explorer = pheno.create_colab_interactive_explorer()
    """
    explorer = ColabInteractiveExplorer(pheno_me, filters=filters, exclude=exclude)
    explorer.show()
    return explorer
