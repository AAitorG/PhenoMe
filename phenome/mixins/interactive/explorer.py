"""
Interactive Visualization Module for PhenoMe.

Host class and factory. Shared constants, HTML builders, protocol, and pure
helpers live in sibling modules; everything is re-exported through
:mod:`phenome.mixins.interactive.__init__` to preserve the historical import
path ``phenome.mixins.interactive.PhenoMeInteractive``.
"""

from __future__ import annotations

import contextlib
import csv
import html
import io
import threading
import time
from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output, display

from ..._logging import get_logger
from ...core import (
    get_all_metadata_keys,
    get_metadata_value_from_dict,
    run_dimensionality_reduction,
)
from ...plotly_display import apply_figurewidget_display_config
from ._constants import (
    CLICK_DEBOUNCE_SEC,
    CONTINUOUS_SCALES,
    DR_RANDOM_STATE,
    EMBEDDING_FIG_HEIGHT_PX,
    EMBEDDING_FIG_WIDTH_PX,
    HALO_COLOR_HIGHLIGHT,
    HALO_COLOR_MULTI,
    HALO_SIZE_MULTIPLIER,
    HIGHLIGHT_DISCRETE_INT_MAX_UNIQUES,
    HIGHLIGHT_OVERLAY_NAME,
    IMAGE_OVERLAY_WIDTH_PX,
    IMAGE_PANEL_MIN_HEIGHT_PX,
    INTERACTIVE_APPEAR_COL_W_PX,
    INTERACTIVE_DESC_STYLE,
    INTERACTIVE_FIELD_MAX_W_PX,
    INTERACTIVE_SEARCH_W_PX,
    INTERACTIVE_SECTION_SEP_HTML,
    INTERACTIVE_VALUES_H,
    INTERACTIVE_VALUES_W_PX,
    MULTI_SELECT_OVERLAY_NAME,
    SEARCH_DEBOUNCE_SEC,
    SELECTION_OVERLAY_NAME,
)
from ._html import (
    embedding_placeholder_computing,
    embedding_placeholder_idle,
    image_panel_idle,
    image_panel_loading,
    stats_bar_html,
    status_html,
)
from ._protocol import _InteractiveExplorerProtocol
from ._utils import (
    apply_hoverlabels_matching_markers,
    figurewidget_safe_figure,
    format_elapsed_time,
    raw_index_from_customdata_row,
    schedule_after_plotly_event_loop,
    set_trace_hoverlabel_continuous,
    set_trace_hoverlabel_uniform,
)

logger = get_logger(__name__)


class PhenoMeInteractive:
    """
    High-performance interactive explorer for PhenoMe results.

    Optimized for large datasets using WebGL and efficient data handling.
    Provides a unified interface for 2D/3D exploration with:
      - Instant colour switching (no recomputation).
      - Highlight mode (shows all points, emphasises a subset).
      - Multi-select categorical highlight and live stats bar.
      - Click-to-inspect image viewer.
      - 2D box/lasso selection with CSV export of selected indices; HTML export
        and clear live in the Selection section (Plotly modebar: pan, zoom, box/lasso, PNG).
      - Dark mode for the plot (Appearance).

        Key design:
          - **Color changes** are instant (no recomputation, only visual update).
          - **Filter** and **Exclude** (separate collapsible sections under Embedding) restrict data before recomputation;
            same field + search + multi-select pattern as Highlight.
          - **Method/Source/Dim** or metadata filter changes trigger dimensionality reduction (expensive).
          - **Highlight mode** shows ALL data points but visually emphasises a matching
            subset (translucent halo + dimmed non-matching points) instead of hiding
            the rest.  This lets the user "find" a group in context.
          - **Multi-value highlight**: categorical fields use filter + multi-select (OR).
          - **Live stats bar** (total / highlighted / selected / lasso-box).
          - **Selected point**: clicking a point shows a black ring (hollow marker) for inspection.
          - **Multi-selection**: box / lasso on the 2D plot stores a persistent set of
            image indices that can be copied or downloaded as CSV.
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
        # Shallow copies: when set, DR always uses these instead of UI widget state.
        self._constructor_filters: dict | None = dict(filters) if filters else None
        self._constructor_exclude: dict | None = dict(exclude) if exclude else None
        self.initial_filters = dict(filters) if filters else {}
        self.initial_exclude = dict(exclude) if exclude else {}
        self._ui_filters = dict(filters) if filters else {}
        self._ui_exclude = dict(exclude) if exclude else {}
        self.hover_features = hover_features
        self.max_points = max_points

        # Cached computation results (set after first compute)
        self._cached_df: pd.DataFrame | None = None
        self._cached_method: str | None = None
        self._cached_filters: dict | None = None
        self._cached_exclude: dict | None = None
        self._cached_dr_obj: Any = None
        self._cached_source: str | None = None
        self._cached_ndims: int | None = None
        # Map pipeline image index -> row position in ``_cached_df`` (avoids repeated dict builds).
        self._cached_index_to_row: dict[int, int] | None = None
        # Incremented after each successful embedding compute; image-load callbacks skip stale work.
        self._compute_seq: int = 0

        # Widget state
        self.fig_widget: go.FigureWidget | None = None
        # Number of data traces from plotly express (excludes overlay traces).
        self._n_data_traces: int = 0
        self._current_color_by: str | None = None
        self._selected_point_index: int | None = None  # Click-to-highlight single point
        self._image_error_msg: str | None = None
        # Group highlight on/off (replaces former checkbox; synced to ``highlight_toggle`` button UI).
        self._highlight_active: bool = False
        # Box/lasso multi-selection: pipeline image indices currently selected (2D only).
        self._multi_selected_indices: list[int] = []
        # Theme state for Appearance dark-mode (plot) toggle.
        self._dark_mode: bool = False
        # Full list of categorical highlight values (for search filtering)
        self._highlight_value_options_all: list[str] = []
        # Pre-compute filter/exclude value lists (search narrows SelectMultiple like Highlight)
        self._filter_value_options_all: list[str] = []
        self._exclude_value_options_all: list[str] = []

        # Background compute (DR runs in a thread so the UI stays responsive)
        self._compute_thread: threading.Thread | None = None
        self._click_lock = threading.Lock()
        self._last_click_time = 0.0
        self._last_deselect_time = 0.0
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
            value=embedding_placeholder_idle(EMBEDDING_FIG_WIDTH_PX, EMBEDDING_FIG_HEIGHT_PX)
        )
        self._embedding_placeholder_computing = widgets.HTML(
            value=embedding_placeholder_computing(EMBEDDING_FIG_WIDTH_PX, EMBEDDING_FIG_HEIGHT_PX)
        )
        _pw = f"{EMBEDDING_FIG_WIDTH_PX}px"
        _ph = f"{EMBEDDING_FIG_HEIGHT_PX}px"
        self._plot_slot = widgets.VBox(
            [self._embedding_placeholder],
            layout=widgets.Layout(
                width=_pw,
                min_width=_pw,
                max_width=_pw,
                min_height=_ph,
            ),
        )
        _ow = f"{IMAGE_OVERLAY_WIDTH_PX}px"
        # Matplotlib inside ``ipywidgets.Output`` does not reliably render from Plotly
        # click callbacks (display goes to the cell / nowhere).  Use PNG +
        # ``widgets.Image`` instead, like ``_plot_slot.children`` for the embedding.
        self._img_idle_placeholder = widgets.HTML(
            value=image_panel_idle(IMAGE_OVERLAY_WIDTH_PX, IMAGE_PANEL_MIN_HEIGHT_PX)
        )
        self._img_loading_placeholder = widgets.HTML(
            value=image_panel_loading(IMAGE_OVERLAY_WIDTH_PX, IMAGE_PANEL_MIN_HEIGHT_PX)
        )
        _imh = f"{IMAGE_PANEL_MIN_HEIGHT_PX}px"
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

        # Build UI widgets (main controls + Selection accordion helpers)
        self._create_widgets()
        self._create_toolbar_widgets()

        # Populate options
        self._update_color_options()
        self._update_highlight_key_options()
        self._update_filter_exclude_options()
        self._update_filter_summary()
        self._update_exclude_summary()

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
        self._update_filter_summary()
        self._update_exclude_summary()

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
            [
                _embed,
                self._embedding_filter_accordion,
                self._embedding_exclude_accordion,
            ],
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
            [self.show_extra_info_checkbox, self.dark_mode_checkbox],
            layout=widgets.Layout(
                width="100%",
                padding="2px 0 0 70px",
                gap="16px",
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
        _selection_panel = widgets.VBox(
            [
                self._selection_help,
                self._selection_summary,
                widgets.HBox(
                    [
                        self.selection_copy_btn,
                        self.selection_download_btn,
                        self.export_html_btn,
                    ],
                    layout=widgets.Layout(
                        width="100%",
                        gap="8px",
                        flex_flow="row wrap",
                        align_items="center",
                        padding="4px 0 0 0",
                    ),
                ),
                self._selection_download_area,
            ],
            layout=widgets.Layout(
                width="100%",
                gap="8px",
                padding="10px 12px 12px 12px",
            ),
        )
        accordion = widgets.Accordion(
            children=[_embed_full, _appear, _hl, _selection_panel],
            layout=widgets.Layout(width="100%"),
        )
        accordion.set_title(0, "⊞ Embedding")
        accordion.set_title(1, "◑ Appearance")
        accordion.set_title(2, "◎ Highlight")
        accordion.set_title(3, "⬚ Selection (box / lasso)")
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

        # Plot column: figure slot only (Plotly modebar: pan, zoom, box, lasso, PNG, home).
        plot_column = widgets.VBox(
            [self.output_area],
            layout=widgets.Layout(
                gap="4px",
                width=f"{EMBEDDING_FIG_WIDTH_PX}px",
                min_width=f"{EMBEDDING_FIG_WIDTH_PX}px",
                max_width=f"{EMBEDDING_FIG_WIDTH_PX}px",
                align_items="flex-start",
            ),
        )
        main_area = widgets.HBox(
            [plot_column, self.img_output],
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
        self._update_selection_summary()

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
        self._constructor_filters = dict(filters) if filters else None
        self._constructor_exclude = dict(exclude) if exclude else None
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
        _ac = INTERACTIVE_APPEAR_COL_W_PX
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
            options=CONTINUOUS_SCALES,
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
            value=status_html("Ready", "info"),
            layout=widgets.Layout(flex="0 1 auto"),
        )
        self.stats_bar = widgets.HTML(
            value=stats_bar_html(0, 0, None, None),
            layout=widgets.Layout(flex="1 1 200px"),
        )

        # Image viewer + dark mode checkbox (paired in "Appearance" panel)
        self.show_extra_info_checkbox = widgets.Checkbox(
            value=True,
            description="Image extra info",
            indent=False,
            layout=widgets.Layout(
                min_width="180px",
                max_width=f"{INTERACTIVE_FIELD_MAX_W_PX}px",
            ),
        )
        self.dark_mode_checkbox = widgets.Checkbox(
            value=False,
            description="Dark mode",
            indent=False,
            layout=widgets.Layout(
                min_width="120px",
                max_width="200px",
            ),
        )

    # ------------------------------------------------------------------
    # Selection section widgets (export HTML / copy / CSV / clear)
    # ------------------------------------------------------------------
    def _create_toolbar_widgets(self) -> None:
        """Create widgets for the Selection accordion (box/lasso tools live on Plotly's modebar)."""
        # --- Selection (box / lasso) accordion panel widgets ---
        self._selection_help = widgets.HTML(
            value=(
                '<div style="font-size:11px;color:#5D6D7E;line-height:1.45;">'
                "Use the Plotly <b>modebar</b> above the figure (e.g. box or lasso select) "
                "to select multiple points (<b>2D only</b>). <b>Double-click the plot to clear selections.</b> "
                "</div>"
            ),
            layout=widgets.Layout(width="100%"),
        )
        self._selection_summary = widgets.HTML(
            value=('<div style="font-size:11px;color:#64748B;"><i>No points selected.</i></div>'),
            layout=widgets.Layout(width="100%"),
        )
        self.selection_copy_btn = widgets.Button(
            description="Copy indices",
            icon="clipboard",
            button_style="info",
            tooltip="Copy selected pipeline indices (comma separated)",
            layout=widgets.Layout(width="auto", min_width="140px"),
        )
        self.selection_download_btn = widgets.Button(
            description="Download CSV",
            icon="download",
            button_style="info",
            tooltip="Download selected rows of the embedding DataFrame as CSV",
            layout=widgets.Layout(width="auto", min_width="150px"),
        )
        self.export_html_btn = widgets.Button(
            description="Download HTML",
            icon="file-code-o",
            button_style="",
            tooltip="Download current figure as a standalone HTML file",
            layout=widgets.Layout(width="auto", min_width="150px"),
        )
        self._selection_download_area = widgets.HTML(value="", layout=widgets.Layout(width="100%"))

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
                select_widget.value = (filtered[0],)
        else:
            select_widget.value = ()

    def _create_filter_exclude_widgets(self) -> None:
        """Create embedding filter/exclude controls (same layout pattern as Highlight)."""
        _vh = INTERACTIVE_VALUES_H
        _sw = INTERACTIVE_SEARCH_W_PX
        _vw = INTERACTIVE_VALUES_W_PX
        _fmax = INTERACTIVE_FIELD_MAX_W_PX
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
                "Press <b>Compute</b> to apply.</div>"
            ),
            layout=widgets.Layout(width="100%"),
        )
        self._exclude_help = widgets.HTML(
            value=(
                '<div style="font-size:11px;color:#5D6D7E;line-height:1.45;">'
                "<b>Exclude</b> removes rows matching any selected value. "
                "Press <b>Compute</b> to apply.</div>"
            ),
            layout=widgets.Layout(width="100%"),
        )
        self.filter_key_dropdown = widgets.Dropdown(
            description="Field:",
            style=INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(width="100%", min_width="0", max_width=f"{_fmax}px"),
        )
        self.filter_search_input = widgets.Text(
            value="",
            description="Search:",
            placeholder="Search values…",
            style=INTERACTIVE_DESC_STYLE,
            layout=_fe_search_layout,
        )
        self.filter_value_select = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Values:",
            style=INTERACTIVE_DESC_STYLE,
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
            style=INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(width="100%", min_width="0", max_width=f"{_fmax}px"),
        )
        self.exclude_search_input = widgets.Text(
            value="",
            description="Search:",
            placeholder="Search values…",
            style=INTERACTIVE_DESC_STYLE,
            layout=_fe_search_layout,
        )
        self.exclude_value_select = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Values:",
            style=INTERACTIVE_DESC_STYLE,
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
                    'margin:0 0 2px 0;padding-bottom:4px;border-bottom:1px solid #E2E8F0;">'
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
        self.filter_clear_btn = widgets.Button(
            description="Clear All",
            button_style="warning",
            icon="trash",
            layout=widgets.Layout(width="110px"),
            tooltip="Clear all active filters",
        )
        self.filter_add_btn = widgets.Button(
            description="Add Filter",
            button_style="info",
            icon="plus",
            layout=widgets.Layout(width="110px"),
            tooltip="Add currently selected values to filters",
        )
        self.filter_summary = widgets.HTML(
            value='<div style="font-size:11px; color:#64748B;"><i>No active filters</i></div>',
            layout=widgets.Layout(width="100%", padding="4px 0"),
        )

        self.exclude_clear_btn = widgets.Button(
            description="Clear All",
            button_style="warning",
            icon="trash",
            layout=widgets.Layout(width="110px"),
            tooltip="Clear all active exclusions",
        )
        self.exclude_add_btn = widgets.Button(
            description="Add Exclude",
            button_style="info",
            icon="plus",
            layout=widgets.Layout(width="110px"),
            tooltip="Add currently selected values to exclusions",
        )
        self.exclude_summary = widgets.HTML(
            value='<div style="font-size:11px; color:#64748B;"><i>No active exclusions</i></div>',
            layout=widgets.Layout(width="100%", padding="4px 0"),
        )

        self._filter_accordion_panel = widgets.VBox(
            [
                self._filter_help,
                self._filter_include_block,
                widgets.HBox(
                    [self.filter_add_btn, self.filter_clear_btn],
                    layout=widgets.Layout(width="100%", gap="8px", padding="4px 0 0 70px"),
                ),
                widgets.HTML(
                    '<div style="font-size:10px;font-weight:600;color:#334155;margin-top:8px;">Active Filters:</div>'
                ),
                self.filter_summary,
            ],
            layout=widgets.Layout(width="100%", gap="4px", padding="10px 12px 12px 12px"),
        )
        self._exclude_accordion_panel = widgets.VBox(
            [
                self._exclude_help,
                self._filter_exclude_block,
                widgets.HBox(
                    [self.exclude_add_btn, self.exclude_clear_btn],
                    layout=widgets.Layout(width="100%", gap="8px", padding="4px 0 0 70px"),
                ),
                widgets.HTML(
                    '<div style="font-size:10px;font-weight:600;color:#334155;margin-top:8px;">Active Exclusions:</div>'
                ),
                self.exclude_summary,
            ],
            layout=widgets.Layout(width="100%", gap="4px", padding="10px 12px 12px 12px"),
        )
        self._embedding_filter_accordion = widgets.Accordion(
            children=[self._filter_accordion_panel],
            layout=widgets.Layout(width="100%"),
        )
        self._embedding_filter_accordion.set_title(0, "Filter")
        self._embedding_filter_accordion.selected_index = None
        self._embedding_exclude_accordion = widgets.Accordion(
            children=[self._exclude_accordion_panel],
            layout=widgets.Layout(width="100%"),
        )
        self._embedding_exclude_accordion.set_title(0, "Exclude")
        self._embedding_exclude_accordion.selected_index = None

    def _create_highlight_widgets(self) -> None:
        """Create highlight/filter widgets."""
        _values_h = INTERACTIVE_VALUES_H
        _sw = INTERACTIVE_SEARCH_W_PX
        _vw = INTERACTIVE_VALUES_W_PX
        _fmax = INTERACTIVE_FIELD_MAX_W_PX
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
            style=INTERACTIVE_DESC_STYLE,
            layout=widgets.Layout(width="100%", min_width="0", max_width=f"{_fmax}px"),
        )
        self.highlight_search_input = widgets.Text(
            value="",
            description="Search:",
            placeholder="Search values…",
            style=INTERACTIVE_DESC_STYLE,
            layout=_hl_search_layout,
        )
        self.highlight_value_select = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Values:",
            style=INTERACTIVE_DESC_STYLE,
            layout=_hl_values_layout,
        )
        self.highlight_range_slider = widgets.FloatRangeSlider(
            value=(0.0, 1.0),
            min=0.0,
            max=1.0,
            step=0.01,
            description="Range:",
            style=INTERACTIVE_DESC_STYLE,
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
            value=INTERACTIVE_SECTION_SEP_HTML,
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
        self.filter_add_btn.on_click(self._on_filter_add_clicked)
        self.exclude_clear_btn.on_click(self._on_exclude_clear_clicked)
        self.exclude_add_btn.on_click(self._on_exclude_add_clicked)
        self.highlight_key_dropdown.observe(self._on_highlight_key_changed, names="value")
        self.highlight_search_input.observe(self._on_highlight_search_changed, names="value")
        self.highlight_value_select.observe(self._on_highlight_value_changed, names="value")
        self.highlight_range_slider.observe(self._on_highlight_value_changed, names="value")
        self.highlight_toggle.on_click(self._on_highlight_click)
        self.point_size_slider.observe(self._on_marker_style_changed, names="value")
        self.opacity_slider.observe(self._on_marker_style_changed, names="value")
        self.dark_mode_checkbox.observe(self._on_dark_mode_changed, names="value")

        # Selection section callbacks
        self.selection_copy_btn.on_click(self._on_selection_copy_clicked)
        self.selection_download_btn.on_click(self._on_selection_download_clicked)
        self.export_html_btn.on_click(self._on_export_html_clicked)

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

        # Filter out non-informative columns (all NaNs, all same, or unique per row).
        informative_cols = []
        results = self.pheno.results
        n_total = results.n_images

        for col in cols:
            vals = []
            is_meta = col in get_all_metadata_keys(results)
            if is_meta:
                for i in range(min(n_total, 1000)):
                    meta = results.metadata[i]
                    if isinstance(meta, dict):
                        v = get_metadata_value_from_dict(meta, col)
                        if v is not None:
                            vals.append(v)
            else:
                for i in range(min(n_total, 1000)):
                    props = results.properties[i]
                    if isinstance(props, dict):
                        v = props.get(col)
                        if v is not None and not (
                            isinstance(v, (float, np.floating)) and np.isnan(v)
                        ):
                            vals.append(v)

            if not vals:
                continue

            unique_vals = set(vals)
            if len(unique_vals) <= 1:
                continue

            # Unique-per-row id-like columns (id, filename, path) are not informative for colouring.
            if (
                len(unique_vals) > n_total * 0.9
                and n_total > 10
                and col.lower() in ("id", "filename", "file_path", "path", "index")
            ):
                continue

            informative_cols.append(col)

        cols = sorted(informative_cols)
        self.color_dropdown.options = cols

        for default in ["drug", "cluster", "treatment", "condition", "time"]:
            if default in cols:
                self.color_dropdown.value = default
                return
        if cols:
            self.color_dropdown.value = cols[0]

    def _update_highlight_key_options(self) -> None:
        """Populate highlight field dropdown from cached DataFrame or pipeline."""
        if self._cached_df is not None:
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
        that list once and use ``get_metadata_value_from_dict`` per row (avoids
        re-fetching the metadata list on every index as ``get_metadata_value`` does).
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
        """Set ``initial_filters`` / ``initial_exclude`` from UI state or constructor overrides."""
        ui_filters = dict(self._ui_filters) if self._ui_filters else None
        ui_exclude = dict(self._ui_exclude) if self._ui_exclude else None

        fk = self.filter_key_dropdown.value
        fv = list(self.filter_value_select.value)
        if fk and fv:
            if ui_filters is None:
                ui_filters = {}
            if fk not in ui_filters:
                ui_filters[fk] = fv

        ek = self.exclude_key_dropdown.value
        ev = list(self.exclude_value_select.value)
        if ek and ev:
            if ui_exclude is None:
                ui_exclude = {}
            if ek not in ui_exclude:
                ui_exclude[ek] = ev

        self.initial_filters = (
            self._constructor_filters if self._constructor_filters is not None else ui_filters
        )
        self.initial_exclude = (
            self._constructor_exclude if self._constructor_exclude is not None else ui_exclude
        )

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

        is_bool = pd.api.types.is_bool_dtype(col)
        is_numeric = pd.api.types.is_numeric_dtype(col) and not is_bool
        n_unique_non_na = int(col.dropna().nunique()) if is_numeric else 0
        # Cluster-like integer columns: multi-select, not a range slider.
        discrete_int_categorical = (
            is_numeric
            and pd.api.types.is_integer_dtype(col)
            and n_unique_non_na <= HIGHLIGHT_DISCRETE_INT_MAX_UNIQUES
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
            if col_min == col_max:
                col_max = col_min + 1.0

            span = col_max - col_min
            step = round(span / 200, 6) if span > 0 else 0.01

            if span >= 100:
                readout_fmt = ".1f"
            elif span >= 1:
                readout_fmt = ".2f"
            elif span >= 0.01:
                readout_fmt = ".4f"
            else:
                readout_fmt = ".6f"

            # Update slider bounds.  Order matters because traitlets enforces
            # min <= value <= max at every step; widen first, then narrow.
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
            # The value change implicitly schedules the highlight update if it changed.

        self._debounce_timer = threading.Timer(SEARCH_DEBOUNCE_SEC, _do_search)
        self._debounce_timer.start()

    # ------------------------------------------------------------------
    # Compute (expensive - dimensionality reduction)
    # ------------------------------------------------------------------
    def _compute_embedding(self) -> float:
        """Run dimensionality reduction and populate ``self._cached_df``."""
        method_label = self.method_dropdown.value
        method_key = method_label.lower().replace("-", "")
        n_dims = self.dim_toggle.value
        source = self.source_dropdown.value

        self._sync_embed_filters_from_widgets()

        t0 = time.time()
        try:
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
        except ImportError as e:
            if "umap" in str(e).lower():
                raise ImportError(
                    "UMAP is not installed. Please install it with: pip install umap-learn"
                ) from e
            if "torchdr" in str(e).lower():
                raise ImportError(
                    "TorchDR is not installed. Please install it with: pip install torchdr"
                ) from e
            raise e
        elapsed = time.time() - t0

        if df is None:
            raise ValueError("Dimensionality reduction returned no data.")

        if len(df) > self.max_points:
            df = df.sample(n=self.max_points, random_state=DR_RANDOM_STATE).copy()

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

        # Recomputation invalidates the previous multi-selection (row positions shift).
        self._multi_selected_indices = []

        return elapsed

    def _refresh_index_row_map(self) -> None:
        """Build ``Index`` column -> row index map for O(1) lookups (highlight / overlay)."""
        df = self._cached_df
        if df is None or "Index" not in df.columns:
            self._cached_index_to_row = None
            return
        idx_arr = df["Index"].to_numpy()
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
        """Return plotly express data traces only (exclude overlay traces)."""
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

    def _multi_select_overlay_trace(self) -> Any | None:
        """The extra trace used to outline box/lasso multi-selected points (2D only)."""
        if self.fig_widget is None or self._n_data_traces <= 0:
            return None
        if len(self.fig_widget.data) <= self._n_data_traces + 1:
            return None
        return self.fig_widget.data[self._n_data_traces + 1]

    def _highlight_overlay_trace(self) -> Any | None:
        """Halo for group-Highlight (2D only; empty stub in 3D builds)."""
        if self.fig_widget is None or self._n_data_traces <= 0:
            return None
        if len(self.fig_widget.data) <= self._n_data_traces + 2:
            return None
        return self.fig_widget.data[self._n_data_traces + 2]

    def _highlight_marker_size(self) -> int:
        """Halo marker diameter (multi-select + group-highlight overlays) vs base point size."""
        ns = int(self.point_size_slider.value)
        return max(int(ns * HALO_SIZE_MULTIPLIER), ns + 2)

    def _click_select_ring_size(self) -> int:
        """Ring marker size for single-point click (classic hollow + border, not a filled halo)."""
        ns = int(self.point_size_slider.value)
        return max(ns + 5, int(ns * 2))

    def _make_selection_overlay_trace(self, is_3d: bool) -> go.Scatter3d | go.Scatter:
        """Single extra trace: hollow marker with black border (click-to-select; WebGL has no per-point line width on data)."""
        hs = self._click_select_ring_size()
        marker = {
            "size": hs,
            "color": "rgba(0,0,0,0)",
            "opacity": 1.0,
            "line": {"color": "black", "width": 2},
        }
        if is_3d:
            return go.Scatter3d(
                name=SELECTION_OVERLAY_NAME,
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
            name=SELECTION_OVERLAY_NAME,
            x=[],
            y=[],
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            visible=False,
            marker=marker,
            customdata=[],
        )

    def _make_multi_select_overlay_trace(self) -> go.Scatter:
        """Extra 2D trace: translucent blue halos for box/lasso multi-selected points."""
        hs = self._highlight_marker_size()
        marker = {
            "size": hs,
            "color": HALO_COLOR_MULTI,
            "opacity": 1.0,
            "line": {"width": 0, "color": "rgba(0,0,0,0)"},
        }
        return go.Scatter(
            name=MULTI_SELECT_OVERLAY_NAME,
            x=[],
            y=[],
            mode="markers",
            showlegend=False,
            hoverinfo="skip",
            visible=False,
            marker=marker,
            customdata=[],
        )

    def _make_highlight_overlay_trace(self) -> go.Scatter:
        """Extra 2D trace: pink halos for group-Highlight over matching rows."""
        hs = self._highlight_marker_size()
        marker = {
            "size": hs,
            "color": HALO_COLOR_HIGHLIGHT,
            "opacity": 1.0,
            "line": {"width": 0, "color": "rgba(0,0,0,0)"},
        }
        return go.Scatter(
            name=HIGHLIGHT_OVERLAY_NAME,
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

    def _figure_theme_layout(self) -> dict[str, Any]:
        """Return a layout dict patch for the current theme (light/dark)."""
        if self._dark_mode:
            return {
                "template": "plotly_dark",
                "paper_bgcolor": "#111827",
                "plot_bgcolor": "#1f2937",
                "font_color": "#e5e7eb",
            }
        return {
            "template": "plotly_white",
            "paper_bgcolor": "white",
            "plot_bgcolor": "#FAFCFE",
            "font_color": "#141428",
        }

    def _build_figure(self) -> None:
        """Build a new FigureWidget from the cached DataFrame."""
        df = self._cached_df
        if df is None:
            return

        theme = self._figure_theme_layout()

        if len(df) == 0:
            # No data - show empty placeholder and clear selection
            self._selected_point_index = None
            self._multi_selected_indices = []
            self._n_data_traces = 0
            self.fig_widget = go.FigureWidget(
                layout=go.Layout(
                    template=theme["template"],
                    autosize=False,
                    width=EMBEDDING_FIG_WIDTH_PX,
                    height=EMBEDDING_FIG_HEIGHT_PX,
                    title={
                        "text": "<b>No data</b>",
                        "x": 0.5,
                        "xanchor": "center",
                        "y": 0.98,
                        "yanchor": "top",
                        "font": {
                            "size": 17,
                            "family": "Inter, Helvetica Neue, Arial, sans-serif",
                            "color": theme["font_color"],
                        },
                    },
                    paper_bgcolor=theme["paper_bgcolor"],
                    plot_bgcolor=theme["plot_bgcolor"],
                    xaxis={"visible": False},
                    yaxis={"visible": False},
                    annotations=[
                        {
                            "text": "Compute embedding to see data",
                            "showarrow": False,
                            "font": {"color": theme["font_color"]},
                        }
                    ],
                    margin={"l": 50, "r": 50, "t": 72, "b": 50},
                )
            )
            apply_figurewidget_display_config(self.fig_widget)
            return
        x_col, y_col, z_col = self._coord_columns()
        color_by = self.color_dropdown.value

        color_column: str | None = None
        is_continuous = False
        if color_by:
            try:
                color_column, is_continuous = self.pheno._get_color_column(color_by, df)
            except ValueError:
                color_column = None

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
            "custom_data": ["Index"],
            "width": EMBEDDING_FIG_WIDTH_PX,
            "height": EMBEDDING_FIG_HEIGHT_PX,
        }
        if z_col:
            kwargs["z"] = z_col

        if color_column:
            kwargs["color"] = color_column
            if is_continuous:
                kwargs["color_continuous_scale"] = self.colorscale_dropdown.value
            else:
                kwargs["category_orders"] = {
                    color_column: sorted(df[color_column].dropna().unique(), key=str)
                }

        fig = px.scatter_3d(**kwargs) if z_col else px.scatter(**kwargs)

        if hover_cols:
            ht_parts = [f"{col}=%{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]
            fig.update_traces(hovertemplate="<br>".join(ht_parts) + "<extra></extra>")

        marker_size = self.point_size_slider.value
        opacity = self.opacity_slider.value
        fig.update_traces(marker={"size": marker_size, "opacity": opacity, "line": {"width": 0}})
        if not color_column:
            fig.update_traces(marker_color="#636EFA")

        apply_hoverlabels_matching_markers(
            fig,
            color_column=color_column,
            is_continuous=is_continuous,
            colorscale_name=self.colorscale_dropdown.value,
            fallback_uniform="#636EFA",
        )

        _font = {"family": "Inter, Helvetica Neue, Arial, sans-serif", "size": 12}
        _grid_color = "#2d3748" if self._dark_mode else "#EBEBEB"
        _axis_2d = {
            "showline": False,
            "zeroline": False,
            "gridcolor": _grid_color,
            "ticks": "",
        }
        _hoverlabel = {
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        }
        _legend_bg = "rgba(31,41,55,0.92)" if self._dark_mode else "rgba(248,250,252,0.92)"
        _legend_border = "#374151" if self._dark_mode else "#E2E8F0"
        _legend_font_color = "#e5e7eb" if self._dark_mode else "#334155"
        _legend = {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.14,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": _legend_bg,
            "bordercolor": _legend_border,
            "borderwidth": 1,
            "font": {"size": 10, "family": _font["family"], "color": _legend_font_color},
            "itemsizing": "constant",
            "tracegroupgap": 8,
            "itemwidth": 30,
        }
        if color_column and not is_continuous:
            _legend["title"] = {
                "text": color_column,
                "side": "top center",
                "font": {"size": 11, "family": _font["family"], "color": _legend_font_color},
            }

        source_label = self._cached_source if self._cached_source else "?"
        title_main = f"{self._cached_method} ({source_label})"
        _title_color = theme["font_color"]
        _subtitle_color = "#cbd5f5" if self._dark_mode else "#64748b"
        title_html = (
            f'<span style="font-size:17px;font-weight:600;letter-spacing:-0.02em;color:{_title_color};">'
            f"{html.escape(title_main)}</span>"
        )
        if color_by:
            title_html += (
                f'<br><span style="font-size:12px;font-weight:500;color:{_subtitle_color};">'
                f'Colored by <span style="color:{_title_color};">{html.escape(str(color_by))}</span></span>'
            )
        _title_layout = {
            "text": title_html,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"t": 6},
            "font": {"family": _font["family"], "size": 14, "color": _title_color},
        }

        x_title, y_title, z_title = x_col, y_col, z_col
        if self._cached_method == "PCA" and self._cached_dr_obj is not None:
            pca = self._cached_dr_obj
            x_title = f"Component 1 ({pca.explained_variance_ratio_[0]:.1%})"
            y_title = f"Component 2 ({pca.explained_variance_ratio_[1]:.1%})"
            if z_col:
                z_title = f"Component 3 ({pca.explained_variance_ratio_[2]:.1%})"

        _drag_mode = "pan"
        if z_col:
            _scene_bg = "#111827" if self._dark_mode else "#FAFCFE"
            _scene_grid = "#1f2937" if self._dark_mode else "#E8ECF0"
            _scene_axis = {
                "showbackground": True,
                "backgroundcolor": _scene_bg,
                "gridcolor": _scene_grid,
                "showline": False,
                "zeroline": False,
                "ticks": "",
            }
            fig.update_layout(
                template=theme["template"],
                font=_font,
                paper_bgcolor=theme["paper_bgcolor"],
                hovermode="closest",
                autosize=False,
                title=_title_layout,
                margin={"l": 50, "r": 50, "t": 56, "b": 124},
                hoverlabel=_hoverlabel,
                legend=_legend,
                scene={
                    "bgcolor": _scene_bg,
                    "xaxis": {**_scene_axis, "title": x_title},
                    "yaxis": {**_scene_axis, "title": y_title},
                    "zaxis": {**_scene_axis, "title": z_title},
                },
            )
        else:
            fig.update_layout(
                template=theme["template"],
                font=_font,
                paper_bgcolor=theme["paper_bgcolor"],
                plot_bgcolor=theme["plot_bgcolor"],
                hovermode="closest",
                autosize=False,
                xaxis={**_axis_2d, "title": x_title},
                yaxis={**_axis_2d, "title": y_title},
                title=_title_layout,
                margin={"l": 50, "r": 50, "t": 56, "b": 124},
                hoverlabel=_hoverlabel,
                legend=_legend,
                dragmode=_drag_mode if _drag_mode else None,
            )

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
                        "title": {
                            "text": color_column,
                            "side": "top",
                            "font": {"size": 12, "color": _legend_font_color},
                        },
                        "tickfont": {"size": 10, "color": _legend_font_color},
                        "ticklen": 5,
                        "tickcolor": _legend_border,
                        "outlinewidth": 1,
                        "outlinecolor": _legend_border,
                    }
                },
                margin_b=max(fig.layout.margin.b or 124, 152),
            )

        fig.update_layout(
            autosize=False,
            width=EMBEDDING_FIG_WIDTH_PX,
            height=EMBEDDING_FIG_HEIGHT_PX,
        )
        # Convert to FigureWidget (strip numpy from trace/layout for widget sync).
        self.fig_widget = go.FigureWidget(figurewidget_safe_figure(fig))
        apply_figurewidget_display_config(self.fig_widget)
        self._n_data_traces = len(self.fig_widget.data)
        self._current_color_by = color_by

        # Overlay traces: click = ring on a separate trace; box/lasso + group-Highlight = halos.
        # scattergl/scatter3d do not support per-point line width on data; use extra traces.
        self.fig_widget.add_trace(self._make_selection_overlay_trace(is_3d=bool(z_col)))
        # Multi-select overlay (2D only). Add an empty 3D-compatible trace otherwise
        # so the overlay index stays consistent across builds.
        if z_col:
            self.fig_widget.add_trace(
                go.Scatter3d(
                    name=MULTI_SELECT_OVERLAY_NAME,
                    x=[],
                    y=[],
                    z=[],
                    mode="markers",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=False,
                    marker={"size": 1, "color": "rgba(0,0,0,0)"},
                    customdata=[],
                )
            )
            # Group highlight halo: 2D only; in 3D keep a stub so overlay indices match.
            self.fig_widget.add_trace(
                go.Scatter3d(
                    name=HIGHLIGHT_OVERLAY_NAME,
                    x=[],
                    y=[],
                    z=[],
                    mode="markers",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=False,
                    marker={"size": 1, "color": "rgba(0,0,0,0)"},
                    customdata=[],
                )
            )
        else:
            self.fig_widget.add_trace(self._make_multi_select_overlay_trace())
            self.fig_widget.add_trace(self._make_highlight_overlay_trace())

        if self._highlight_active:
            self._apply_highlight()

        if self._selected_point_index is not None:
            self._apply_selected_point_highlight()

        # Click + 2D selection / deselect callbacks (data traces only; not overlay halos)
        if self.fig_widget is not None:
            for trace in self._get_data_traces():
                trace.on_click(self._on_figure_click)
            # Box/lasso: selection events; double-click clear fires plotly_deselect (2D only).
            if not z_col:
                try:
                    for trace in self._get_data_traces():
                        trace.on_selection(self._on_figure_selection)
                        trace.on_deselect(self._on_figure_deselect)
                except Exception:
                    logger.debug("Could not wire selection/deselect callbacks", exc_info=True)

        # Repaint persistent multi-selection if still applicable.
        self._apply_multi_select_overlay()

    # ------------------------------------------------------------------
    # Instant colour update (no recomputation)
    # ------------------------------------------------------------------
    def _try_patch_continuous_recolor(self, color_column: str) -> bool:
        """Update continuous coloring in-place (preserves zoom/pan).

        Plotly express uses one data trace + ``layout.coloraxis`` for continuous
        colour. Categorical plots use one trace per category; those must still
        go through ``_build_figure``.

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
                set_trace_hoverlabel_continuous(tr, self.colorscale_dropdown.value)
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
            u = "#636EFA"
            with self.fig_widget.batch_update():
                for trace in self._get_data_traces():
                    trace.marker.color = u
                    set_trace_hoverlabel_uniform(trace, u)
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
    # Click / selection logic
    # ------------------------------------------------------------------
    def _resolve_click_point_index(self, trace: Any, points: Any) -> int | None:
        """Marker index within ``trace`` for a click (handles empty ``point_inds`` on WebGL).

        Expected shapes:
            ``trace.x`` / ``trace.y``: length ``n_points`` coordinate arrays.
            ``points.xs`` / ``points.ys``: click coordinates from the frontend.
        """
        if points is None:
            return None
        if hasattr(points, "point_inds") and points.point_inds:
            return int(points.point_inds[0])

        ttype = getattr(trace, "type", "")
        if ttype not in ("scatter", "scattergl", "scatter3d"):
            return None

        if not hasattr(points, "xs") or not points.xs or not hasattr(points, "ys") or not points.ys:
            return None

        try:
            x0, y0 = float(points.xs[0]), float(points.ys[0])
            tx = np.asarray(trace.x, dtype=float)
            ty = np.asarray(trace.y, dtype=float)
            if tx.size == 0 or ty.size == 0 or tx.shape != ty.shape:
                return None
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
        t_now = time.time()
        if t_now - self._last_click_time < CLICK_DEBOUNCE_SEC:
            return
        if not self._click_lock.acquire(blocking=False):
            return

        scheduled = False
        try:
            # Skip clicks on overlay traces themselves.
            if getattr(trace, "name", "") in (
                SELECTION_OVERLAY_NAME,
                MULTI_SELECT_OVERLAY_NAME,
                HIGHLIGHT_OVERLAY_NAME,
            ):
                return

            point_idx = self._resolve_click_point_index(trace, points)
            if point_idx is None:
                return

            try:
                if trace.customdata is None or point_idx >= len(trace.customdata):
                    return
                cdata = trace.customdata[point_idx]
                idx = raw_index_from_customdata_row(cdata)
                if idx < 0:
                    return
            except (TypeError, ValueError, IndexError):
                return

            self._last_click_time = t_now

            def _apply_click_safe() -> None:
                try:
                    self._image_error_msg = None
                    # Toggle selected point: same click = deselect, different = select
                    if self._selected_point_index == idx:
                        self._selected_point_index = None
                    else:
                        self._selected_point_index = idx

                    self._apply_selected_point_overlay()
                    self._update_stats()

                    if self._selected_point_index is None:
                        self._set_image_panel_idle()
                        return

                    sel_idx = self._selected_point_index
                    self.img_output.children = (self._img_loading_placeholder,)
                    seq_at_load = self._compute_seq

                    def _load_and_show_image_task() -> None:
                        try:
                            loading_idx = sel_idx
                            png, details_text, title = self.pheno.image_preview_png_bytes(
                                loading_idx,
                                apply_transforms=False,
                                downsample=max(360, IMAGE_OVERLAY_WIDTH_PX * 2),
                                show_extra_info=self.show_extra_info_checkbox.value,
                            )

                            def _update_ui() -> None:
                                if self._compute_seq != seq_at_load:
                                    return
                                if self._selected_point_index != loading_idx:
                                    return

                                preview = widgets.Image(
                                    value=png,
                                    format="png",
                                    layout=widgets.Layout(
                                        width="100%",
                                        max_width=f"{IMAGE_OVERLAY_WIDTH_PX}px",
                                        max_height="400px",
                                        object_fit="contain",
                                    ),
                                )

                                children_list = []

                                if title:
                                    title_widget = widgets.HTML(
                                        value=(
                                            "<div style='text-align:center; font-weight:bold; "
                                            "font-size:14px; margin-bottom:8px;'>"
                                            f"{html.escape(title)}</div>"
                                        ),
                                        layout=widgets.Layout(width="100%"),
                                    )
                                    children_list.append(title_widget)

                                children_list.append(preview)

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
                                    children_list.append(extra)

                                self.img_output.children = tuple(children_list)
                                self._update_stats()

                            schedule_after_plotly_event_loop(_update_ui)
                        except Exception as e:
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

                            schedule_after_plotly_event_loop(_update_err)

                    threading.Thread(target=_load_and_show_image_task, daemon=True).start()

                finally:
                    self._click_lock.release()

            schedule_after_plotly_event_loop(_apply_click_safe)
            scheduled = True

        except Exception:
            raise
        finally:
            if not scheduled:
                self._click_lock.release()

    def _on_figure_selection(self, trace: Any, points: Any, _state: Any) -> None:
        """Handle 2D box/lasso selection from FigureWidget.

        Selection events fire per-trace; we accumulate pipeline indices across
        all data traces into ``_multi_selected_indices`` and draw a single
        overlay. Called from the Plotly event thread → schedule the UI update.
        """
        if self.fig_widget is None or self._cached_df is None:
            return
        if getattr(trace, "name", "") in (
            SELECTION_OVERLAY_NAME,
            MULTI_SELECT_OVERLAY_NAME,
            HIGHLIGHT_OVERLAY_NAME,
        ):
            return

        if points is None or not hasattr(points, "point_inds") or not points.point_inds:
            # Empty selection (e.g. clicking outside) clears per-trace; only fully clear
            # when the user explicitly hits the Clear button or no traces are selected.
            return

        # Collect pipeline indices for the selected points on this trace.
        try:
            cd = trace.customdata
            if cd is None:
                return
            selected_ids = []
            for pi in points.point_inds:
                if pi < 0 or pi >= len(cd):
                    continue
                idx = raw_index_from_customdata_row(cd[pi])
                if idx >= 0:
                    selected_ids.append(idx)
        except (TypeError, ValueError, IndexError):
            return

        if not selected_ids:
            return

        # Merge across traces (deduplicate). Selections from multiple traces arrive via
        # consecutive callbacks; replace-on-different-trace gives best UX.
        existing = set(self._multi_selected_indices)
        merged = list(dict.fromkeys(list(existing) + selected_ids))
        if merged == self._multi_selected_indices:
            return

        self._multi_selected_indices = merged

        def _do() -> None:
            self._apply_multi_select_overlay()
            self._update_selection_summary()
            self._update_stats()
            self.status_label.value = status_html(
                f"Selected {len(self._multi_selected_indices):,} point(s)", "ok"
            )

        schedule_after_plotly_event_loop(_do)

    def _on_figure_deselect(self, trace: Any, _points: Any) -> None:
        """When the user double-clicks to clear Plotly's box/lasso, drop every kind of selection."""
        if getattr(trace, "name", "") in (
            SELECTION_OVERLAY_NAME,
            MULTI_SELECT_OVERLAY_NAME,
            HIGHLIGHT_OVERLAY_NAME,
        ):
            return
        t_now = time.time()
        if t_now - self._last_deselect_time < 0.08:
            return
        self._last_deselect_time = t_now

        self._clear_all_selections(reset_dragmode=False)

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
                    trace_indices = [raw_index_from_customdata_row(cd) for cd in trace.customdata]
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

        Does not modify the selection overlay trace; use ``_selection_overlay_trace`` for that.
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
        """Refresh the stats bar (total / highlighted / selected index / multi-selected)."""
        if self._cached_df is None:
            self.stats_bar.value = stats_bar_html(
                0,
                0,
                self._selected_point_index,
                getattr(self, "_image_error_msg", None),
                n_multi_selected=len(self._multi_selected_indices),
            )
            return
        n_total = len(self._cached_df)
        n_hl = 0
        if self._highlight_active:
            mask = self._build_highlight_mask()
            if mask is not None:
                n_hl = int(mask.sum())
        self.stats_bar.value = stats_bar_html(
            n_total,
            n_hl,
            self._selected_point_index,
            getattr(self, "_image_error_msg", None),
            n_multi_selected=len(self._multi_selected_indices),
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
            lo, hi = self.highlight_range_slider.value
            numeric_vals = pd.to_numeric(col_vals, errors="coerce")
            return ((numeric_vals >= lo) & (numeric_vals <= hi)).values
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
        """Dim non-matching points and show a pink halo on matches (or restore uniform when off)."""
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
            self._apply_highlight_halo_overlay()
            self._update_stats()
            return

        highlight_mask = self._build_highlight_mask()
        if highlight_mask is None:
            self._apply_highlight_halo_overlay()
            self._update_stats()
            return

        trace_masks = self._get_per_trace_masks(highlight_mask)
        normal_size = self.point_size_slider.value
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
            if has_any:
                op: Any = np.where(tm, normal_opacity, dim_opacity).tolist()
            else:
                op = dim_opacity
            sizes_l.append(normal_size)
            op_l.append(op)
            lw_l.append(0)
        self._apply_marker_style_to_traces(sizes_l, op_l, lw_l)
        self._apply_highlight_halo_overlay()
        self._update_stats()

    def _apply_highlight_halo_overlay(self) -> None:
        """Draw or clear the 2D pink halo for group-Highlight; always hidden in 3D (stub trace)."""
        if self.fig_widget is None or self._cached_df is None:
            return
        ov = self._highlight_overlay_trace()
        if ov is None:
            return
        _, _, z_col = self._coord_columns()
        if z_col:
            with self.fig_widget.batch_update():
                ov.visible = False
            return
        if not self._highlight_active:
            with self.fig_widget.batch_update():
                ov.x = []
                ov.y = []
                ov.customdata = []
                ov.visible = False
            return
        mask = self._build_highlight_mask()
        if mask is None or not np.any(mask):
            with self.fig_widget.batch_update():
                ov.x = []
                ov.y = []
                ov.customdata = []
                ov.visible = False
            return
        df = self._cached_df
        if "Index" not in df.columns:
            with self.fig_widget.batch_update():
                ov.visible = False
            return
        x_col, y_col, _ = self._coord_columns()
        xs: list[float] = []
        ys: list[float] = []
        cd: list[list[int]] = []
        for j in range(len(mask)):
            if not bool(mask[j]):
                continue
            row = df.iloc[j]
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
            cd.append([int(row["Index"])])
        with self.fig_widget.batch_update():
            ov.x = xs
            ov.y = ys
            ov.customdata = cd
            ov.marker.size = self._highlight_marker_size()
            ov.marker.color = HALO_COLOR_HIGHLIGHT
            ov.marker.line = {"width": 0, "color": "rgba(0,0,0,0)"}
            ov.visible = bool(xs)

    def _apply_selected_point_highlight(self) -> None:
        """Apply base styling and show selection via overlay trace."""
        self._apply_base_styling()
        self._apply_selected_point_overlay()
        self._apply_multi_select_overlay()
        self._update_stats()

    def _apply_base_styling(self) -> None:
        if self._highlight_active:
            self._apply_highlight()
        else:
            self._apply_uniform_data_traces()
            self._apply_highlight_halo_overlay()

    def _apply_selected_point_overlay(self) -> None:
        if self.fig_widget is None or self._cached_df is None:
            return

        ov = self._selection_overlay_trace()
        x_col, y_col, z_col = self._coord_columns()

        if self._selected_point_index is not None:
            m = self._cached_index_to_row
            if m is None or self._selected_point_index not in m:
                self._selected_point_index = None

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
                        ring_size = self._click_select_ring_size()
                        ov.marker.size = ring_size
                        ov.marker.color = "rgba(0,0,0,0)"
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

    def _apply_multi_select_overlay(self) -> None:
        """Paint the multi-selection overlay from ``_multi_selected_indices`` (2D only)."""
        if self.fig_widget is None or self._cached_df is None:
            return
        ov = self._multi_select_overlay_trace()
        if ov is None:
            return
        _, _, z_col = self._coord_columns()
        if z_col:
            # 3D: overlay trace exists but we don't support box/lasso in 3D.
            with self.fig_widget.batch_update():
                ov.visible = False
            return

        m = self._cached_index_to_row or {}
        df = self._cached_df
        x_col, y_col, _ = self._coord_columns()
        xs: list[float] = []
        ys: list[float] = []
        cd: list[list[int]] = []
        kept: list[int] = []
        for idx in self._multi_selected_indices:
            j = m.get(idx)
            if j is None:
                continue
            row = df.iloc[j]
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
            cd.append([int(idx)])
            kept.append(idx)

        # Keep list aligned with what we can actually display (strip stale ids).
        self._multi_selected_indices = kept

        with self.fig_widget.batch_update():
            ov.x = xs
            ov.y = ys
            ov.customdata = cd
            ov.marker.size = self._highlight_marker_size()
            ov.marker.color = HALO_COLOR_MULTI
            ov.visible = bool(xs)

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------
    def _on_compute_clicked(self, _btn: Any) -> None:
        """Handle Compute button click - runs DR and rebuilds figure.

        DR runs in a daemon thread so the UI stays responsive; a ticker updates
        elapsed time in the status label.  The embedding plot is mounted by
        updating ``_plot_slot.children`` so it still appears when compute
        finishes on that thread (see class docstring / Output widget threading
        notes).  Library warnings from DR may still print to the notebook or
        kernel log.
        """
        if self._compute_thread is not None and self._compute_thread.is_alive():
            return

        compute_done = threading.Event()
        t_start = time.time()
        self.compute_button.disabled = True
        self.status_label.value = status_html(f"Computing… {format_elapsed_time(0.0)}", "warn")
        # Full-size computing state so the plot area is not a blank gap while DR runs.
        self._plot_slot.children = (self._embedding_placeholder_computing,)

        def _tick() -> None:
            while True:
                if compute_done.wait(timeout=0.5):
                    break
                elapsed = time.time() - t_start
                self.status_label.value = status_html(
                    f"Computing… {format_elapsed_time(elapsed)}", "warn"
                )

        def _run() -> None:
            ticker = threading.Thread(target=_tick, daemon=True)
            ticker.start()
            elapsed = 0.0
            try:
                elapsed = self._compute_embedding()

                def _display_new_fig() -> None:
                    try:
                        with self._click_lock:
                            self._build_figure()
                            self._display_figure()
                            # Selection is cleared on recompute; refresh UI reflections.
                            self._update_selection_summary()
                    except Exception as e:
                        self.status_label.value = status_html(f"Display error: {e}", "err")

                schedule_after_plotly_event_loop(_display_new_fig)

                n = len(self._cached_df) if self._cached_df is not None else 0
                self.status_label.value = status_html(
                    f"Done — {n:,} points in {format_elapsed_time(elapsed)}", "ok"
                )
            except Exception as e:
                self.status_label.value = status_html(f"Error: {e}", "err")
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
        self.status_label.value = status_html("Updating colours…", "info")

        def _do() -> None:
            try:
                self._recolor_figure()
                self._apply_selected_point_highlight()
                self.status_label.value = status_html("Colour updated", "ok")
                self._update_stats()
            except Exception as e:
                self.status_label.value = status_html(f"Colour error: {e}", "err")

        schedule_after_plotly_event_loop(_do)

    def _on_colorscale_changed(self, change: Any) -> None:
        """Instant colourscale change for continuous variables."""
        if self._cached_df is None:
            return

        def _do() -> None:
            try:
                self._recolor_figure()
                self._apply_selected_point_highlight()
                self._update_stats()
            except Exception as e:
                self.status_label.value = status_html(f"Colourscale error: {e}", "err")

        schedule_after_plotly_event_loop(_do)

    def _on_highlight_key_changed(self, change: Any) -> None:
        """Update available values when highlight field changes."""
        self._update_highlight_value_options()
        if not (self._highlight_active and self.fig_widget is not None):
            return

        def _do() -> None:
            self._apply_highlight()
            self._update_highlight_status()

        schedule_after_plotly_event_loop(_do)

    def _sync_highlight_button_appearance(self) -> None:
        """Sync highlight button label/style with ``_highlight_active``."""
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

        schedule_after_plotly_event_loop(_do)

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
                self.status_label.value = status_html("Highlight off", "ok")
                self._update_stats()

        schedule_after_plotly_event_loop(_do)

    def _update_highlight_status(self) -> None:
        """Show how many points are highlighted in the status bar."""
        self._update_stats()
        mask = self._build_highlight_mask()
        if mask is not None:
            n_hl = int(mask.sum())
            n_total = len(mask)
            self.status_label.value = status_html(
                f"Highlighted {n_hl:,} / {n_total:,} points", "ok"
            )
        else:
            self.status_label.value = status_html("Highlight on", "ok")

    def _set_image_panel_idle(self) -> None:
        """Reset the click-to-inspect panel to the placeholder (no PNG)."""
        self.img_output.children = (self._img_idle_placeholder,)

    def _debounce_search(self, timer_attr: str, apply_fn: Callable[[], None]) -> None:
        """Cancel any pending timer and schedule ``apply_fn`` after ``SEARCH_DEBOUNCE_SEC``."""
        timer: threading.Timer | None = getattr(self, timer_attr)
        if timer is not None:
            timer.cancel()
        new_timer = threading.Timer(SEARCH_DEBOUNCE_SEC, apply_fn)
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
        self.status_label.value = status_html(
            f"{label}: {key}={values if values else 'None'}", "info"
        )

    def _on_filter_key_changed(self, change: Any) -> None:
        """Update available filter values when filter field changes."""
        self._constructor_filters = None
        self._update_filter_value_options()

    def _on_filter_value_changed(self, change: Any) -> None:
        """Sync pipeline filter dict when filter values change."""
        self._constructor_filters = None
        self._on_slot_value_changed("Filter", self.filter_key_dropdown, self.filter_value_select)

    def _on_exclude_key_changed(self, change: Any) -> None:
        """Update available exclude values when exclude field changes."""
        self._constructor_exclude = None
        self._update_exclude_value_options()

    def _on_exclude_value_changed(self, change: Any) -> None:
        """Sync pipeline exclude dict when exclude values change."""
        self._constructor_exclude = None
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

    def _on_filter_add_clicked(self, _btn: Any) -> None:
        """Add current selection to filters."""
        fk = self.filter_key_dropdown.value
        fv = list(self.filter_value_select.value)
        if fk and fv:
            self._ui_filters[fk] = fv
            self._update_filter_summary()
            self.status_label.value = status_html(f"Added filter: {fk}", "ok")

    def _on_exclude_add_clicked(self, _btn: Any) -> None:
        """Add current selection to exclusions."""
        ek = self.exclude_key_dropdown.value
        ev = list(self.exclude_value_select.value)
        if ek and ev:
            self._ui_exclude[ek] = ev
            self._update_exclude_summary()
            self.status_label.value = status_html(f"Added exclude: {ek}", "ok")

    def _on_filter_clear_clicked(self, _btn: Any) -> None:
        """Clear all filters (include) - recompute still required."""
        self._ui_filters = {}
        self._constructor_filters = None
        self.filter_search_input.value = ""
        self.filter_value_select.value = ()
        self._update_filter_value_options(select_default=False)
        self._update_filter_summary()
        self.status_label.value = status_html("All filters cleared", "info")

    def _on_exclude_clear_clicked(self, _btn: Any) -> None:
        """Clear all exclusions - recompute still required."""
        self._ui_exclude = {}
        self._constructor_exclude = None
        self.exclude_search_input.value = ""
        self.exclude_value_select.value = ()
        self._update_exclude_value_options(select_default=False)
        self._update_exclude_summary()
        self.status_label.value = status_html("All exclusions cleared", "info")

    def _update_filter_summary(self) -> None:
        """Update the active filters HTML summary."""
        if not self._ui_filters:
            self.filter_summary.value = (
                '<div style="font-size:11px; color:#64748B;"><i>No active filters</i></div>'
            )
        else:
            items = []
            for k, v in self._ui_filters.items():
                val_str = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                items.append(f"<b>{k}</b>: {val_str}")
            self.filter_summary.value = (
                '<div style="font-size:11px; color:#475569; line-height:1.4;">'
                + "<br>".join(items)
                + "</div>"
            )

    def _update_exclude_summary(self) -> None:
        """Update the active exclusions HTML summary."""
        if not self._ui_exclude:
            self.exclude_summary.value = (
                '<div style="font-size:11px; color:#64748B;"><i>No active exclusions</i></div>'
            )
        else:
            items = []
            for k, v in self._ui_exclude.items():
                val_str = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                items.append(f"<b>{k}</b>: {val_str}")
            self.exclude_summary.value = (
                '<div style="font-size:11px; color:#475569; line-height:1.4;">'
                + "<br>".join(items)
                + "</div>"
            )

    def _on_marker_style_changed(self, change: Any) -> None:
        """Adjust marker size / opacity without recomputation."""
        if self.fig_widget is None:
            return

        def _do() -> None:
            fig = self.fig_widget
            if fig is None:
                return
            self._apply_selected_point_highlight()

        schedule_after_plotly_event_loop(_do)

    def _on_dark_mode_changed(self, change: Any) -> None:
        """Toggle dark/light theme (requires a rebuild to change plotly template)."""
        new_val = bool(change.get("new", change.get("owner").value if change else False))
        self._dark_mode = new_val

        if self._cached_df is None:
            return

        def _do() -> None:
            try:
                with self._click_lock:
                    self._build_figure()
                    self._display_figure()
                self.status_label.value = status_html(
                    f"Theme: {'dark' if self._dark_mode else 'light'}", "ok"
                )
            except Exception as e:
                self.status_label.value = status_html(f"Theme error: {e}", "err")

        schedule_after_plotly_event_loop(_do)

    # ------------------------------------------------------------------
    # Download helpers (Selection panel)
    # ------------------------------------------------------------------
    def _data_uri_html(self, label: str, filename: str, mimetype: str, payload: bytes) -> str:
        """Return a compact "click to download" HTML link backed by a base64 data URI."""
        import base64

        b64 = base64.b64encode(payload).decode("ascii")
        esc_label = html.escape(label)
        esc_name = html.escape(filename)
        return (
            f'<a download="{esc_name}" href="data:{mimetype};base64,{b64}" '
            f'style="display:inline-block;margin-left:4px;padding:2px 8px;border-radius:4px;'
            f"background:#EAFAF1;border:1px solid #27ae60;color:#1e8449;"
            f'font-size:11px;text-decoration:none;">⬇ {esc_label}</a>'
        )

    def _on_export_html_clicked(self, _btn: Any) -> None:
        """Save the current figure as a standalone HTML file and offer a download link."""
        fig = self.fig_widget
        if fig is None:
            self.status_label.value = status_html("No figure to export", "warn")
            return
        try:
            buf = io.StringIO()
            fig.write_html(buf, include_plotlyjs="cdn", full_html=True)
            html_bytes = buf.getvalue().encode("utf-8")
            link = self._data_uri_html(
                "Download HTML", "phenome_embedding.html", "text/html", html_bytes
            )
            self._selection_download_area.value = link
            self.status_label.value = status_html("HTML ready", "ok")
        except Exception as e:
            self._selection_download_area.value = ""
            self.status_label.value = status_html(f"HTML export failed: {type(e).__name__}", "err")

    # ------------------------------------------------------------------
    # Selection panel callbacks
    # ------------------------------------------------------------------
    def _selected_indices_as_csv_bytes(self) -> bytes:
        """CSV bytes for the selected rows of ``_cached_df``."""
        df = self._cached_df
        if df is None or not self._multi_selected_indices:
            return b""
        m = self._cached_index_to_row or {}
        rows = [m[i] for i in self._multi_selected_indices if i in m]
        if not rows:
            return b""
        sub = df.iloc[rows]
        buf = io.StringIO()
        sub.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL)
        return buf.getvalue().encode("utf-8")

    def _update_selection_summary(self) -> None:
        """Refresh the Selection accordion summary block."""
        n = len(self._multi_selected_indices)
        if n == 0:
            self._selection_summary.value = (
                '<div style="font-size:11px;color:#64748B;"><i>No points selected.</i></div>'
            )
            return
        preview = self._multi_selected_indices[: min(20, n)]
        extra = "" if n <= 20 else f" <i>(+{n - 20:,} more)</i>"
        ids_html = ", ".join(
            f'<code style="background:#EEF2F7;padding:1px 4px;border-radius:3px;">{i}</code>'
            for i in preview
        )
        self._selection_summary.value = (
            f'<div style="font-size:11px;color:#334155;line-height:1.5;">'
            f"<b>{n:,}</b> points selected: {ids_html}{extra}</div>"
        )

    def _on_selection_copy_clicked(self, _btn: Any) -> None:
        """Offer the list of selected indices as a plain-text download."""
        if not self._multi_selected_indices:
            self.status_label.value = status_html("No box/lasso selection", "warn")
            return
        text = ",".join(str(i) for i in self._multi_selected_indices).encode("utf-8")
        link = self._data_uri_html(
            "Download indices (.txt)",
            "phenome_selected_indices.txt",
            "text/plain",
            text,
        )
        self._selection_download_area.value = link
        self.status_label.value = status_html(
            f"{len(self._multi_selected_indices):,} indices ready", "ok"
        )

    def _on_selection_download_clicked(self, _btn: Any) -> None:
        """Save selected rows of the embedding DataFrame as a CSV download link."""
        if not self._multi_selected_indices or self._cached_df is None:
            self.status_label.value = status_html("No box/lasso selection", "warn")
            return
        payload = self._selected_indices_as_csv_bytes()
        if not payload:
            self.status_label.value = status_html("Nothing to export", "warn")
            return
        link = self._data_uri_html(
            "Download CSV",
            "phenome_selected_rows.csv",
            "text/csv",
            payload,
        )
        self._selection_download_area.value = link
        self.status_label.value = status_html(
            f"CSV ready ({len(self._multi_selected_indices):,} rows)", "ok"
        )

    def _clear_plotly_selection_visuals(self, *, reset_dragmode: bool = True) -> None:
        """Remove Plotly's selection outline and restore marker opacity.

        Uses batch_update to ensure traces and layout updates are sent together reliably.
        """
        fig = self.fig_widget
        if fig is None:
            return

        try:
            with fig.batch_update():
                # 1. Clear selection and unselected styling for ALL traces
                fig.update_traces(selectedpoints=None, unselected=None)

                # 2. Drop drawn box/lasso
                fig.layout.selections = ()

                # 3. Reset dragmode if requested
                if reset_dragmode:
                    fig.layout.dragmode = "pan"
        except Exception:
            logger.debug("Could not clear Plotly selection visuals", exc_info=True)

    def _clear_all_selections(self, *, reset_dragmode: bool = True) -> None:
        """Drop every kind of selection state and repaint the plot.

        Clears, in order:
          1. the box/lasso multi-selection (``_multi_selected_indices``),
          2. the single click-selected point (``_selected_point_index``),
          3. Plotly's visible selection outline + per-trace ``selectedpoints``,
          4. marker styling (``_apply_base_styling``) to remove residual dimming,
          5. ``dragmode`` back to ``pan``,
          6. any download link left in the Selection accordion footer.
        """
        had_multi = bool(self._multi_selected_indices)
        had_single = self._selected_point_index is not None

        self._multi_selected_indices = []
        self._selected_point_index = None
        self._selection_download_area.value = ""

        def _do() -> None:
            # 1. Repaint our halos to empty state.
            self._apply_multi_select_overlay()
            self._apply_selected_point_overlay()

            # 2. Restore data trace markers (uniform or highlight) via batch_update.
            if self.fig_widget is not None:
                try:
                    self._apply_base_styling()
                except Exception:
                    logger.debug("Could not re-apply base styling after clear", exc_info=True)

            # 3. Final atomic sync: clear Plotly's internal selection/dimming for ALL traces.
            # This uses batch_update to force the browser to exit selection mode.
            self._clear_plotly_selection_visuals(reset_dragmode=reset_dragmode)

            # Reset image side-panel, since the click selection is gone.
            if had_single:
                self._set_image_panel_idle()

            self._update_selection_summary()
            self._update_stats()

            parts = []
            if had_multi:
                parts.append("box/lasso")
            if had_single:
                parts.append("click")
            if not parts:
                parts.append("plot")
            self.status_label.value = status_html(f"Cleared {' + '.join(parts)} selection", "ok")

        schedule_after_plotly_event_loop(_do)

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

    Creates a :class:`PhenoMeInteractive` instance and displays it. Use in Jupyter
    notebooks to explore embeddings via PCA/t-SNE/UMAP with instant color switching,
    highlight mode, box/lasso multi-selection, and click-to-inspect image viewing.

    Args:
        pheno_me: Processed PhenoMe instance with embeddings
            and optional properties. Must have run ``process_images()`` first.
        filters: Optional metadata filters to restrict which images are shown.
            Dict mapping metadata keys to allowed values or lists of values.
            Example: ``{'condition': 'Control', 'time': ['24h', '48h']}``.
        exclude: Optional metadata exclusions (same structure as filters).
        hover_features: Optional list of metadata or property keys to show in
            hover tooltips. If ``None``, uses metadata keys from the pipeline.

    Returns:
        PhenoMeInteractive: The explorer instance. Call ``.show()`` again to re-display.

    Example:
        >>> from phenome import PhenoMe, load_dinov2_model
        >>> wrapper = load_dinov2_model()
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
