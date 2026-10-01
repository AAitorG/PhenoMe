"""Colab adaptations for the interactive explorer.

``colab=None`` detects the runtime. ``True`` or ``False`` forces the layout.
"""

from __future__ import annotations

import contextlib
from typing import Any

import ipywidgets as widgets
from IPython.display import clear_output, display

from ..._logging import get_logger

logger = get_logger(__name__)


def running_in_colab() -> bool:
    """Return True when this kernel is Google Colab."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_colab(colab: bool | None) -> bool:
    """Use an explicit ``colab`` flag, or detect the runtime when it is ``None``."""
    if colab is None:
        return running_in_colab()
    return colab


def install_layout_compat() -> None:
    """Ignore layout kwargs that ipywidgets 7 does not define.

    Colab ships ipywidgets 7.7.1. ``Layout`` there has no ``gap``,
    ``background``, or ``border_radius``, and passing them raises
    ``TraitError`` while the explorer is built.
    """
    base = widgets.Layout
    if getattr(base, "_phenome_layout_compat", False):
        return
    known = set(base.class_traits())
    if {"gap", "background", "border_radius"} <= known:
        return

    class _SafeLayout(base):  # type: ignore[valid-type]
        _phenome_layout_compat = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{key: value for key, value in kwargs.items() if key in known})

    widgets.Layout = _SafeLayout


def button_style(**kwargs: Any) -> Any:
    """Build a ``ButtonStyle``, dropping traits missing on ipywidgets 7.

    ``text_color`` exists only on ipywidgets 8. Colab rejects it and the
    explorer never appears.
    """
    known = set(widgets.ButtonStyle.class_traits())
    return widgets.ButtonStyle(**{key: value for key, value in kwargs.items() if key in known})


def prepare_dashboard_cell(previous: widgets.Widget | None, *, colab: bool) -> None:
    """Clear the cell output, or hide the previous dashboard on Colab.

    Jupyter removes the previous widget tree with ``clear_output``. Colab
    applies that clear asynchronously and erases the widget displayed next,
    so the previous dashboard is hidden instead, after the widget manager
    is enabled.
    """
    if not colab:
        clear_output(wait=True)
        return
    prepare_colab_widgets()
    if previous is None:
        return
    previous.layout.display = "none"
    with contextlib.suppress(Exception):
        previous.close()


def prepare_colab_widgets() -> None:
    """Enable the widget manager Colab needs for Plotly FigureWidget.

    FigureWidget is a third-party widget. Colab leaves it blank unless this
    runs before the dashboard is displayed. ``no_vertical_scroll`` lifts the
    cell's height cap so the controls are not clipped.
    """
    try:
        from google.colab import output

        output.enable_custom_widget_manager()
        output.no_vertical_scroll()
    except Exception:
        logger.warning("Colab widget setup failed", exc_info=True)


def section_group(
    sections: list[tuple[str, widgets.Widget]],
    *,
    open_index: int | None,
    colab: bool,
) -> widgets.Widget:
    """Group titled panels.

    Jupyter uses ``Accordion``. Colab's custom widget manager, required for
    FigureWidget, throws while rendering ``Accordion`` and ``Tab`` because
    the frontend reads a ``titles`` key that ipywidgets 7 does not sync.
    ``colab=True`` uses a button per section instead.
    """
    if not colab:
        accordion = widgets.Accordion(
            children=[child for _, child in sections],
            layout=widgets.Layout(width="100%"),
        )
        for index, (title, _) in enumerate(sections):
            accordion.set_title(index, title)
        accordion.selected_index = open_index
        return accordion

    headers: list[widgets.Button] = []
    bodies: list[widgets.Widget] = []

    def _apply(opened: int | None) -> None:
        for index, (header, body) in enumerate(zip(headers, bodies, strict=True)):
            is_open = opened is not None and index == opened
            body.layout.display = "flex" if is_open else "none"
            header.button_style = "info" if is_open else ""

    rows: list[widgets.Widget] = []
    for index, (title, body) in enumerate(sections):
        header = widgets.Button(
            description=title,
            tooltip=title,
            layout=widgets.Layout(width="100%"),
        )
        headers.append(header)
        bodies.append(body)

        def _on_click(_btn: Any, idx: int = index) -> None:
            current = next(
                (i for i, panel in enumerate(bodies) if panel.layout.display != "none"),
                None,
            )
            _apply(None if current == idx else idx)

        header.on_click(_on_click)
        rows.append(widgets.VBox([header, body], layout=widgets.Layout(width="100%")))

    _apply(open_index)
    return widgets.VBox(rows, layout=widgets.Layout(width="100%"))


def output_area_layout(*, colab: bool) -> widgets.Layout:
    """Layout for the traceback output.

    In Jupyter the output holds the plot and must be allowed to shrink.
    In Colab the plot is a sibling, and this widget is only the traceback.
    """
    if colab:
        return widgets.Layout(width="100%", overflow_y="auto")
    return widgets.Layout(flex="1 1 0%", min_width="0px", width="auto")


def plot_column_children(
    plot_slot: widgets.Widget,
    output_area: widgets.Widget,
    *,
    colab: bool,
) -> list[widgets.Widget]:
    """Children of the plot column.

    Colab does not draw a FigureWidget nested inside an ``Output`` widget,
    so the plot slot is a direct child there.
    """
    if colab:
        return [plot_slot, output_area]
    return [output_area]


def mount_plot(
    plot_slot: widgets.VBox,
    output_area: widgets.Output,
    placeholder: widgets.Widget,
    *,
    colab: bool,
) -> None:
    """Show the idle plot placeholder.

    Jupyter displays the slot inside ``output_area``. Colab assigns it as a
    direct child; a later compute only replaces ``plot_slot.children``.
    """
    plot_slot.children = (placeholder,)
    if colab:
        return
    with output_area:
        clear_output(wait=True)
        display(plot_slot)
