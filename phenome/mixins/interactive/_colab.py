"""Colab adaptations for the interactive explorer.

``colab=None`` detects the runtime. ``True`` or ``False`` forces the layout.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
from IPython.display import Javascript, clear_output, display

from ..._logging import get_logger

logger = get_logger(__name__)

# Colab syncs a widget only while it is handling a frontend request. Trait
# changes made from a compute thread never reach the page (colabtools#3373).
# A browser poll invokes this callback, and that invocation is the request.
COMPUTE_PUMP_CALLBACK = "phenome.explorerPump"
PUMP_RUN = "phenome-pump-run"
PUMP_STOP = "phenome-pump-stop"
_PUMP_INTERVAL_MS = 500


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
    direct child. The finished figure is mounted later by the compute pump,
    which is the request Colab will sync.
    """
    plot_slot.children = (placeholder,)
    if colab:
        return
    with output_area:
        clear_output(wait=True)
        display(plot_slot)


def push_widget_state(*widgets_to_push: Any) -> None:
    """Send current widget state to the frontend.

    Used from the Colab compute poll. Assigning a trait from that poll is not
    always enough; ``send_state`` is what actually reaches the page.
    """
    for widget in widgets_to_push:
        send_state = getattr(widget, "send_state", None)
        if send_state is None:
            continue
        with contextlib.suppress(Exception):
            send_state()


def _pump_script(generation: int) -> str:
    """JavaScript that polls ``COMPUTE_PUMP_CALLBACK`` until this generation stops."""
    gen = int(generation)
    return f"""
(function() {{
  var gen = {gen};
  window._phenomePumpGen = gen;
  window._phenomePumpEpoch = (window._phenomePumpEpoch || 0) + 1;
  var epoch = window._phenomePumpEpoch;
  window._phenomePumpFails = 0;
  if (window._phenomePumpTimer) {{
    clearTimeout(window._phenomePumpTimer);
    window._phenomePumpTimer = null;
  }}
  var kernel = (typeof google !== "undefined" && google.colab && google.colab.kernel)
    || (typeof colab !== "undefined" && colab.kernel);
  if (!kernel || !kernel.invokeFunction) return;
  function stillCurrent() {{
    return window._phenomePumpGen === gen && window._phenomePumpEpoch === epoch;
  }}
  function isStop(result) {{
    try {{
      if (result === "{PUMP_STOP}") return true;
      var plain = result && result.data && result.data["text/plain"];
      if (plain && String(plain).indexOf("{PUMP_STOP}") !== -1) return true;
      var str = JSON.stringify(result);
      if (str && str.indexOf("{PUMP_STOP}") !== -1) return true;
    }} catch (e) {{}}
    return false;
  }}
  function schedule() {{
    if (!stillCurrent()) return;
    window._phenomePumpTimer = setTimeout(tick, {_PUMP_INTERVAL_MS});
  }}
  function tick() {{
    if (!stillCurrent()) return;
    Promise.resolve(kernel.invokeFunction("{COMPUTE_PUMP_CALLBACK}", [], {{}})).then(function(result) {{
      if (!stillCurrent()) return;
      window._phenomePumpFails = 0;
      if (isStop(result)) return;
      schedule();
    }}).catch(function() {{
      if (!stillCurrent()) return;
      window._phenomePumpFails = (window._phenomePumpFails || 0) + 1;
      if (window._phenomePumpFails > 8) return;
      schedule();
    }});
  }}
  tick();
}})();
"""


def launch_compute_pump(on_tick: Callable[[], str], generation: int) -> bool:
    """Start a Colab poll that calls ``on_tick`` on the kernel request thread.

    Returns False when this process is not Colab. ``generation`` lets a newer
    compute retire the previous poll without a late stop killing the new one.
    ``on_tick`` must return ``PUMP_RUN`` or ``PUMP_STOP``.
    """
    try:
        from google.colab import output
    except ImportError:
        return False

    def _invoke(*_args: Any, **_kwargs: Any) -> str:
        try:
            token = on_tick()
        except Exception:
            logger.warning("Colab compute pump failed", exc_info=True)
            return PUMP_STOP
        if token == PUMP_STOP:
            return PUMP_STOP
        return PUMP_RUN

    try:
        output.register_callback(COMPUTE_PUMP_CALLBACK, _invoke)
    except Exception:
        logger.warning("Could not register the Colab compute pump", exc_info=True)
        return False

    script = _pump_script(generation)
    # A widget callback does not always get a cell eval. Start the poll both in
    # the cell (``eval_js``, same window as ``request_pump_stop``) and in an
    # output frame (``display``). Each copy stops when the tick returns
    # ``PUMP_STOP``; a newer generation ignores a late stop from the older one.
    started = False
    try:
        output.eval_js(script, ignore_result=True)
        started = True
    except Exception:
        logger.warning("Could not eval the Colab compute pump", exc_info=True)
    try:
        display(Javascript(script))
        started = True
    except Exception:
        logger.warning("Could not display the Colab compute pump", exc_info=True)
    return started


def request_pump_stop(generation: int) -> None:
    """Ask the browser to stop the poll started for ``generation``."""
    gen = int(generation)
    try:
        from google.colab import output
    except ImportError:
        return
    with contextlib.suppress(Exception):
        output.eval_js(
            f"if (window._phenomePumpGen === {gen}) window._phenomePumpGen = -1;",
            ignore_result=True,
        )
