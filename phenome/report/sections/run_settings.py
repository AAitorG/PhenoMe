"""Optional Methods / settings section for the HTML report."""

from __future__ import annotations

import json
from typing import Any

from .._components import generate_info_box, generate_table


def generate_run_settings_section(settings: dict[str, Any] | None) -> str:
    """Render environment, init, and recorded steps as HTML.

    Args:
        settings: Dict from ``PhenoMe.get_run_settings()``. If empty, a short
            notice is shown.

    Returns:
        HTML string for the Methods / settings section.
    """
    if not settings:
        return generate_info_box(
            "No run settings were recorded on this pipeline instance.",
            "info",
        )

    env = settings.get("environment") or {}
    init = settings.get("init") or {}
    steps = settings.get("steps") or []

    env_rows = [[str(k), str(v)] for k, v in env.items()]
    init_rows = [[str(k), str(v)] for k, v in init.items()]
    step_rows: list[list[str]] = []
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        kwargs = json.dumps(step.get("kwargs") or {}, default=str, sort_keys=True)
        extras = step.get("extras")
        extras_s = json.dumps(extras, default=str, sort_keys=True) if extras else ""
        step_rows.append(
            [
                str(i),
                str(step.get("method", "")),
                str(step.get("at", "")),
                kwargs,
                extras_s,
            ]
        )

    parts = [
        generate_info_box(
            "Opt-in dump of parameters recorded during this session. "
            "Defaults and plot formats are unchanged unless you passed them explicitly.",
            "info",
        )
    ]
    if env_rows:
        parts.append("<h4>Environment</h4>")
        parts.append(generate_table(["Key", "Value"], env_rows, sortable=True))
    if init_rows:
        parts.append("<h4>Init</h4>")
        parts.append(generate_table(["Parameter", "Value"], init_rows, sortable=True))
    parts.append("<h4>Steps</h4>")
    if step_rows:
        parts.append(
            generate_table(
                ["#", "Method", "At", "Kwargs", "Extras"],
                step_rows,
                sortable=True,
                numeric_columns=[0],
            )
        )
    else:
        parts.append("<p>No pipeline methods recorded yet.</p>")
    return "\n".join(parts)
