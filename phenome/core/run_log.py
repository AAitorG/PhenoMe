"""Silent JSON-safe capture of pipeline method calls for experiment export.

Recording is in-memory only: no log lines, no files. Output happens when the
caller invokes ``export_experiment_config`` or ``export_methods_markdown``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_MAX_DEPTH = 8
_MAX_SEQUENCE = 64


def json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Return a JSON-serializable copy of *value*.

    Callables become their ``__name__``. NumPy scalars become Python scalars.
    Large arrays become a ``shape`` stub. Objects that cannot be encoded become
    ``repr(value)``.
    """
    if _depth > _MAX_DEPTH:
        return repr(value)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    type_name = type(value).__name__
    module = getattr(type(value), "__module__", "")
    if type_name == "device" and "torch" in module:
        return str(value)
    if callable(value) and not isinstance(value, type):
        return getattr(value, "__name__", type_name)
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        items = list(value)
        if len(items) > _MAX_SEQUENCE:
            return [json_safe(v, _depth=_depth + 1) for v in items[:_MAX_SEQUENCE]] + [
                f"... ({len(items) - _MAX_SEQUENCE} more)"
            ]
        return [json_safe(v, _depth=_depth + 1) for v in items]
    if isinstance(value, set | frozenset):
        return sorted(json_safe(v, _depth=_depth + 1) for v in value)
    if hasattr(value, "tolist") and hasattr(value, "dtype"):
        try:
            ndim = int(getattr(value, "ndim", 1))
            if ndim == 0:
                return json_safe(value.item(), _depth=_depth + 1)
            size = int(getattr(value, "size", 0))
            if size <= _MAX_SEQUENCE:
                return json_safe(value.tolist(), _depth=_depth + 1)
            return {"type": "ndarray", "shape": list(value.shape)}
        except (TypeError, ValueError, AttributeError):
            return repr(value)
    if type_name == "DataFrame":
        try:
            columns = [str(c) for c in value.columns]
            return {"type": "DataFrame", "n_rows": len(value), "columns": columns}
        except (TypeError, AttributeError):
            return {"type": "DataFrame"}
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def snapshot_environment(device: Any = None) -> dict[str, Any]:
    """Collect Python, package, and device versions (no I/O, no logging)."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        from phenome import __version__ as pkg_version
    except ImportError:
        pkg_version = "unknown"
    env["phenome"] = pkg_version
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    if device is not None:
        env["device"] = str(device)
    return env


def record_step(
    obj: Any,
    method: str,
    kwargs: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Record *method* on *obj* when it implements ``_record_step``.

    Mixins call this so recording is a no-op on objects without a run log.
    """
    fn = getattr(obj, "_record_step", None)
    if callable(fn):
        fn(method, kwargs, extras)


class RunLog:
    """In-memory record of pipeline init, environment, and method steps."""

    def __init__(self, init_kwargs: dict[str, Any], environment: dict[str, Any]) -> None:
        self.created_at: str = datetime.now().isoformat()
        self.init: dict[str, Any] = json_safe(init_kwargs)
        self.environment: dict[str, Any] = dict(environment)
        self.steps: list[dict[str, Any]] = []

    def record(
        self,
        method: str,
        kwargs: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Append one silent step. Does not log or write files."""
        step: dict[str, Any] = {
            "method": method,
            "at": datetime.now().isoformat(),
            "kwargs": json_safe(kwargs or {}),
        }
        if extras:
            step["extras"] = json_safe(extras)
        self.steps.append(step)

    def clear_steps(self) -> None:
        """Drop recorded steps. Keep init and environment (used by ``reset``)."""
        self.steps.clear()

    def latest_model(self) -> dict[str, Any] | None:
        """Return the most recent ``extras['model']`` from a processing step."""
        for step in reversed(self.steps):
            extras = step.get("extras")
            if isinstance(extras, dict) and extras.get("model"):
                model = extras["model"]
                return model if isinstance(model, dict) else {"name": model}
        return None

    def latest_reference_filters(self) -> dict[str, Any] | None:
        """Return ``reference_filters`` from the last distance step, if any."""
        for step in reversed(self.steps):
            kwargs = step.get("kwargs")
            if isinstance(kwargs, dict) and kwargs.get("reference_filters") is not None:
                filters = kwargs["reference_filters"]
                return filters if isinstance(filters, dict) else None
        return None

    def to_dict(
        self,
        *,
        processing_params: dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the experiment-config dict (backward-compatible keys plus log)."""
        config: dict[str, Any] = {
            "seed": self.init.get("seed"),
            "use_gpu_for_dr": self.init.get("use_gpu_for_dr"),
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": self.environment.get("phenome", "unknown"),
            "environment": dict(self.environment),
            "init": dict(self.init),
            "steps": [dict(s) for s in self.steps],
        }
        model = self.latest_model()
        if model:
            config["model"] = model
            name = model.get("name") or model.get("model_name")
            if name:
                config["model_name"] = name
        ref = self.latest_reference_filters()
        if ref is not None:
            config["reference_filters"] = ref
        if processing_params is not None:
            config["processing_params"] = json_safe(processing_params)
        if checkpoint_path is not None:
            config["checkpoint_path"] = checkpoint_path
        if extra:
            config.update(json_safe(extra))
        return config

    def to_markdown(self) -> str:
        """Render a methods-section markdown document from the log."""
        lines = [
            "# PhenoMe run settings",
            "",
            f"Recorded at `{datetime.now().isoformat()}`.",
            "",
            "## Environment",
            "",
        ]
        for key, value in self.environment.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.extend(["", "## Init", ""])
        for key, value in self.init.items():
            lines.append(f"- **{key}**: `{value}`")
        lines.extend(["", "## Steps", ""])
        if not self.steps:
            lines.append("No pipeline methods recorded yet.")
        else:
            for i, step in enumerate(self.steps, start=1):
                kwargs = json.dumps(step.get("kwargs") or {}, default=str, sort_keys=True)
                lines.append(f"{i}. `{step.get('method')}` ({step.get('at')})")
                lines.append(f"   - kwargs: `{kwargs}`")
                extras = step.get("extras")
                if extras:
                    lines.append(
                        f"   - extras: `{json.dumps(extras, default=str, sort_keys=True)}`"
                    )
        lines.append("")
        return "\n".join(lines)
