"""Callable signature rendering for API docs."""

from __future__ import annotations

import inspect
from typing import Any


def _format_return_annotation(ret: Any) -> str:
    """Render a return annotation as a compact string."""
    if ret is inspect.Signature.empty:
        return ""
    if hasattr(ret, "__name__"):
        return ret.__name__
    return str(ret).replace("typing.", "")


def callable_sig_str(obj: Any) -> str:
    """Format a callable signature as a multi-line Python-like string.

    Returns ``""`` if the signature cannot be introspected.

    Args:
        obj: Callable (function, method, class).

    Returns:
        Multi-line signature string starting with ``(`` and ending with
        ``) -> ReturnType`` when a return annotation is present.
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""

    params = list(sig.parameters.values())
    base = "()" if not params else "(\n    " + ",\n    ".join(str(p) for p in params) + "\n)"

    ret = _format_return_annotation(sig.return_annotation)
    if ret:
        return f"{base} -> {ret}"
    return base


def class_init_sig_str(cls: type) -> str:
    """Return the ``__init__`` signature formatted as ``ClassName(...)``.

    The ``self`` parameter is stripped. Returns ``""`` if no signature is
    available (e.g. classes inheriting from ``object`` without overriding).
    """
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return ""

    params = list(sig.parameters.values())
    base = "()" if not params else "(\n    " + ",\n    ".join(str(p) for p in params) + "\n)"
    return base
