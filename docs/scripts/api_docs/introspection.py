"""Class and module member discovery for API docs."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Any


def iter_public_names(cls: type) -> Iterable[str]:
    """Yield public (non-underscore) names from ``dir(cls)``."""
    for name in dir(cls):
        if name.startswith("_"):
            continue
        yield name


def defining_class(owner: type, name: str) -> type | None:
    """Return the class in ``owner.__mro__`` that defines ``name``, or None."""
    for cls in owner.__mro__:
        if name in cls.__dict__:
            return cls
    return None


def collect_class_members(
    cls: type,
    *,
    exclude_defining: Sequence[type] | None = None,
    exclude_names: frozenset[str] | None = None,
    include_properties: bool = True,
) -> list[tuple[str, Any, type]]:
    """List public routines/properties as ``(name, obj, defining_cls)``.

    Args:
        cls: Class to introspect.
        exclude_defining: Skip members whose defining class matches one of
            these (useful to hide inherited methods).
        exclude_names: Names to skip.
        include_properties: Include ``property`` descriptors.

    Returns:
        Alphabetically-sorted list of unique ``(name, obj, defining_cls)``
        tuples.
    """
    exclude_defining = exclude_defining or ()
    exclude_defining_names = {c.__name__ for c in exclude_defining}
    skip_names = exclude_names or frozenset()

    out: list[tuple[str, Any, type]] = []
    seen: set[str] = set()
    for name in sorted(iter_public_names(cls), key=str.lower):
        if name in seen or name in skip_names:
            continue
        try:
            obj = getattr(cls, name)
        except AttributeError:
            continue
        defining = defining_class(cls, name)
        if defining is None:
            continue
        if defining.__name__ in exclude_defining_names:
            continue
        if inspect.isroutine(obj) or (include_properties and isinstance(obj, property)):
            out.append((name, obj, defining))
            seen.add(name)
    return out


def iter_module_entries(
    module: Any,
    names: Sequence[str] | None = None,
) -> list[tuple[str, Any, str]]:
    """Yield ``(name, obj, kind)`` for functions/classes in a module.

    Uses ``__all__`` when ``names`` is ``None`` and ``__all__`` is defined,
    otherwise enumerates objects defined in the module.
    """
    raw_all = getattr(module, "__all__", None)
    if names is not None:
        use_names = list(names)
    elif raw_all:
        use_names = sorted(raw_all, key=str.lower)
    else:
        use_names = []
        for n, o in inspect.getmembers(module):
            if n.startswith("_") or inspect.ismodule(o):
                continue
            if getattr(o, "__module__", None) != module.__name__:
                continue
            use_names.append(n)
        use_names = sorted(use_names, key=str.lower)

    entries: list[tuple[str, Any, str]] = []
    for n in use_names:
        try:
            o = getattr(module, n)
        except AttributeError:
            continue
        if inspect.isclass(o):
            entries.append((n, o, "class"))
        elif inspect.isroutine(o):
            entries.append((n, o, "func"))
    return entries
