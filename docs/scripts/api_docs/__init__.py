"""Generate narrative API Markdown for PhenoMe from Python docstrings.

This package replaces the monolithic ``generate_api_docs.py``. The public
entry point is :func:`api_docs.site.build`, invoked from the thin
``generate_api_docs.py`` delegator so the ``npm run build``/``dev`` contract is
preserved.

Modules:
    - :mod:`api_docs.loader`: ``sys.path`` setup and module imports.
    - :mod:`api_docs.docstring`: Google/NumPy docstring -> Markdown parser.
    - :mod:`api_docs.signatures`: Callable signature rendering.
    - :mod:`api_docs.introspection`: Class/module member discovery.
    - :mod:`api_docs.sorting`: Section order and per-slug defaults.
    - :mod:`api_docs.rendering`: Method/class Markdown blocks.
    - :mod:`api_docs.pages`: Generic page renderers.
    - :mod:`api_docs.config`: ``PageSpec`` and ``PAGE_SPECS`` table.
    - :mod:`api_docs.site`: Iterates ``PAGE_SPECS`` and writes files.
"""

from __future__ import annotations

__all__ = ["build"]


def build() -> None:
    """Lazy import to avoid circulars during partial package builds."""
    from .site import build as _build

    _build()
