"""Repo-root ``sys.path`` setup so ``import phenome`` works without install."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve()
"""Absolute path to this module (``docs/scripts/api_docs/loader.py``)."""

_DOCS = _SCRIPT.parents[2]
"""Absolute path to ``docs/`` directory."""

_REPO_ROOT = _SCRIPT.parents[3]
"""Absolute path to repository root (contains ``phenome/``)."""

_OUT = _DOCS / "src" / "content" / "docs" / "advanced" / "api"
"""Directory where generated API Markdown files live."""


def ensure_repo_root_on_sys_path() -> Path | None:
    """Prepend repository root so ``import phenome`` works without editable install.

    Returns:
        The repo root path if ``phenome/`` is found there, else ``None``.
    """
    if not (_REPO_ROOT / "phenome").is_dir():
        return None
    root_s = str(_REPO_ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return _REPO_ROOT


def import_module(name: str) -> ModuleType:
    """Thin wrapper around :func:`importlib.import_module` for consistency."""
    return importlib.import_module(name)


def out_dir() -> Path:
    """Return the output directory, creating it if necessary."""
    _OUT.mkdir(parents=True, exist_ok=True)
    return _OUT


def docs_dir() -> Path:
    """Return the ``docs/`` directory."""
    return _DOCS


def repo_root() -> Path:
    """Return the repository root directory."""
    return _REPO_ROOT
