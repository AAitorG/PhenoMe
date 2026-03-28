"""
Notebook-safe logging configuration for phenome.

Problem
-------
The standard ``logging.getLogger(__name__)`` pattern silently drops INFO
messages in Jupyter notebooks because Python's root logger defaults to
WARNING level and has no handler attached.  If the user manually adds a
handler, duplicate output appears on every cell re-run.

Solution
--------
We configure a single package-level logger (``phenome``) with:

* A ``StreamHandler`` writing to ``sys.stderr`` (rendered by Jupyter as
  plain text below the cell).
* INFO as default level, so progress messages are always visible.
* A guard that prevents adding duplicate handlers on module reload
  (``%autoreload`` / repeated imports).

All sub-modules should call::

    from ._logging import get_logger
    logger = get_logger(__name__)

instead of ``logging.getLogger(__name__)``.

Users who want to silence the output can do::

    import logging
    logging.getLogger("phenome").setLevel(logging.WARNING)
"""

import logging
import sys

# Package-level logger name
_PKG_LOGGER_NAME = "phenome"

# Sentinel attribute on the package logger so we configure at most once.
_CONFIGURED_ATTR = "_phenome_configured"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for notebook-safe output.

    Typical usage at the top of each module::

        from ._logging import get_logger
        logger = get_logger(__name__)

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.
    """
    _ensure_configured()
    return logging.getLogger(name)


def _ensure_configured() -> None:
    """Attach a StreamHandler to the package logger if not done already."""
    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)

    if getattr(pkg_logger, _CONFIGURED_ATTR, False):
        return

    # Library best-practice: add NullHandler so that if the user never
    # configures logging at all, nothing breaks.
    pkg_logger.addHandler(logging.NullHandler())

    # Only add our StreamHandler if the package logger (or its parents)
    # have no *effective* handlers yet.  This avoids fighting with a
    # user who already configured logging before importing us.
    if not _has_effective_handler(pkg_logger):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)

    # Set INFO as the default level for the package tree.
    # Only override if the logger is still at NOTSET (i.e. the user hasn't
    # explicitly set a level).
    if pkg_logger.level == logging.NOTSET:
        pkg_logger.setLevel(logging.INFO)

    setattr(pkg_logger, _CONFIGURED_ATTR, True)


def _has_effective_handler(logger: logging.Logger) -> bool:
    """Check if *logger* or any ancestor already has a non-NullHandler."""
    current: logging.Logger | None = logger
    while current:
        for h in current.handlers:
            if not isinstance(h, logging.NullHandler):
                return True
        if not current.propagate:
            break
        current = current.parent
    return False
