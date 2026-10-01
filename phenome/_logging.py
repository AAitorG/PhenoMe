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

Progress lines are the message only (``Processing pipeline:``), without the
stdlib ``INFO:phenome.pipeline:`` prefix.  That prefix is used when the
package logger is set to DEBUG.

Users who want to silence the output, or to include the prefix while
debugging, can do::

    import logging
    logging.getLogger("phenome").setLevel(logging.WARNING)  # silence
    logging.getLogger("phenome").setLevel(logging.DEBUG)    # prefix + debug
"""

import logging
import sys

# Package-level logger name
_PKG_LOGGER_NAME = "phenome"

# Sentinel attribute on the package logger so we configure at most once.
_CONFIGURED_ATTR = "_phenome_configured"

# ``logging.basicConfig()`` and several notebook kernels use this format.
# It is the noisy ``INFO:phenome.pipeline:`` prefix, not a user-chosen layout.
_DEFAULT_PREFIX_FORMAT = logging.BASIC_FORMAT


class _PhenomeFormatter(logging.Formatter):
    """Message-only formatter; add level and logger name while debugging."""

    def __init__(self) -> None:
        super().__init__("%(message)s")
        self._verbose = logging.Formatter(_DEFAULT_PREFIX_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as the message, or with the debug prefix."""
        if logging.getLogger(_PKG_LOGGER_NAME).getEffectiveLevel() <= logging.DEBUG:
            return self._verbose.format(record)
        return super().format(record)


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
    """Attach a message-only handler unless the user already configured logging."""
    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)

    if getattr(pkg_logger, _CONFIGURED_ATTR, False):
        return

    # Library best-practice: add NullHandler so that if the user never
    # configures logging at all, nothing breaks.
    pkg_logger.addHandler(logging.NullHandler())

    # A root handler with the stdlib default format (common in notebooks)
    # would print ``INFO:phenome.pipeline:``.  Own the stream in that case.
    # A handler whose format the user chose is left in charge.
    if not _ancestor_has_custom_format(pkg_logger):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_PhenomeFormatter())
        pkg_logger.addHandler(handler)
        pkg_logger.propagate = False

    # Set INFO as the default level for the package tree.
    # Only override if the logger is still at NOTSET (i.e. the user hasn't
    # explicitly set a level).
    if pkg_logger.level == logging.NOTSET:
        pkg_logger.setLevel(logging.INFO)

    setattr(pkg_logger, _CONFIGURED_ATTR, True)


def _ancestor_has_custom_format(logger: logging.Logger) -> bool:
    """Return True when a parent handler uses a format other than the stdlib default."""
    current: logging.Logger | None = logger.parent
    while current is not None:
        for handler in current.handlers:
            if isinstance(handler, logging.NullHandler):
                continue
            fmt = _handler_format(handler)
            if fmt is not None and fmt != _DEFAULT_PREFIX_FORMAT:
                return True
        if not current.propagate:
            break
        current = current.parent
    return False


def _handler_format(handler: logging.Handler) -> str | None:
    """Return the handler's format string, if it has one."""
    formatter = handler.formatter
    if formatter is None:
        return None
    fmt = getattr(formatter, "_fmt", None)
    return fmt if isinstance(fmt, str) else None
