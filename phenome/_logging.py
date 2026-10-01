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
    """Attach a message-only stderr handler when no user sink should keep the records.

    A handler already on the phenome logger is left alone. Ancestor stdout
    or stderr handlers that use the stdlib default format are replaced, so
    notebooks do not print the ``INFO:phenome:`` prefix. A file handler, or
    any format the user chose, keeps propagation and is not duplicated.
    """
    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)

    if getattr(pkg_logger, _CONFIGURED_ATTR, False):
        return

    # Library best-practice: add NullHandler so that if the user never
    # configures logging at all, nothing breaks.
    pkg_logger.addHandler(logging.NullHandler())

    # A root handler with the stdlib default format (common in notebooks)
    # would print ``INFO:phenome.pipeline:``.  Own stderr in that case.
    # A file, a handler already on this logger, or a format the user chose
    # stays in charge.
    if not _package_has_user_handler(pkg_logger) and _should_own_stream(pkg_logger):
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


def _package_has_user_handler(logger: logging.Logger) -> bool:
    """Return True when *logger* already has a handler other than NullHandler."""
    return any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def _should_own_stream(logger: logging.Logger) -> bool:
    """Return True when phenome can replace ancestor handlers with its own stderr stream.

    Own the stream when there is no ancestor handler, or every ancestor
    handler is a stdout/stderr ``StreamHandler`` with the stdlib default
    format or no formatter. Any other sink keeps receiving records.
    """
    current: logging.Logger | None = logger.parent
    while current is not None:
        for handler in current.handlers:
            if isinstance(handler, logging.NullHandler):
                continue
            if not _is_default_stdio_handler(handler):
                return False
        if not current.propagate:
            break
        current = current.parent
    return True


def _is_default_stdio_handler(handler: logging.Handler) -> bool:
    """Return True for a stdout/stderr handler using the stdlib default format."""
    if isinstance(handler, logging.FileHandler):
        return False
    if not isinstance(handler, logging.StreamHandler):
        return False
    if getattr(handler, "stream", None) not in (sys.stdout, sys.stderr):
        return False
    formatter = handler.formatter
    if formatter is None:
        return True
    fmt = getattr(formatter, "_fmt", None)
    return isinstance(fmt, str) and fmt == _DEFAULT_PREFIX_FORMAT
