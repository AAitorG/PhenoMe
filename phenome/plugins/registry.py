"""
Lightweight plugin registry for phenotyping extensions.

Uses simple dict-based storage. No entry points required.
"""

from collections.abc import Callable
from typing import Any

# Type for property functions: (image2d, mask2d) -> Dict[str, float]
PropertyFunction = Callable[[Any | None, Any | None], dict[str, float | int]]


# ---------------------------------------------------------------------------
# Property function registry
# ---------------------------------------------------------------------------

_PROPERTY_FUNCTIONS: dict[str, PropertyFunction] = {}


def register_property(name: str, fn: PropertyFunction) -> None:
    """Register a custom property function.

    The function should have signature (image2d, mask2d) -> Dict[str, float].
    Use with compute_properties via get_property(name) or pass the function directly.

    Args:
        name: Property identifier (e.g. "blob", "my_custom").
        fn: Callable(image2d, mask2d) -> dict of property name -> value.
    """
    _PROPERTY_FUNCTIONS[name] = fn


def get_property(name: str) -> PropertyFunction | None:
    """Get a registered property function by name."""
    return _PROPERTY_FUNCTIONS.get(name)


def list_properties() -> list[str]:
    """Return names of all registered property functions."""
    return list(_PROPERTY_FUNCTIONS.keys())


# ---------------------------------------------------------------------------
# Metadata extractor registry
# ---------------------------------------------------------------------------

_METADATA_EXTRACTORS: dict[str, Callable[[str], dict[str, Any]]] = {}


def register_metadata_extractor(name: str, fn: Callable[[str], dict[str, Any]]) -> None:
    """Register a custom metadata extractor.

    The function should accept a file path and return a dict with at least
    'file_path' (full path). Other keys become metadata columns.

    Args:
        name: Extractor identifier.
        fn: Callable(path) -> metadata dict.
    """
    _METADATA_EXTRACTORS[name] = fn


def get_metadata_extractor(
    name: str,
) -> Callable[[str], dict[str, Any]] | None:
    """Get a registered metadata extractor by name."""
    return _METADATA_EXTRACTORS.get(name)


# ---------------------------------------------------------------------------
# Report section registry
# ---------------------------------------------------------------------------

_REPORT_SECTIONS: dict[str, Callable[..., str]] = {}


def register_report_section(name: str, fn: Callable[..., str]) -> None:
    """Register a custom report section generator.

    The function will be called during report generation. Signature should
    accept (pipeline, ...) and return HTML string for the section.

    Args:
        name: Section identifier (used for nav and ordering).
        fn: Callable that generates the section HTML.
    """
    _REPORT_SECTIONS[name] = fn


def get_report_sections() -> dict[str, Callable[..., str]]:
    """Return all registered report section generators."""
    return dict(_REPORT_SECTIONS)
