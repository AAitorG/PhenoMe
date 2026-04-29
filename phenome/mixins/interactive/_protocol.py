"""Protocol describing the host object the interactive explorer needs.

Kept in its own module so ``_InteractiveExplorerProtocol`` remains importable
as ``phenome.mixins.interactive._InteractiveExplorerProtocol`` (preserves the
public surface that lives in the visualization mixin and the docs).
"""

from __future__ import annotations

from typing import Any, Protocol


class _InteractiveExplorerProtocol(Protocol):
    """Protocol for objects that can be explored interactively (pipeline or visualization mixin)."""

    results: Any
    device: Any

    def get_available_property_keys(self) -> list[str]:
        """Return available scalar property column names."""
        ...

    def _get_color_column(self, color_by: str, df: Any) -> tuple[str, bool]:
        """Resolve ``color_by`` to a DataFrame column and whether it is continuous."""
        ...

    def plot_image_by_index(
        self,
        idx: int,
        distance_results: dict | None = None,
        channels: Any | None = None,
        figsize: tuple[float, float] = (6.0, 6.0),
        title_fields: list[str] | None = None,
        show_extra_info: bool = True,
        apply_transforms: bool = True,
        downsample: int | None = None,
        ax: Any | None = None,
        return_fig: bool = False,
    ) -> Any:
        """Display the image for result index ``idx`` in the notebook."""
        ...

    def image_preview_png_bytes(
        self,
        idx: int,
        distance_results: dict | None = None,
        channels: Any | None = None,
        title_fields: list[str] | None = None,
        show_extra_info: bool = False,
        apply_transforms: bool = True,
        downsample: int | None = None,
    ) -> tuple[bytes, str | None, str | None]:
        """PNG bytes and optional details text (same as ``plot_image_by_index`` extra block)."""
        ...
