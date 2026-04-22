"""
Utility functions for the phenotyping report.

This module provides helper functions for:
- Plotly figure styling (dark theme)
- Plotly HTML fragment rendering (offline-safe, no per-plot CDN script)
- Image encoding to base64
- HTML escaping for safe insertion of user content
"""

import base64
from functools import lru_cache
from html import escape as html_escape
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from PIL import Image


# Dark theme colors for Plotly plots
PLOT_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(255,255,255,0.05)",
    "font_color": "#f1f5f9",
    "title_font_color": "#f1f5f9",
    "legend_font_color": "#94a3b8",
    "gridcolor": "#334155",
    "zerolinecolor": "#475569",
}

# Color palette for categorical data
CATEGORY_COLORS = [
    "#6366f1",  # Primary (indigo)
    "#10b981",  # Secondary (emerald)
    "#f59e0b",  # Accent (amber)
    "#ef4444",  # Danger (red)
    "#8b5cf6",  # Purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#84cc16",  # Lime
    "#f97316",  # Orange
    "#14b8a6",  # Teal
]


def apply_dark_theme(fig: "go.Figure") -> "go.Figure":
    """
    Apply dark theme styling to a Plotly figure.

    Args:
        fig: Plotly figure to style

    Returns:
        Styled figure
    """
    fig.update_layout(
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font_color=PLOT_THEME["font_color"],
        title_font_color=PLOT_THEME["title_font_color"],
        legend_font_color=PLOT_THEME["legend_font_color"],
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    try:
        fig.update_xaxes(
            gridcolor=PLOT_THEME["gridcolor"], zerolinecolor=PLOT_THEME["zerolinecolor"]
        )
        fig.update_yaxes(
            gridcolor=PLOT_THEME["gridcolor"], zerolinecolor=PLOT_THEME["zerolinecolor"]
        )
    except (AttributeError, KeyError):
        pass  # Some figure types don't support axis updates

    return fig


def image_to_base64(
    image: Union[np.ndarray, "Image.Image"], size: tuple | None = None, format: str = "PNG"
) -> str:
    """
    Convert an image to base64 encoded string.

    Args:
        image: NumPy array (H, W, C) or PIL Image
        size: Optional (width, height) to resize to
        format: Output format ('PNG' or 'JPEG')

    Returns:
        Base64 encoded string
    """
    from PIL import Image as PILImage

    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        # Ensure proper dtype and range
        if image.dtype == np.float64 or image.dtype == np.float32:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Handle different channel configurations
        if image.ndim == 2:
            # Grayscale
            pil_image = PILImage.fromarray(image, mode="L")
        elif image.shape[2] == 1:
            # Single channel
            pil_image = PILImage.fromarray(image[:, :, 0], mode="L")
        elif image.shape[2] == 3:
            # RGB
            pil_image = PILImage.fromarray(image, mode="RGB")
        elif image.shape[2] == 4:
            # RGBA
            pil_image = PILImage.fromarray(image, mode="RGBA")
        else:
            # Multi-channel: take first 3 or convert to grayscale
            if image.shape[2] >= 3:
                pil_image = PILImage.fromarray(image[:, :, :3], mode="RGB")
            else:
                pil_image = PILImage.fromarray(image[:, :, 0], mode="L")
    else:
        pil_image = image

    # Downscale to fit inside box while preserving aspect ratio (no upscaling).
    if size is not None:
        pil_image = pil_image.copy()
        pil_image.thumbnail(size, PILImage.Resampling.LANCZOS)

    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format=format, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def plotly_to_html_fragment(
    fig: "go.Figure",
    config: dict[str, Any] | None = None,
    div_id: str | None = None,
) -> str:
    """Render a Plotly figure as an HTML fragment without inlining the Plotly library.

    All reports embed the Plotly library exactly once (see
    :func:`get_plotly_bundle`), so every individual plot uses
    ``include_plotlyjs=False`` to avoid duplicating ~5 MB of JS per chart.

    Args:
        fig: Plotly figure to render.
        config: Optional Plotly ``config`` dict (e.g. PLOTLY_DISPLAY_CONFIG).
        div_id: Optional explicit DOM id for the plot container.

    Returns:
        HTML fragment containing the chart div and its bootstrap script.
    """
    import plotly.io as pio

    kwargs: dict[str, Any] = {
        "full_html": False,
        "include_plotlyjs": False,
    }
    if config is not None:
        kwargs["config"] = config
    if div_id is not None:
        kwargs["div_id"] = div_id
    return pio.to_html(fig, **kwargs)


@lru_cache(maxsize=1)
def _cached_plotlyjs() -> str:
    """Return the Plotly library JS source, cached to avoid repeated reads."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def get_plotly_bundle(offline: bool = True) -> str:
    """Return an HTML snippet that loads the Plotly JavaScript library.

    Args:
        offline: If True (default), inline the full Plotly JS so the report
            works without network access. If False, load from the Plotly CDN.

    Returns:
        HTML ``<script>...</script>`` snippet to inject into the document head.
    """
    if offline:
        return f"<script>{_cached_plotlyjs()}</script>"
    return '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'


def safe_html(s: str) -> str:
    """Escape a string for safe insertion into HTML to prevent XSS."""
    return html_escape(str(s))


def truncate_path(path: str, max_length: int = 50) -> str:
    """
    Truncate a file path for display.

    Args:
        path: File path
        max_length: Maximum length

    Returns:
        Truncated path with ellipsis if needed
    """
    if len(path) <= max_length:
        return path
    return "..." + path[-(max_length - 3) :]
