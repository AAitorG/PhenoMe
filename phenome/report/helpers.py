"""
Utility functions for the phenotyping report.

This module provides helper functions for:
- Plotly figure styling (dark theme)
- Image encoding to base64
- HTML escaping for safe insertion of user content
"""

import base64
from html import escape as html_escape
from io import BytesIO
from typing import TYPE_CHECKING, Union

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

    # Resize if specified
    if size is not None:
        pil_image = pil_image.resize(size, PILImage.Resampling.LANCZOS)

    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format=format, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
