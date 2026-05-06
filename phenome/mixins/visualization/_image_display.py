"""Image display methods for PhenoMeVisualization."""

import os
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..._logging import get_logger
from ...io import ensure_hwc, read_image
from ...utils.transforms import scale_minmax
from ._helpers import format_property_value as _format_property_value

logger = get_logger(__name__)


class _ImageDisplayMixin:
    """Mixin providing image display and colorization."""

    mean: tuple
    std: tuple
    results: Any
    image_transforms: Any | None

    def _colorize_multichannel(self, img: np.ndarray) -> np.ndarray:
        """Assign colors to each channel and merge into RGB (FIJI-style overlay)."""
        _h, _w, c = img.shape

        # Normalize intensities for display
        img_scaled = scale_minmax(img)

        # Single-channel: replicate to RGB
        if c == 1:
            return np.repeat(img_scaled, 3, axis=-1)

        # Predefined FIJI-style color cycle
        colors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [0.25, 1.0, 0.0],
                [0.0, 0.5, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Select a color for each channel
        color_weights = colors[np.arange(c) % len(colors)]  # (c, 3)

        # Broadcast and sum over channels: (H, W, C, 1) * (1, 1, C, 3) -> (H, W, C, 3) -> (H, W, 3)
        rgb = (img_scaled[..., :, None] * color_weights[None, None, :, :]).sum(axis=2)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def _inverse_transform(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Inverse normalize image tensor for display."""
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        display_img = img_tensor * std + mean
        return np.clip(display_img.permute(1, 2, 0).numpy(), 0, 1)

    def _build_image_title(
        self,
        idx: int,
        metadata: dict[str, Any],
        distance: float | None,
        mask_properties: dict[str, float],
        title_fields: list[str] | None,
    ) -> str:
        """Build title string for image plot."""
        title_lines = [f"Index: {idx}"]

        def find_field_value(field_name: str) -> tuple | None:
            field_lower = field_name.lower()
            if field_lower == "distance" and distance is not None:
                return ("Distance", distance)
            for meta_key, meta_value in metadata.items():
                if meta_key.lower() == field_lower:
                    return (meta_key.capitalize(), meta_value)
            for prop_key, prop_value in mask_properties.items():
                if prop_key.lower() == field_lower:
                    return (prop_key.capitalize(), prop_value)
            return None

        if title_fields is not None:
            title_parts = []
            for field_name in title_fields:
                field_info = find_field_value(field_name)
                if field_info is not None:
                    display_name, value = field_info
                    fmt = (
                        f"{value:.4f}"
                        if display_name.lower() == "distance"
                        else (f"{value:.2f}" if isinstance(value, float) else str(value))
                    )
                    title_parts.append(f"{display_name}: {fmt}")
            if title_parts:
                title_lines.append(" | ".join(title_parts))
        return "\n".join(title_lines)

    def _image_details_text(
        self,
        idx: int,
        all_info: dict[str, Any],
        metadata: dict[str, Any],
        img_name: str,
        img_path: Any,
    ) -> str | None:
        """Build the same multi-line details string as logging in ``_print_image_details``."""
        extra_info = {k: v for k, v in all_info.items() if k != "idx" and v is not None}
        basic_id_fields = {"img_name", "img_path"}
        has_extra = any(k not in basic_id_fields for k in extra_info)
        if not has_extra:
            return None

        lines = [
            f"\n{'=' * 50}",
            f"IMAGE DETAILS - Index {idx}",
            f"{'=' * 50}",
            f"  Name:     {all_info.get('img_name', img_name)}",
            f"  Path:     {all_info.get('img_path', img_path)}",
        ]
        metadata_keys = set(metadata.keys()) if metadata else set()
        metadata_info = {
            k: v for k, v in extra_info.items() if k in metadata_keys and k not in basic_id_fields
        }
        distance_info = {}
        if "distance" in extra_info:
            distance_info["distance"] = extra_info["distance"]
        if "is_reference" in extra_info:
            distance_info["is_reference"] = extra_info["is_reference"]
        properties_info = {
            k: v
            for k, v in extra_info.items()
            if k not in metadata_keys and k not in basic_id_fields and k not in distance_info
        }

        if metadata_info:
            lines.append("  Metadata:")
            for key, value in sorted(metadata_info.items()):
                lines.append(f"    {key.capitalize()}: {value}")
        if distance_info:
            if "distance" in distance_info:
                lines.append(f"  Distance: {distance_info['distance']:.4f}")
            if distance_info.get("is_reference"):
                lines.append("  Is Reference: True")
        if properties_info:
            lines.append(f"{'-' * 50}")
            lines.append("  Properties:")
            for prop_name, prop_value in sorted(properties_info.items()):
                if isinstance(prop_value, (int, float, np.integer, np.floating)):
                    lines.append(
                        f"    {prop_name.capitalize():<30}: {_format_property_value(prop_value)}"
                    )
                else:
                    lines.append(f"    {prop_name.capitalize():<30}: {prop_value}")

        lines.append(f"{'=' * 50}")
        return "\n".join(lines)

    def _print_image_details(
        self,
        idx: int,
        all_info: dict[str, Any],
        metadata: dict[str, Any],
        img_name: str,
        img_path: Any,
    ) -> None:
        """Print formatted image details (metadata, distance, properties) to console."""
        text = self._image_details_text(idx, all_info, metadata, img_name, img_path)
        if text:
            logger.info(text)

    def _load_image_display_data(
        self,
        idx: int,
        distance_results: pd.DataFrame | None = None,
        channels: int | list | None = None,
        title_fields: list[str] | None = None,
        apply_transforms: bool = True,
        downsample: int | None = 720,
    ) -> tuple[np.ndarray, str, bool, dict[str, Any], dict[str, Any], str, Any]:
        """Load and prepare image data for plotting or PNG export.

        Returns:
            display_img, title, was_downsampled, all_info, metadata, img_name, img_path
        """
        n_images = len(self.results.img_path)
        if idx < 0 or idx >= n_images:
            raise ValueError(f"Index {idx} out of range [0, {n_images - 1}]")

        img_path = self.results.img_path[idx]
        img_name = (
            os.path.basename(img_path[0])
            if isinstance(img_path, list)
            else os.path.basename(img_path)
        )

        metadata: dict[str, Any] = {}
        metadata_list = self.results.metadata
        if idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
            metadata = metadata_list[idx]

        use_custom_channels = channels is not None and (
            not isinstance(channels, list)
            or (len(channels) > 0 and not (len(channels) == 1 and channels[0] is None))
        )

        img = read_image(img_path)
        img = ensure_hwc(img)

        preprocessing_fn = getattr(self, "preprocessing_fn", None)
        if preprocessing_fn is not None:
            img = preprocessing_fn(img)

        image_transforms = getattr(self, "image_transforms", None)
        if apply_transforms and image_transforms is not None:
            from torchvision import transforms

            t_list = []
            for t in image_transforms.transforms:
                if not isinstance(t, transforms.Normalize):
                    t_list.append(t)
            if use_custom_channels and channels is not None:
                selected = sorted([channels] if isinstance(channels, int) else list(channels))
                img = img[..., selected]
            img = np.ascontiguousarray(img)
            temp_transform = transforms.Compose(t_list)
            img_tensor = temp_transform(img)
            if isinstance(img_tensor, torch.Tensor):
                img = img_tensor.permute(1, 2, 0).numpy()
            else:
                img = np.array(img_tensor)
                if img.ndim == 3 and img.shape[0] < img.shape[2]:
                    img = img.transpose(1, 2, 0)
        else:
            if image_transforms is None and apply_transforms:
                logger.warning("image_transforms not found, showing raw image.")
            if use_custom_channels and channels is not None:
                selected = sorted([channels] if isinstance(channels, int) else list(channels))
                img = img[..., selected]

        was_downsampled = False
        if downsample is not None and downsample > 0:
            h, w = img.shape[:2]
            longest_edge = max(h, w)
            if longest_edge > downsample:
                scale = downsample / float(longest_edge)
                new_h = max(1, round(h * scale))
                new_w = max(1, round(w * scale))
                step_h = max(1, h // new_h)
                step_w = max(1, w // new_w)
                img = img[::step_h, ::step_w, ...]
                was_downsampled = True
        display_img = self._colorize_multichannel(img)

        if hasattr(self, "get_image_info"):
            all_info = self.get_image_info(idx, distance_results)
        else:
            all_info = {
                "idx": idx,
                "img_name": img_name,
                "img_path": img_path,
            }
            if metadata:
                all_info.update(metadata)
            if distance_results is not None and "distance" in distance_results.columns:
                all_info["distance"] = distance_results.at[idx, "distance"]

        distance = all_info.get("distance")
        basic_fields = {"idx", "img_name", "img_path", "distance", "is_reference"}
        properties_for_title = {
            k: v for k, v in all_info.items() if k not in basic_fields and v is not None
        }

        title = self._build_image_title(idx, metadata, distance, properties_for_title, title_fields)

        return display_img, title, was_downsampled, all_info, metadata, img_name, img_path

    def image_preview_png_bytes(
        self,
        idx: int,
        distance_results: pd.DataFrame | None = None,
        channels: int | list | None = None,
        title_fields: list[str] | None = None,
        show_extra_info: bool = False,
        apply_transforms: bool = True,
        downsample: int | None = 720,
    ) -> tuple[bytes, str | None, str | None]:
        """Rasterize the same view as ``plot_image_by_index`` to PNG bytes.

        Used by the interactive explorer with ``ipywidgets.Image`` because matplotlib
        ``display()`` / ``plt.show()`` from Plotly click callbacks does not reliably
        target a nested ``Output`` widget (output goes to the cell or nowhere).

        When ``show_extra_info`` is True, also emits the same details as
        ``plot_image_by_index`` via logging and returns their text for UI display.

        Returns:
            ``(png_bytes, details_text_or_none)`` — ``details_text_or_none`` is the
            formatted extra-info block when ``show_extra_info`` is True and details exist.
        """
        display_img, title, _was_downsampled, all_info, metadata, img_name, img_path = (
            self._load_image_display_data(
                idx,
                distance_results=distance_results,
                channels=channels,
                title_fields=title_fields,
                apply_transforms=apply_transforms,
                downsample=downsample,
            )
        )
        details_text: str | None = None
        if show_extra_info:
            details_text = self._image_details_text(idx, all_info, metadata, img_name, img_path)
            if details_text:
                logger.info(details_text)

        # Convert numpy array to PIL Image for much faster saving than Matplotlib
        if display_img.dtype != np.uint8:
            if display_img.max() <= 1.0:
                img_uint8 = (display_img * 255).astype(np.uint8)
            else:
                img_uint8 = display_img.astype(np.uint8)
        else:
            img_uint8 = display_img

        pil_img = Image.fromarray(img_uint8)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")

        return buf.getvalue(), details_text, title

    def plot_image_by_index(
        self,
        idx: int,
        distance_results: pd.DataFrame | None = None,
        channels: int | list | None = None,
        figsize: tuple = (6, 6),
        title_fields: list[str] | None = None,
        show_extra_info: bool = True,
        apply_transforms: bool = False,
        downsample: int | None = 720,
        ax: Any = None,
        return_fig: bool = False,
        show_mask_overlay: bool = False,
    ) -> Any:
        """
        Plot a specific image by its index.

        Args:
            idx: Image index in results
            distance_results: Optional distance computation results DataFrame
            channels: specific channels to plot
            figsize: Figure size
            title_fields: Optional list of field names to display in title
            show_extra_info: If True, prints detailed information after plotting
            apply_transforms: If True, apply pipeline image_transforms
            downsample: If not None, approximate desired size (in pixels) for the
                longest image edge when downsampling. The final image size may
                differ slightly due to integer stepping. If None, no downsampling.
            ax: Optional matplotlib axes to plot on. If provided, a new figure is not created.
            return_fig: If True, returns the matplotlib figure object.
            show_mask_overlay: If True, draws the segmentation mask as a semi-transparent
                yellow overlay with a crisp contour. Requires masks to have been discovered
                via ``mask_dir`` in ``find_files``. Useful for verifying that masks load
                correctly and spatially align with their images.
        """
        display_img, title, was_downsampled, all_info, metadata, img_name, img_path = (
            self._load_image_display_data(
                idx,
                distance_results=distance_results,
                channels=channels,
                title_fields=title_fields,
                apply_transforms=apply_transforms,
                downsample=downsample,
            )
        )

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.imshow(display_img, interpolation="nearest")

        if show_mask_overlay:
            _file_df = getattr(self, "_file_df", None)
            _mask_col = getattr(self, "_mask_path_col", None)
            _mask_path = None
            if _file_df is not None and _mask_col is not None and idx < len(_file_df):
                _mask_path = _file_df.iloc[idx][_mask_col]
            if _mask_path and isinstance(_mask_path, str):
                try:
                    _mask = ensure_hwc(read_image(_mask_path))
                    _mask_2d = (_mask.max(axis=-1) > 0.5).astype(np.float32)
                    if _mask_2d.shape != display_img.shape[:2]:
                        _pil_mask = Image.fromarray((_mask_2d * 255).astype(np.uint8))
                        _mask_2d = (
                            np.array(
                                _pil_mask.resize(
                                    (display_img.shape[1], display_img.shape[0]),
                                    Image.NEAREST,
                                )
                            )
                            / 255.0
                        )

                    # Create RGBA overlay (Yellow with very faint opacity to not obscure bio details)
                    _rgba = np.zeros((*_mask_2d.shape, 4), dtype=np.float32)
                    _rgba[..., 0] = 1.0  # R
                    _rgba[..., 1] = 1.0  # G (R+G = Yellow)
                    _rgba[..., 3] = np.where(_mask_2d > 0.5, 0.15, 0.0)  # Faint alpha
                    ax.imshow(_rgba, interpolation="nearest")

                    # Add a crisp contour to clearly delineate the foreground object
                    if _mask_2d.max() > 0.5:
                        ax.contour(
                            _mask_2d, levels=[0.5], colors="yellow", linewidths=1.0, alpha=1.0
                        )
                except Exception as exc:
                    logger.warning("Could not load mask for overlay (idx=%d): %s", idx, exc)
            else:
                logger.warning(
                    "show_mask_overlay=True but no mask path found for idx=%d. "
                    "Call find_files with mask_dir to enable mask overlay.",
                    idx,
                )

        # If image was downsampled for visualization, hide axes for a cleaner look.
        ax.axis("off" if (was_downsampled or ax is not None) else "on")
        ax.set_title(title, fontsize=11)

        if fig is not None:
            fig.tight_layout()
            # Use IPython display when available: ``plt.show()`` then immediate ``plt.close()``
            # often drops the figure in notebooks/VS Code, and ``plt.show()`` does not reliably
            # target a nested ``ipywidgets.Output`` (e.g. interactive click-to-inspect).
            try:
                from IPython import get_ipython
                from IPython.display import display as ipy_display_fig
            except ImportError:
                ipython_shell = None
            else:
                ipython_shell = get_ipython()

            if not return_fig:
                if ipython_shell is not None:
                    ipy_display_fig(fig)
                else:
                    plt.show()
                plt.close(fig)

        if show_extra_info:
            self._print_image_details(idx, all_info, metadata, img_name, img_path)

        return fig if return_fig else None
