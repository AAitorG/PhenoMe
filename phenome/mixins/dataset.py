"""
Dataset classes for the phenotyping pipeline.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .._logging import get_logger
from ..io import ensure_hwc, read_image

logger = get_logger(__name__)


class PhenoMeDataset(Dataset):
    """Dataset for loading and preprocessing images for phenotyping analysis.

    Args:
        file_list: List of dicts with 'file_path' and 'metadata' keys.
        transform: Optional torchvision transforms to apply.
        channel_mode: 'split' or 'combined'.
            - 'split': Each selected channel is processed independently and concatenated at embedding level.
            - 'combined': Selected channels are merged into a single image. When ``force_rgb=True``
              only the first 3 selected channels are used (extras discarded, fewer padded to 3).
              When ``force_rgb=False`` all selected channels are kept as-is.
        channels: Optional list of 0-based channel indices to select from the loaded image.
            None means all channels are used. Out-of-range indices are silently skipped.
        preprocessing_fn: Optional callable to apply custom preprocessing to the raw image
            before channel preparation. Should accept (H, W) or (H, W, C) numpy array
            and return numpy array of same shape.
        force_rgb: Whether to force RGB (3 channels) output. Default True.
    """

    def __init__(
        self,
        file_list: list[dict[str, Any]],
        transform: Callable | None = None,
        channel_mode: Literal["split", "combined"] = "split",
        channels: Sequence[int] | None = None,
        preprocessing_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        force_rgb: bool = True,
    ):
        self.file_list = file_list
        self.transform = transform
        self.channel_mode = channel_mode
        self.channels = list(channels) if channels is not None else None
        self.preprocessing_fn = preprocessing_fn
        self.force_rgb = force_rgb

        if channel_mode not in ["split", "combined"]:
            raise ValueError(f"channel_mode must be 'split' or 'combined', got: {channel_mode}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Any:
        item = self.file_list[idx]
        try:
            file_path = item["file_path"]
            img = read_image(file_path)
            img = ensure_hwc(img)

            # Apply custom preprocessing if provided
            if self.preprocessing_fn is not None:
                img = self.preprocessing_fn(img)

            processed_imgs = self._prepare_image_channels(img)

            if self.transform:
                if self.channel_mode == "split":
                    # Split mode: processed_imgs is a list
                    return [self.transform(p) for p in processed_imgs], item
                else:
                    # Combined mode: processed_imgs is a single array
                    return self.transform(processed_imgs), item

            return processed_imgs, item

        except Exception as e:
            file_path = item["file_path"]
            logger.warning(
                "Failed loading: %s (%s)",
                file_path if isinstance(file_path, str) else file_path[0] if file_path else "?",
                e,
            )
            return None

    def _select_channels(self, img: np.ndarray) -> np.ndarray:
        """Return a (H, W, C') view containing only the requested channels.

        Args:
            img: Image array with shape (H, W, C).

        Returns:
            np.ndarray with shape (H, W, len(valid_indices)), or the original array
            when self.channels is None (all channels are used).
        """
        if self.channels is None:
            return img
        n_ch = img.shape[-1]
        valid = [c for c in self.channels if 0 <= c < n_ch]
        if not valid:
            raise ValueError(
                f"No valid channel indices in {self.channels} for image with {n_ch} channel(s)."
            )
        return img[..., valid]

    def _prepare_image_channels(self, img: np.ndarray) -> list[np.ndarray] | np.ndarray:
        """Prepare image channels for model input.

        Channel selection (self.channels) is applied first. In 'combined' mode,
        only the first 3 selected channels are used; additional ones are ignored.

        Args:
            img: Input image array (H, W) or (H, W, C).

        Returns:
            - If channel_mode='split': List of images, one per selected channel.
              If force_rgb=True, each has shape (H, W, 3).
              If force_rgb=False, each has shape (H, W, 1).
            - If channel_mode='combined': Single image with shape (H, W, C').
              If force_rgb=True, C'=3 (only the first 3 selected channels are used;
              extras are discarded, fewer than 3 are padded/repeated to reach 3).
              If force_rgb=False, C' equals the number of selected channels (all kept).
        """
        # Ensure 3D: (H, W, C)
        if img.ndim == 2:
            img = img[..., None]

        # Apply channel selection before any mode-specific processing
        img = self._select_channels(img)

        if self.channel_mode == "split":
            # Split mode: Return list of images, one per selected channel
            if self.force_rgb:
                return [np.repeat(img[..., c : c + 1], 3, axis=-1) for c in range(img.shape[-1])]
            else:
                return [img[..., c : c + 1] for c in range(img.shape[-1])]
        else:  # combined mode
            if not self.force_rgb:
                # Return the selected channels as-is (no RGB forcing, no extra clipping)
                return img

            # Combined mode with force_rgb=True: use only the first 3 selected channels
            img = img[..., :3]
            n_channels = img.shape[-1]

            if n_channels == 1:
                img_rgb = np.repeat(img, 3, axis=-1)
            elif n_channels == 2:
                img_rgb = np.zeros((*img.shape[:2], 3), dtype=img.dtype)
                img_rgb[..., :2] = img
            else:
                # 3 channels (already exactly 3 after the slice above)
                img_rgb = img

            return img_rgb


def collate_fn(batch: list[Any]) -> Any:
    """Custom collate function that handles None values and split/combined channels.

    Args:
        batch: List of (image, item) tuples from dataset. May contain None values.

    Returns:
        Tuple of (batch_tensor, items) where batch_tensor is:
            - (B, C, H, W) for combined mode or single image
            - (B, N_ch, C, H, W) for split channel mode
        Returns (None, None) if batch is empty after filtering.
    """
    batch = [x for x in batch if x is not None]
    if not batch:
        return None, None

    images, items = zip(*batch, strict=False)

    # Check if first image is a list (split mode) or tensor/array (combined mode)
    if isinstance(images[0], list):
        # Split mode: images is tuple of lists
        stacked_per_item = [torch.stack(imgs) for imgs in images]
        batch_tensor = torch.stack(stacked_per_item)  # (B, N_ch, C, H, W)
    else:
        # Combined mode: images is tuple of tensors/arrays
        batch_tensor = torch.stack(images)  # (B, C, H, W)

    return batch_tensor, items
