"""Torch dataset for the F-actin conformations dataset.

The dataset on disk is a pair of flat folders::

    factin-conformations-dataset/
        images/<uuid>.tif        # (H, W) uint16 STED image
        masks/<uuid>_mask.tif    # (3, H, W) uint16 binary annotation stack

The three mask channels are the annotated F-actin conformations, in channel order:
'Nanoclusters', 'Compact assemblies' and 'Protrusions'.

`FActinConformationsDataset` mirrors the conventions of STED-FM's
`NeuralActivityStates`: images are read once into memory, intensities are min-max
normalized, a single channel tile is optionally tiled into an RGB image normalized
with the STED-FM statistics, and the user transform is applied last. The original
images are larger than the 224x224 input the models expect, so they are cut into a
regular grid of tiles and `__getitem__` returns one tile with its matching mask crop.
"""

import glob
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

DATASET_PATH = "/Users/renaud/datasets/factin-conformations-dataset"

# F-actin conformations, in mask channel order
CLASS_NAMES = ("Nanoclusters", "Compact assemblies", "Protrusions")

# get_metadata key per mask channel, e.g. 'Compact assemblies' -> 'mask_fraction_compact_assemblies'
MASK_FRACTION_KEYS = tuple(
    f"mask_fraction_{name.lower().replace(' ', '_')}" for name in CLASS_NAMES
)

# Intensity statistics of the min-max normalized STED-FM F-actin tiles
STEDFM_MEAN = 0.023
STEDFM_STD = 0.027


class FActinConformationsDataset(Dataset):
    """Tile dataset over the annotated F-actin conformations images.

    Every image is cut into a grid of `image_size` tiles; `__getitem__` returns the
    tile and the matching crop of the annotation stack. Tile starts are spaced by
    `image_size * step` and the last start of each axis is clamped to the image
    border, so the whole image is covered without relying on padding. Images smaller
    than `image_size` are the only ones padded (symmetrically, image and mask alike).

    Attributes:
        classes: Names of the returned mask channels, in channel order.
        samples: One (image index, row, col) tuple per tile, in `__getitem__` order.
    """

    def __init__(
        self,
        path: str = DATASET_PATH,
        transform: Callable | None = None,
        n_channels: int = 1,
        image_size: int = 224,
        step: float = 1.0,
        normalize: str = "image",
        min_annotated_ratio: float = 0.0,
        mean: float = STEDFM_MEAN,
        std: float = STEDFM_STD,
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        mask_suffix: str = "_mask",
        return_metadata: bool = False,
    ) -> None:
        """Read the dataset folder and enumerate the tiles.

        Args:
            path: Root folder holding the `images_subdir` and `masks_subdir` folders.
            transform: Transform applied last, to the (n_channels, image_size,
                image_size) float32 tensor. None applies no transform.
            n_channels: 1 keeps the tile single channel, 3 tiles it into an RGB image
                normalized with `mean` and `std`.
            image_size: Side length of the square tiles.
            step: Tile stride as a fraction of `image_size`; 1.0 gives contiguous
                tiles, smaller values overlapping ones.
            normalize: 'image' min-max normalizes each original image before tiling
                (the STED-FM convention), 'tile' normalizes each tile independently
                and 'none' keeps the raw intensities as float32.
            min_annotated_ratio: Minimum fraction of tile pixels annotated in any
                conformation channel; tiles below it are discarded. 0.0 keeps every
                tile of every image.
            mean: Per-channel mean used by the `n_channels=3` normalization.
            std: Per-channel standard deviation used by the same normalization.
            images_subdir: Name of the image folder inside `path`.
            masks_subdir: Name of the mask folder inside `path`.
            mask_suffix: Suffix appended to the image stem to build the mask name.
            return_metadata: When True, `__getitem__` also returns the tile metadata
                dict built by `get_metadata`.

        Raises:
            FileNotFoundError: No image was found in `path/images_subdir`, or an
                image has no matching mask file.
            ValueError: `normalize` is not one of 'image', 'tile' or 'none', or an
                image and its mask have mismatched shapes.
        """
        if normalize not in ("image", "tile", "none"):
            raise ValueError(f"normalize must be 'image', 'tile' or 'none', got: {normalize}")

        self.path = path
        self.transform = transform
        self.n_channels = n_channels
        self.image_size = image_size
        self.step = step
        self.normalize = normalize
        self.min_annotated_ratio = min_annotated_ratio
        self.mean = mean
        self.std = std
        self.return_metadata = return_metadata

        self.classes = list(CLASS_NAMES)
        self.num_classes = len(self.classes)

        images_dir = os.path.join(path, images_subdir)
        masks_dir = os.path.join(path, masks_subdir)
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
        if not image_paths:
            raise FileNotFoundError(f"No .tif image found in {images_dir}")

        self.image_names: list[str] = []
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        for image_path in tqdm(image_paths, desc="Processing dataset.."):
            name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(masks_dir, f"{name}{mask_suffix}.tif")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"No mask {mask_path} for image {image_path}")

            image = tifffile.imread(image_path).astype(np.float32)
            mask = tifffile.imread(mask_path)
            if mask.ndim != 3 or mask.shape[0] != len(CLASS_NAMES):
                raise ValueError(
                    f"Expected a ({len(CLASS_NAMES)}, H, W) mask in {mask_path}, got {mask.shape}"
                )
            if mask.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"Mask {mask_path} has shape {mask.shape[-2:]} but image {image_path} "
                    f"has shape {image.shape[-2:]}"
                )

            if self.normalize == "image":
                image = minmax_normalize(image)

            self.image_names.append(name)
            self.images.append(image)
            self.masks.append(mask > 0)

        self.samples = self.generate_valid_samples()

    def generate_valid_samples(self) -> list[tuple[int, int, int]]:
        """Enumerate the tiles kept by `min_annotated_ratio`.

        Returns:
            List of (image index, row, col) tuples, where row and col are the
            top-left corner of the tile in the original image.
        """
        min_annotated = self.min_annotated_ratio * self.image_size**2
        samples = []
        for index, mask in enumerate(self.masks):
            height, width = mask.shape[-2:]
            for row in self._tile_starts(height):
                for col in self._tile_starts(width):
                    if min_annotated > 0:
                        crop = self._mask_crop(index, row, col)
                        if crop.any(axis=0).sum() < min_annotated:
                            continue
                    samples.append((index, row, col))
        return samples

    def _mask_crop(self, index: int, row: int, col: int) -> np.ndarray:
        """Return the boolean (len(classes), h, w) mask crop of one tile.

        Args:
            index: Index of the original image in `self.masks`.
            row: Top-left row of the tile in that image.
            col: Top-left column of the tile in that image.

        Returns:
            View of the annotation stack. It is smaller than `image_size` on an axis only
            when the original image is, which is the case `__getitem__` pads.
        """
        return self.masks[index][:, row : row + self.image_size, col : col + self.image_size]

    def _tile_starts(self, extent: int) -> list[int]:
        """List the tile start positions covering `extent` pixels along one axis."""
        last = max(extent - self.image_size, 0)
        stride = max(1, round(self.image_size * self.step))
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    def __len__(self) -> int:
        return len(self.samples)

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Describe the tile at `idx` without decoding its image.

        Only the in-memory annotation stack is read, to measure how much of the tile each
        conformation covers.

        Args:
            idx: Index of the tile, in the same order as `__getitem__`.

        Returns:
            Dict with the original 'image_name', its 'image-idx' in the dataset, the
            tile's top-left 'row' and 'col' in that image, the tile's own 'dataset-idx',
            and one `MASK_FRACTION_KEYS` entry per mask channel holding the fraction of
            the tile it annotates. Fractions are relative to `image_size ** 2`, the same
            denominator as `min_annotated_ratio`, and are measured on the unpadded crop,
            so for an image smaller than `image_size` they ignore the symmetric padding
            `__getitem__` adds.
        """
        index, row, col = self.samples[idx]
        # Fractions rather than pixel counts: whole numbers with few distinct values are
        # plotted as a discrete legend by PhenoMe's get_color_column, fractions get a colorbar
        fractions = self._mask_crop(index, row, col).sum(axis=(1, 2)) / self.image_size**2
        return {
            "image_name": self.image_names[index],
            "image-idx": index,
            "row": row,
            "col": col,
            "dataset-idx": idx,
            **{
                key: float(fraction)
                for key, fraction in zip(MASK_FRACTION_KEYS, fractions, strict=True)
            },
        }

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return one image tile and its mask crop, optionally with metadata.

        Args:
            idx: Index of the tile.

        Returns:
            Tuple of (image, mask) where image has shape (n_channels, image_size,
            image_size) and mask (len(classes), image_size, image_size), both
            float32. When the dataset was built with `return_metadata=True`,
            `get_metadata(idx)` is appended as a third element.
        """
        index, row, col = self.samples[idx]

        image_crop = self.images[index][row : row + self.image_size, col : col + self.image_size]
        mask_crop = self._mask_crop(index, row, col)

        # Only images smaller than a tile need padding; the grid covers the rest
        if image_crop.shape != (self.image_size, self.image_size):
            pad = (
                (0, self.image_size - image_crop.shape[-2]),
                (0, self.image_size - image_crop.shape[-1]),
            )
            image_crop = np.pad(image_crop, pad, mode="symmetric")
            mask_crop = np.pad(mask_crop, ((0, 0), *pad), mode="symmetric")

        if self.normalize == "tile":
            image_crop = minmax_normalize(image_crop)

        if self.n_channels == 3:
            img = np.tile(image_crop[np.newaxis], (3, 1, 1))
            img = torch.tensor(img, dtype=torch.float32)
            img = transforms.Normalize(mean=[self.mean] * 3, std=[self.std] * 3)(img)
        else:
            img = torch.tensor(image_crop[np.newaxis], dtype=torch.float32)

        img = self.transform(img) if self.transform is not None else img
        mask = torch.tensor(mask_crop, dtype=torch.float32)

        if self.return_metadata:
            return img, mask, self.get_metadata(idx)

        return img, mask

    def __repr__(self) -> str:
        return (
            f"Dataset(F-actin conformations) -- {len(self)} tiles of "
            f"{self.image_size}x{self.image_size} from {len(self.images)} images\n"
            f"classes: {', '.join(self.classes)}"
        )


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1].

    The raw images are uint16 whose maxima sit far below the 16-bit range, so the
    per-image min-max used by STED-FM is what gives the model usable contrast.

    Args:
        image: Array of any shape.

    Returns:
        float32 array of the same shape, all zeros when the input is constant.
    """
    image = np.asarray(image, dtype=np.float32)
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


def main():
    dataset = FActinConformationsDataset(
        path=DATASET_PATH,
        n_channels=1,
        image_size=224,
    )
    print(dataset)

    for img, mask in dataset:
        print(img.shape, mask.shape)


if __name__ == "__main__":
    main()
