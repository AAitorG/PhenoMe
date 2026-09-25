"""Helpers to turn the F-actin conformations dataset into a PhenoMe-ready folder layout.

The Zenodo archive holds full-size STED images (`<uuid>.tif`, H x W) and their
3-channel binary annotation masks (`<uuid>_mask.tif`, C x H x W). `ActinDataset`
pairs them and serves non-overlapping square tiles, and `mask_channel_ratios`
summarizes how much of a tile each annotated structure covers.
"""

import io
import os
import warnings
import zipfile
from collections.abc import Callable, Sequence

import numpy as np
import tifffile
from torch.utils.data import Dataset
from tqdm import tqdm

# Mask channel order in the F-actin conformations dataset
CHANNEL_NAMES = ("nanoclusters", "compact_assemblies", "protrusions")

_MASK_SUFFIX = "_mask"


def mask_channel_ratios(
    mask: np.ndarray, channel_names: Sequence[str] = CHANNEL_NAMES
) -> dict[str, float]:
    """Compute the proportion of pixels covered by each channel of a mask.

    Args:
        mask: Mask of shape (C, H, W), or (H, W) for a single channel. Any strictly
            positive pixel counts as covered.
        channel_names: One name per channel, used as the key prefix. When the number
            of names does not match the number of channels, generic
            `channel_<index>` names are used instead.

    Returns:
        A mapping from `<name>_ratio` to the covered fraction, in [0, 1].
    """
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[np.newaxis]
    if len(channel_names) != mask.shape[0]:
        channel_names = [f"channel_{c}" for c in range(mask.shape[0])]
    return {
        f"{name}_ratio": float((channel > 0).mean())
        for name, channel in zip(channel_names, mask, strict=True)
    }


class ActinDataset(Dataset):
    """Non-overlapping tiles of the F-actin conformations dataset, read from its zip.

    Each item is an image tile and a metadata dict holding the source image stem,
    the tile's top-left offset and the matching mask tile. Tiles that would extend
    past the image border are dropped.

    Attributes:
        archive_path: Path to the dataset .zip file.
        tile_size: Side length of the square tiles, in pixels.
        transform: Optional callable applied to each image tile.
        images: Full-size images keyed by stem.
        masks: Full-size masks keyed by stem, each of shape (C, H, W).
        tiles: `(stem, y, x)` tuples, one per tile.
    """

    def __init__(
        self,
        archive_path: str,
        tile_size: int = 224,
        transform: Callable | None = None,
    ) -> None:
        """Loads every image/mask pair from the archive and indexes the tiles.

        Args:
            archive_path: Path to the dataset .zip file.
            tile_size: Side length of the square tiles, in pixels.
            transform: Optional callable applied to each image tile.
        """
        super().__init__()
        self.archive_path = archive_path
        self.tile_size = tile_size
        self.transform = transform

        images: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(archive_path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith((".tif", ".tiff"))]
            for member in tqdm(members, desc="... Loading dataset from archive ..."):
                stem = os.path.splitext(os.path.basename(member))[0]
                array = tifffile.imread(io.BytesIO(archive.read(member)))
                if stem.endswith(_MASK_SUFFIX):
                    masks[stem[: -len(_MASK_SUFFIX)]] = array
                else:
                    images[stem] = array

        unpaired = sorted(set(images) ^ set(masks))
        if unpaired:
            warnings.warn(f"Skipping {len(unpaired)} unpaired file(s): {unpaired}", stacklevel=2)
        stems = sorted(set(images) & set(masks))
        self.images = {stem: images[stem] for stem in stems}
        self.masks = {
            stem: masks[stem][np.newaxis] if masks[stem].ndim == 2 else masks[stem]
            for stem in stems
        }

        self.tiles: list[tuple[str, int, int]] = []
        for stem in stems:
            height, width = self.images[stem].shape[-2:]
            for y in range(0, height - tile_size + 1, tile_size):
                for x in range(0, width - tile_size + 1, tile_size):
                    self.tiles.append((stem, y, x))

    def __len__(self) -> int:
        """Returns the number of tiles."""
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        """Returns one image tile and its metadata.

        Args:
            idx: Index of the tile.

        Returns:
            The image tile of shape (tile_size, tile_size) as float32 (after
            `transform`, if any), and a dict with the source image stem (`source`),
            the tile's top-left offset (`y`, `x`) and the mask tile (`mask`, of shape
            (C, tile_size, tile_size)).
        """
        stem, y, x = self.tiles[idx]
        window = (slice(y, y + self.tile_size), slice(x, x + self.tile_size))
        image = self.images[stem][window].astype(np.float32)
        mask = self.masks[stem][(slice(None), *window)]

        if self.transform is not None:
            image = self.transform(image)

        metadata = {"source": stem, "y": y, "x": x, "mask": mask}
        return image, metadata
