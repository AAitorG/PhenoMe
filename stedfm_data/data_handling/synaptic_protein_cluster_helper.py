"""Cluster-level variant of `synaptic_protein_helper`: one crop per synaptic cluster.

`export_object_crops` treats every connected component of a class channel as an object and
cuts a fixed-size window around its centroid. This module instead exports one crop per
*cluster*: components whose closest pixels are within `min_distance` are merged into a single
ROI, and each crop is that ROI's own bounding box, so what is kept follows the object rather
than a constant window. The bounding box is then centered in a zero-filled canvas (224x224
by default), which gives the crops a uniform size without resizing the object.

The output layout is unchanged, so the result still feeds
`pheno.find_files(images_dir, mask_dir=masks_dir,
metadata_fn=make_dataframe_metadata_fn(metadata_df))`. Crops are cut from the
full-resolution original images held in `dataset.images`, and every recorded coordinate is
global. Everything that does not depend on the two changes above is imported from
`synaptic_protein_helper`.
"""

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import scipy.ndimage
import skimage
import tifffile

from .synaptic_protein_helper import (
    CLASS_NAMES,
    _scale_image,
    extract_crop,
    match_annotation,
    minmax_preprocessing_fn,
    parse_image_name,
    threshold_mask,
)

__all__ = [
    "CLASS_NAMES",
    "bbox_window",
    "count_components",
    "export_cluster_crops",
    "find_clusters",
    "group_nearby_objects",
    "minmax_preprocessing_fn",
    "pad_crop",
]


def group_nearby_objects(binary: np.ndarray, min_distance: float = 0.0) -> np.ndarray:
    """Label a binary mask, merging components closer than `min_distance`.

    Two components are considered the same ROI when their closest pixels are at most
    `min_distance` apart, which is resolved on the pixel grid and so is exact to within a
    pixel. Merged components keep a single label but are not filled in: only the annotated
    pixels carry it.

    Args:
        binary: 2D boolean array, one class channel of a thresholded label stack.
        min_distance: Minimum gap in pixels between two components for them to stay
            separate ROIs. 0 labels plain connected components, as `find_objects` does.

    Returns:
        int array of the same shape as `binary`, 0 on background. Label values are not
        contiguous and a single label may cover disconnected pieces.
    """
    if min_distance <= 0:
        return skimage.measure.label(binary)

    # A halo of min_distance / 2 around every component bridges the pairs whose closest
    # pixels are at most min_distance apart; masking back keeps the true object pixels.
    distance = scipy.ndimage.distance_transform_edt(~binary)
    bridged = skimage.measure.label(distance <= min_distance / 2)
    return np.where(binary, bridged, 0)


def find_clusters(
    class_masks: np.ndarray,
    threshold: float = 0.0,
    min_area: int = 9,
    min_distance: float = 0.0,
) -> Iterator[tuple[int, np.ndarray, Any]]:
    """Find the annotated clusters of every class in one image channel.

    Each class channel is grouped independently, which keeps objects of different classes
    separate even when they overlap or sit within `min_distance` of each other.

    Args:
        class_masks: Soft label stack of shape (n_classes, H, W).
        threshold: Passed to `threshold_mask`.
        min_area: Minimum region area in pixels, applied to the merged region; smaller
            regions are dropped.
        min_distance: Passed to `group_nearby_objects`.

    Yields:
        Tuple of (class_index, label_image, region), where `label_image` is the grouped
        labelling of that class channel and `region` a `skimage.measure.regionprops` entry.
    """
    binary = threshold_mask(class_masks, threshold=threshold)

    for class_index in range(binary.shape[0]):
        label_image = group_nearby_objects(binary[class_index], min_distance=min_distance)
        for region in skimage.measure.regionprops(label_image):
            if region.area < min_area:
                continue
            yield class_index, label_image, region


def bbox_window(bbox: tuple[int, int, int, int], padding: int = 0) -> tuple[int, int, int, int]:
    """Window covering a region bounding box, optionally grown by `padding` pixels.

    The window is not clipped to the image; `extract_crop` pads whatever falls outside.

    Args:
        bbox: (min_row, min_col, max_row, max_col) as returned by `region.bbox`; the stop
            bounds are exclusive.
        padding: Number of pixels added on each side of the bounding box.

    Returns:
        Tuple (row0, col0, row1, col1); the stop bounds are exclusive.
    """
    min_row, min_col, max_row, max_col = bbox
    return min_row - padding, min_col - padding, max_row + padding, max_col + padding


def pad_crop(crop: np.ndarray, size: int | tuple[int, int], fill: int = 0) -> np.ndarray:
    """Center a crop in a constant-filled canvas of `size`.

    Unlike `extract_crop`, the border is filled with `fill` rather than read from the
    original image, so the crop keeps showing its ROI alone. An axis of the crop already
    at least as long as the target is left untouched, so a large cluster is never cut.

    Args:
        crop: 2D array to pad.
        size: Target (height, width), or a single int for a square canvas.
        fill: Value used for the padding.

    Returns:
        Array of shape (max(height, crop height), max(width, crop width)) with the dtype of
        `crop`.
    """
    height, width = (size, size) if isinstance(size, int) else size
    crop_height, crop_width = crop.shape
    if crop_height >= height and crop_width >= width:
        return crop

    padded = np.full((max(height, crop_height), max(width, crop_width)), fill, dtype=crop.dtype)
    row0 = (padded.shape[0] - crop_height) // 2
    col0 = (padded.shape[1] - crop_width) // 2
    padded[row0 : row0 + crop_height, col0 : col0 + crop_width] = crop
    return padded


def count_components(region: Any) -> int:
    """Count the connected components merged into one grouped region.

    Args:
        region: A `skimage.measure.regionprops` entry from `find_clusters`.

    Returns:
        Number of connected components carrying the region's label.
    """
    return int(skimage.measure.label(region.image).max())


def export_cluster_crops(
    dataset: Any,
    output_dir: str,
    min_distance: float = 0.0,
    bbox_padding: int = 0,
    pad_to_size: int | tuple[int, int] | None = 224,
    threshold: float = 0.0,
    min_area: int = 9,
    intensity_mode: str = "raw",
    images_subdir: str = "images",
    masks_subdir: str = "masks",
    metadata_filename: str = "metadata.csv",
    attach_annotation_metadata: bool = True,
    max_match_distance: float = 40.0,
    scale_percentile: float = 99.9,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Export every annotated synaptic cluster as an image crop, mask and metadata row.

    For each original image and image channel, the soft label stack is thresholded,
    components closer than `min_distance` are merged per class, and the bounding box of each
    resulting cluster is cut from the full-resolution image. Each bounding-box crop is then
    centered in a `pad_to_size` canvas of zeros, so the crops share one size while every
    cluster keeps its own scale; `pad_to_size=None` writes the bounding boxes as they are,
    and they then vary in size. The mask written next to a crop holds that cluster only,
    since PhenoMe's shape properties use the largest connected region of the mask and its
    masked intensity statistics use `mask > 0.5`.

    Args:
        dataset: A `SemanticProteinSegmentationDataset` instance.
        output_dir: Destination directory; `images_subdir`, `masks_subdir` and the metadata
            csv are created inside it.
        min_distance: Minimum gap in pixels between two annotated components for them to be
            exported as separate ROIs; closer ones are merged into a single crop. 0 keeps the
            plain connected-component behaviour of `export_object_crops`.
        bbox_padding: Pixels added on each side of the bounding box. Padding that falls
            outside the image is filled with zeros.
        pad_to_size: Target (height, width) of the written crops, or a single int for a
            square canvas; the bounding-box crop is centered in it and the rest is filled
            with zeros. A cluster larger than the target is written at its own size rather
            than cut, so crops are uniformly sized only when every cluster fits. None writes
            the bounding-box crops as they are.
        threshold: Mask threshold passed to `find_clusters`.
        min_area: Minimum cluster area in pixels, measured after merging.
        intensity_mode: 'raw' keeps the uint16 values from the HDF5;
            'image_percentile' divides each image channel by its `scale_percentile` and clips
            to [0, 1]; 'crop_minmax' stretches every crop individually. The latter two
            produce float32 crops.
        images_subdir: Sub-directory for the image crops.
        masks_subdir: Sub-directory for the instance masks.
        metadata_filename: Name of the csv written in `output_dir`.
        attach_annotation_metadata: Match each cluster to the closest annotation from
            `dataset.metadata` and add its vote-derived columns. A merged cluster is matched
            to the single closest annotation.
        max_match_distance: Maximum distance in pixels between a cluster centroid and an
            annotation for them to be matched. The default matches the +/-40 px half-window
            `create_segmentation` writes around each `id_coord`.
        scale_percentile: Percentile used by `intensity_mode='image_percentile'`.
        overwrite: Overwrite existing crop files instead of raising. Same-named files are
            replaced, but crops left over from a previous run with different settings are not
            deleted; clear `output_dir` yourself when re-exporting with a different
            `threshold`, `min_area`, `min_distance` or `bbox_padding`.

    Returns:
        pd.DataFrame: The metadata table that was written, one row per crop. The 'filename'
        column holds the extension-less stem shared by the image and its mask, which is what
        `make_dataframe_metadata_fn` matches on by default.
    """
    images_dir = os.path.join(output_dir, images_subdir)
    masks_dir = os.path.join(output_dir, masks_subdir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    annotations = getattr(dataset, "metadata", []) if attach_annotation_metadata else []
    rows = []

    for img_name, entry in dataset.images.items():
        image_stem = os.path.splitext(img_name)[0]
        image_metadata = parse_image_name(img_name)

        for channel in range(entry["image"].shape[0]):
            image = _scale_image(entry["image"][channel], intensity_mode, scale_percentile)
            height, width = image.shape

            for object_index, (class_index, label_image, region) in enumerate(
                find_clusters(
                    entry["label"][channel],
                    threshold=threshold,
                    min_area=min_area,
                    min_distance=min_distance,
                )
            ):
                window = bbox_window(region.bbox, padding=bbox_padding)
                image_crop = extract_crop(image, window)
                if intensity_mode == "crop_minmax":
                    image_crop = minmax_preprocessing_fn(image_crop)

                instance = (label_image == region.label).astype(np.uint8) * 255
                mask_crop = extract_crop(instance, window)

                if pad_to_size is not None:
                    # Padded after 'crop_minmax' so the fill does not enter the crop statistics
                    image_crop = pad_crop(image_crop, pad_to_size)
                    mask_crop = pad_crop(mask_crop, pad_to_size)

                class_name = CLASS_NAMES[class_index]
                filename = f"{image_stem}_ch{channel}_{class_name}_{object_index:04d}"
                image_path = os.path.join(images_dir, f"{filename}.tif")
                mask_path = os.path.join(masks_dir, f"{filename}.tif")
                if not overwrite and (os.path.exists(image_path) or os.path.exists(mask_path)):
                    raise FileExistsError(
                        f"{filename}.tif already exists in {output_dir}; "
                        "pass overwrite=True to replace it."
                    )

                tifffile.imwrite(image_path, image_crop)
                tifffile.imwrite(mask_path, mask_crop)

                row0, col0, row1, col1 = window
                min_row, min_col, max_row, max_col = region.bbox
                row = {
                    "filename": filename,
                    "image_name": img_name,
                    "channel": channel,
                    "protein": image_metadata.get(f"protein_ch{channel}"),
                    "partner_protein": image_metadata.get(f"protein_ch{1 - channel}"),
                    "antibody_label": image_metadata.get(f"label_ch{channel}"),
                    "condition": image_metadata.get("condition"),
                    "replicate": image_metadata.get("replicate"),
                    "synapse_class": class_name,
                    "class_index": class_index,
                    "centroid_row": region.centroid[0],
                    "centroid_col": region.centroid[1],
                    "bbox_row_min": min_row,
                    "bbox_col_min": min_col,
                    "bbox_row_max": max_row,
                    "bbox_col_max": max_col,
                    "area": int(region.area),
                    "n_components": count_components(region),
                    "crop_height": image_crop.shape[0],
                    "crop_width": image_crop.shape[1],
                    "content_height": row1 - row0,
                    "content_width": col1 - col0,
                    "min_distance": min_distance,
                    "bbox_padding": bbox_padding,
                    "touches_image_border": bool(
                        min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width
                    ),
                }
                if attach_annotation_metadata:
                    row.update(
                        match_annotation(
                            annotations,
                            img_name,
                            channel,
                            region.centroid,
                            max_distance=max_match_distance,
                        )
                    )
                rows.append(row)

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(os.path.join(output_dir, metadata_filename), index=False)
    return metadata_df
