"""Turn the semantic synaptic-protein segmentation dataset into a PhenoMe-ready
folder layout: one image crop per annotated synapse, a matching instance mask, and
a metadata table keyed by filename.

The `SemanticProteinSegmentationDataset` exposes large semantic segmentation tiles
(image + a 4-channel soft label stack, one channel per synapse class). PhenoMe
instead expects single-object crops discovered with `pheno.find_files(images_dir,
mask_dir=masks_dir, metadata_fn=make_dataframe_metadata_fn(metadata_df))`.
`export_object_crops` performs that conversion.

Crops are cut from the full-resolution original images held in `dataset.images`
rather than from the 1000x1000 tiles, so objects are never split by a tile edge and
all recorded coordinates are global.
"""

import os
import re

import numpy as np
import pandas as pd
import skimage
import tifffile

# Mirrors SemanticProteinSegmentationDataset.classes (mask channel order)
CLASS_NAMES = ("round", "elongated", "perforated", "multidomains")

# R-Homer_GARSTAR635_M-PSD95_GAMAlexa594_GluGly_01.tif and the '-' separated variant
# R-Rim_GARSTAR635-M-Bassoon_GAMAlexa594_0MgGlyBic_02.tif
_IMAGE_NAME_PATTERN = re.compile(
    r"^R-(?P<protein_ch0>[^_-]+)_(?P<label_ch0>[^_-]+)[_-]"
    r"M-(?P<protein_ch1>[^_-]+)_(?P<label_ch1>[^_-]+)_"
    r"(?P<condition>[^_]+)_(?P<replicate>[^_.]+)"
)


def threshold_mask(mask, threshold=0.0):
    """Binarize a soft label mask.

    The label stacks built by `create_segmentation` accumulate the mean of the
    annotator votes, so any strictly positive pixel was annotated at least once.

    Args:
        mask: Array of any shape, e.g. (H, W) or (n_classes, H, W).
        threshold: Pixels strictly above this value are foreground.

    Returns:
        Boolean array with the same shape as `mask`.
    """
    return np.asarray(mask) > threshold


def find_objects(class_masks, threshold=0.0, min_area=9):
    """Find the individual annotated objects of every class in one image channel.

    Each class channel is labelled independently, which keeps objects of different
    classes separate even when they overlap. Two touching objects of the *same*
    class do merge into a single component; `min_area` only removes speckle, not
    that ambiguity.

    Args:
        class_masks: Soft label stack of shape (n_classes, H, W).
        threshold: Passed to `threshold_mask`.
        min_area: Minimum region area in pixels; smaller regions are dropped.

    Yields:
        Tuple of (class_index, label_image, region), where `label_image` is the
        connected-component labelling of that class channel and `region` a
        `skimage.measure.regionprops` entry.
    """
    binary = threshold_mask(class_masks, threshold=threshold)

    for class_index in range(binary.shape[0]):
        label_image = skimage.measure.label(binary[class_index])
        for region in skimage.measure.regionprops(label_image):
            if region.area < min_area:
                continue
            yield class_index, label_image, region


def centered_window(centroid, crop_size):
    """Fixed-size window centered on a (row, col) centroid.

    The window is not clipped to the image; `extract_crop` pads whatever falls
    outside so that every crop ends up the same size.

    Args:
        centroid: (row, col) floats, e.g. `region.centroid`.
        crop_size: Side length of the square window in pixels.

    Returns:
        Tuple (row0, col0, row1, col1); the stop bounds are exclusive.
    """
    row, col = centroid
    row0 = int(round(row)) - crop_size // 2
    col0 = int(round(col)) - crop_size // 2
    return row0, col0, row0 + crop_size, col0 + crop_size


def extract_crop(array, window, fill=0):
    """Extract `window` from `array`, padding the part that falls outside.

    Args:
        array: 2D array to crop.
        window: (row0, col0, row1, col1) as returned by `centered_window`.
        fill: Value used for the out-of-image padding.

    Returns:
        Array of shape (row1 - row0, col1 - col0) with the dtype of `array`.
    """
    row0, col0, row1, col1 = window
    height, width = array.shape

    src_row0, src_col0 = max(row0, 0), max(col0, 0)
    src_row1, src_col1 = min(row1, height), min(col1, width)

    crop = np.full((row1 - row0, col1 - col0), fill, dtype=array.dtype)
    if src_row0 < src_row1 and src_col0 < src_col1:
        crop[src_row0 - row0:src_row1 - row0, src_col0 - col0:src_col1 - col0] = (
            array[src_row0:src_row1, src_col0:src_col1]
        )
    return crop


def parse_image_name(img_name):
    """Extract the experimental metadata encoded in an original image filename.

    Names follow `R-<protein>_<label>_M-<protein>_<label>_<condition>_<replicate>.tif`
    where R-/M- are the rabbit and mouse primary antibodies imaged in channel 0 and
    channel 1 respectively.

    Args:
        img_name: Original image name, with or without the `.tif` extension.

    Returns:
        Dict with 'protein_ch0', 'label_ch0', 'protein_ch1', 'label_ch1',
        'condition' and 'replicate', or an empty dict if the name does not match.
    """
    match = _IMAGE_NAME_PATTERN.match(os.path.splitext(img_name)[0])
    return match.groupdict() if match else {}


def match_annotation(annotations, img_name, channel, centroid, max_distance):
    """Find the per-object annotation record closest to `centroid`.

    `id_coord` in the dataset metadata is expressed in full-image coordinates (that
    is how `create_segmentation` slices it), so centroids can be matched directly.

    Args:
        annotations: Sequence of metadata dicts, e.g. `dataset.metadata`.
        img_name: Original image name to match on.
        channel: Image channel to match on.
        centroid: (row, col) of the object in full-image coordinates.
        max_distance: Maximum euclidean distance for a match, in pixels.

    Returns:
        Dict with 'cluster_id', 'class_majority', 'seg_majority',
        'class_agreement' (fraction of annotators voting the majority class) and
        'n_class_votes', or an empty dict when nothing is close enough.
    """
    candidates = [
        a for a in annotations if a["img_name"] == img_name and a["channel"] == channel
    ]
    if not candidates:
        return {}

    coords = np.array([a["id_coord"] for a in candidates], dtype=float)
    distances = np.linalg.norm(coords - np.asarray(centroid, dtype=float), axis=1)
    best = int(np.argmin(distances))
    if distances[best] > max_distance:
        return {}

    annotation = candidates[best]
    votes = np.asarray(annotation["class_votes"], dtype=float)
    votes = votes[~np.isnan(votes)]
    majority = annotation["class_majority"]
    agreement = float(np.mean(votes == majority)) if votes.size else np.nan

    return {
        "cluster_id": annotation["cluster_id"],
        "class_majority": majority,
        "seg_majority": annotation["seg_majority"],
        "class_agreement": agreement,
        "n_class_votes": int(votes.size),
    }


def minmax_preprocessing_fn(image):
    """Per-crop min-max normalization to [0, 1], for PhenoMe's `preprocessing_fn`.

    Crops are stored as raw uint16 whose maxima sit far below the 16-bit range,
    while PhenoMe normalizes by the dtype scale. Pass this to
    `pheno.process_images(..., preprocessing_fn=minmax_preprocessing_fn)` to give
    the model and the intensity properties usable contrast.

    Args:
        image: Array of shape (H, W) or (H, W, C).

    Returns:
        float32 array of the same shape with values in [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)

def _scale_image(image, intensity_mode, scale_percentile):
    """Apply `intensity_mode` to a full-resolution image channel before cropping."""
    if intensity_mode in ("raw", "crop_minmax"):
        # 'crop_minmax' is applied per crop in export_object_crops
        return image
    if intensity_mode == "image_percentile":
        scale = np.percentile(image, scale_percentile)
        if scale <= 0:
            scale = max(float(image.max()), 1.0)
        return np.clip(image / scale, 0.0, 1.0).astype(np.float32)
    raise ValueError(
        f"intensity_mode must be 'raw', 'image_percentile' or 'crop_minmax', got: {intensity_mode}"
    )


def export_object_crops(
    dataset,
    output_dir,
    crop_size=80,
    threshold=0.0,
    min_area=9,
    intensity_mode="raw",
    images_subdir="images",
    masks_subdir="masks",
    metadata_filename="metadata.csv",
    attach_annotation_metadata=True,
    scale_percentile=99.9,
    overwrite=False,
):
    """Export every annotated synapse as an individual image crop, mask and metadata row.

    For each original image and image channel, the soft label stack is thresholded,
    connected components are measured per class, and a fixed-size window centered on
    each object is cut from the full-resolution image. The mask written next to it
    contains that object only, since PhenoMe's shape properties use the largest
    connected region of the mask and its masked intensity statistics use `mask > 0.5`.

    Args:
        dataset: A `SemanticProteinSegmentationDataset` instance.
        output_dir: Destination directory; `images_subdir`, `masks_subdir` and the
            metadata csv are created inside it.
        crop_size: Side length of the square crops. 80 matches the annotation window
            used in the source HDF5 and covers ~99% of objects.
        threshold: Mask threshold passed to `find_objects`.
        min_area: Minimum object area in pixels.
        intensity_mode: 'raw' keeps the uint16 values from the HDF5;
            'image_percentile' divides each image channel by its `scale_percentile`
            and clips to [0, 1]; 'crop_minmax' stretches every crop individually.
            The latter two produce float32 crops.
        images_subdir: Sub-directory for the image crops.
        masks_subdir: Sub-directory for the instance masks.
        metadata_filename: Name of the csv written in `output_dir`.
        attach_annotation_metadata: Match each object to the closest annotation from
            `dataset.metadata` and add its vote-derived columns.
        scale_percentile: Percentile used by `intensity_mode='image_percentile'`.
        overwrite: Overwrite existing crop files instead of raising. Same-named files
            are replaced, but crops left over from a previous run with different
            settings are not deleted; clear `output_dir` yourself when re-exporting
            with a different `threshold`, `min_area` or `crop_size`.

    Returns:
        pd.DataFrame: The metadata table that was written, one row per crop. The
        'filename' column holds the extension-less stem shared by the image and its
        mask, which is what `make_dataframe_metadata_fn` matches on by default.
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
            object_index = 0

            for class_index, label_image, region in find_objects(
                entry["label"][channel], threshold=threshold, min_area=min_area
            ):
                window = centered_window(region.centroid, crop_size)
                image_crop = extract_crop(image, window)
                if intensity_mode == "crop_minmax":
                    image_crop = minmax_preprocessing_fn(image_crop)

                instance = (label_image == region.label).astype(np.uint8) * 255
                mask_crop = extract_crop(instance, window)

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
                object_index += 1

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
                    "crop_size": crop_size,
                    "touches_image_border": bool(
                        row0 < 0 or col0 < 0 or row1 > height or col1 > width
                    ),
                }
                if attach_annotation_metadata:
                    row.update(
                        match_annotation(
                            annotations,
                            img_name,
                            channel,
                            region.centroid,
                            max_distance=crop_size / 2,
                        )
                    )
                rows.append(row)

    metadata_df = pd.DataFrame(rows)
    metadata_df.to_csv(os.path.join(output_dir, metadata_filename), index=False)
    return metadata_df
