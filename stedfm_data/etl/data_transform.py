import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from stedfm.datasets import NeuralActivityStates, OptimDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_handling import (
    ActinDataset,
    NASProteinsDataset,
    SemanticProteinSegmentationDataset,
    export_cluster_crops,
    mask_channel_ratios,
)


def transform_zoo_dataset(input_path:str, output_path:str, filename:str="zoo_data.hdf5") -> None:
    """ Structures the Zooniverse dataset for PhenoMe usage

    Args:
        input_path: Directory containing the .hdf5 file of the Zooniverse data.
        output_path: Directory where the restructured data will be saved.
        filename: Name of the Zooniverse .hdf5 file.
    """
    metadata_path = Path(os.path.join(output_path, 'metadata.csv'))
    if metadata_path.is_file():
        print('Zoo dataset already transformed!')
        return

    # Use the provided data loader to extract metadata
    dataset = SemanticProteinSegmentationDataset(
        h5file=os.path.join(input_path, filename),
        n_channels=1,
        data_aug=0,
        validation=True,
        image_size=1024,
        return_metadata=True,
    )

    # Create and save crops of individual synaptic protein clusters
    export_cluster_crops(
        dataset=dataset,
        output_dir=output_path,
        min_distance=8,
        threshold=0.3,
        min_area=128,
        intensity_mode="image_percentile",
        bbox_padding=2,
        pad_to_size=None
    )

def transform_nas_dataset(input_path:str, output_path:str, filename:str="nas_data.tar") -> None:
    """ Structures the NAS dataset for PhenoMe usage

    Args:
        input_path: Directory containing the .tar file of the NAS data.
        output_path: Directory where the restructured data will be saved.
        filename: Name of the NAS .tar file.
    """
    metadata_path = Path(os.path.join(output_path, 'metadata.csv'))
    if metadata_path.is_file():
        print('NAS dataset already transformed!')
        return

    tar_path = os.path.join(input_path, filename)
    dataset = NeuralActivityStates(tarpath=tar_path, balance=False)
    images_dir = os.path.join(output_path, "images")
    masks_dir = os.path.join(output_path, "masks")
    metadata_path = os.path.join(output_path, "metadata.csv")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    rows = []
    for image, meta in tqdm(dataset, desc="Saving tiles"):
        stem = (
            f"img_{meta.get('dataset-idx')}"
        )

        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]  # (1, H, W) -> (H, W)
        mask = np.asarray(meta.get('mask'), dtype=np.float32)  # (n_classes, H, W)

        tifffile.imwrite(os.path.join(images_dir, f"{stem}.tif"), image)
        tifffile.imwrite(os.path.join(masks_dir, f"{stem}.tif"), mask)
        row_dict = {k: meta.get(k) for k in meta if k != 'mask'}
        row_dict["filename"] = stem
        rows.append(row_dict)

    # 'filename' is the extension-less stem shared by the image and its mask, which is what
    # make_dataframe_metadata_fn matches on in pheno.find_files().
    tiles_metadata_df = pd.DataFrame(rows)
    tiles_metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved {len(tiles_metadata_df)} tiles to {images_dir} and {masks_dir}")
    print(f"Metadata written to {metadata_path}")

def transform_nas_v2_dataset(input_path:str, output_path:str, filename:str="NAS_v2_test.tar") -> None:
    """Structures the NASv2 dataset for PhenoMe usage, which can be used for any combination of proteins
    (instead of PSD95 only). Here it's for Bassoon specifically.

    Args:
        input_path: Directory containing the .tar file of the NAS data.
        output_path: Directory where the restructured data will be saved.
        filename: Name of the NAS .tar file.
    """
    metadata_path = Path(os.path.join(output_path, 'metadata.csv'))
    if metadata_path.is_file():
        print('NAS dataset already transformed!')
        return

    tar_path = os.path.join(input_path, filename)
    dataset = NASProteinsDataset(archive_path=tar_path,
                                channel = 0, # 0 for bassoon
                                conditions = ["Block", "0MgGlyBic", "GluGly", "48hTTX"],
                                proteins = "PSD95-Basson",)
    images_dir = os.path.join(output_path, "images")
    masks_dir = os.path.join(output_path, "masks")
    metadata_path = os.path.join(output_path, "metadata.csv")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    rows = []
    for i, (image, meta) in enumerate(tqdm(dataset, desc="Saving tiles")):
        stem = (
            f"img_{i}"
        )

        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]  # (1, H, W) -> (H, W)
        mask = np.asarray(meta.get('spots'), dtype=np.float32)  # (n_classes, H, W)

        tifffile.imwrite(os.path.join(images_dir, f"{stem}.tif"), image)
        tifffile.imwrite(os.path.join(masks_dir, f"{stem}.tif"), mask)
        row_dict = {k: meta.get(k) for k in meta if k not in ["spots", "foreground"]}
        row_dict["filename"] = stem
        rows.append(row_dict)

    # 'filename' is the extension-less stem shared by the image and its mask, which is what
    # make_dataframe_metadata_fn matches on in pheno.find_files().
    tiles_metadata_df = pd.DataFrame(rows)
    tiles_metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved {len(tiles_metadata_df)} tiles to {images_dir} and {masks_dir}")
    print(f"Metadata written to {metadata_path}")

def transform_optim_dataset(input_path:str, output_path:str) -> None:
    """ Structures the Optim dataset for PhenoMe usage

    Args:
        input_path: Directory containing the extracted Optim data.
        output_path: Directory where the restructured data will be saved.
    """
    metadata_path = Path(os.path.join(output_path, 'metadata.csv'))
    if metadata_path.is_file():
        print('Optim dataset already transformed!')
        return

    print(os.path.join(input_path, "optim-data"))
    dataset = OptimDataset(data_folder=os.path.join(input_path, "optim-data"),
                           num_samples={'actin': None, 'tubulin': None, 'CaMKII_Neuron': None, "PSD95_Neuron": None},
                           apply_filter=True,
                           classes=['actin', 'tubulin', 'CaMKII_Neuron', 'PSD95_Neuron'])

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=6)

    images_dir = os.path.join(output_path, "images")
    metadata_path = os.path.join(output_path, "metadata.csv")

    tiles_dir = output_path
    tiles_images_dir = os.path.join(tiles_dir, "images")
    tiles_metadata_path = os.path.join(tiles_dir, "metadata.csv")

    labels = ['actin', 'tubulin', 'CaMKII', 'PSD95']

    os.makedirs(tiles_images_dir, exist_ok=True)

    rows = []
    for idx, (image, meta) in enumerate(tqdm(dataloader, desc="Saving tiles")):
        stem = (
            f"img_{idx}"
        )

        image = np.asarray(image, dtype=np.float32)
        while image.ndim > 2:
            image = image[0]

        tifffile.imwrite(os.path.join(images_dir, f"{stem}.tif"), image)
        row_dict = {k: meta.get(k).item() for k in meta}
        row_dict["label"] = labels[row_dict["label"]]
        row_dict["filename"] = stem
        rows.append(row_dict)

    # 'filename' is the extension-less stem shared by the image and its mask, which is what
    # make_dataframe_metadata_fn matches on in pheno.find_files().
    tiles_metadata_df = pd.DataFrame(rows)
    tiles_metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved {len(tiles_metadata_df)} tiles to {tiles_images_dir}")
    print(f"Metadata written to {tiles_metadata_path}")

def transform_actin_dataset(input_path:str, output_path:str, filename:str="actin_data.zip") -> None:
    """ Structures the Actin Conformations dataset for PhenoMe usage

    Images and masks are cut into 224x224 non-overlapping tiles. The metadata holds, for
    each tile, the proportion of pixels covered by each mask channel.

    Args:
        input_path: Directory containing the .zip file of the Actin data.
        output_path: Directory where the restructured data will be saved.
        filename: Name of the Actin .zip file.
    """
    metadata_path = Path(os.path.join(output_path, 'metadata.csv'))
    if metadata_path.is_file():
        print('Actin dataset already transformed!')
        return

    zip_path = os.path.join(input_path, filename)
    dataset = ActinDataset(archive_path=zip_path)
    images_dir = os.path.join(output_path, "images")
    masks_dir = os.path.join(output_path, "masks")
    metadata_path = os.path.join(output_path, "metadata.csv")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    rows = []
    for i, (image, meta) in enumerate(tqdm(dataset, desc="Saving tiles")):
        stem = (
            f"img_{i}"
        )

        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(meta.get('mask'), dtype=np.float32)  # (n_classes, H, W)

        tifffile.imwrite(os.path.join(images_dir, f"{stem}.tif"), image)
        tifffile.imwrite(os.path.join(masks_dir, f"{stem}.tif"), mask)
        row_dict = {k: meta.get(k) for k in meta if k != 'mask'}
        row_dict["filename"] = stem
        row_dict.update(mask_channel_ratios(mask))
        rows.append(row_dict)

    # 'filename' is the extension-less stem shared by the image and its mask, which is what
    # make_dataframe_metadata_fn matches on in pheno.find_files().
    tiles_metadata_df = pd.DataFrame(rows)
    tiles_metadata_df.to_csv(metadata_path, index=False)

    print(f"Saved {len(tiles_metadata_df)} tiles to {images_dir} and {masks_dir}")
    print(f"Metadata written to {metadata_path}")

DATASETS = {
    "Zooniverse":transform_zoo_dataset,
    "NAS":transform_nas_dataset,
    "NASv2":transform_nas_v2_dataset,
    "Optim":transform_optim_dataset,
    "Actin":transform_actin_dataset,
}

def transform_dataset(dataset:str, input_path:str, output_path:str) -> None:
    """ Download a dataset into the given path.
    """
    print(f"Transforming {dataset} dataset...")
    transformer = DATASETS[dataset]
    transformer(input_path, output_path)
    print("Done!")
