import os
import zipfile
from pathlib import Path

import requests


def download_zoo_dataset(path:str, filename:str="zoo_data.hdf5") -> None:
    """ Download the Zooniverse dataset as a .hdf5 file.

    Args:
        path: Path where the data will be downloaded
        filename: Name of the Zooniverse .hdf5 file
    """
    output_path = Path(os.path.join(path, filename))
    if output_path.is_file():
        print('Zoo dataset already exists!')
        return
    url = "https://zenodo.org/records/15480470/files/test_2024-05-16.hdf5?download=1"
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)

def download_actin_dataset(path:str, filename:str="actin_data.zip") -> None:
    """ Download the Actin Conformations dataset as a .hdf5 file.

    Args:
        path: Path where the data will be downloaded
        filename: Name of the Zooniverse .hdf5 file
    """
    output_path = Path(os.path.join(path, filename))
    if output_path.is_file():
        print('Actin dataset already exists!')
        return
    url = "https://zenodo.org/records/15608267/files/factin-conformations-dataset.zip?download=1"
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)

def download_nas_dataset(path:str, filename:str="nas_data.tar") -> None:
    """ Download the NAS dataset as a .tar file.

    Args:
        path: Path where the data will be downloaded
        filename: Name of the NAS .tar file
    """
    output_path = Path(os.path.join(path, filename))
    if output_path.is_file():
        print('NAS dataset already exists!')
        return
    url = "https://s3.valeria.science/flclab-foundation-models/evaluation-data/NeuralActivityStates/NAS_PSD95_test_v2.tar"
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)

def download_optim_dataset(path:str, filename:str="optim_data.zip") -> None:
    """ Download the Optim dataset as a .zip file, and extract it locally.

    Args:
        path: Path where the data will be downloaded
        filename: Name of the Optim .zip file
    """
    zip_path = Path(os.path.join(path, filename))
    if zip_path.is_file():
        print('Optim dataset already exists!')
        return
    url = "https://s3.valeria.science/flclab-foundation-models/evaluation-data/optim-data.zip"
    response = requests.get(url)
    with open(zip_path, "wb") as file:
        file.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)

DATASETS = {
    "Zooniverse":download_zoo_dataset,
    "NAS":download_nas_dataset,
    "Optim":download_optim_dataset,
    "Actin":download_actin_dataset
}

def download_dataset(dataset:str, path:str) -> None:
    """ Download a dataset into the given path.
    """
    print(f"Downloading {dataset} dataset...")
    downloader = DATASETS[dataset]
    downloader(path)
    print("Done!")
