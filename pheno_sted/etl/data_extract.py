import os
import zipfile
from pathlib import Path

import requests

"""
Class to handle the different STED datasets used with STED-FM
"""

def download_zoo_dataset(path:str, filename:str="zoo_data.hdf5") -> None:
    """ Download the Zooniverse dataset
    """
    output_path = Path(os.path.join(path, filename))
    if output_path.is_file():
        print('Zoo dataset already exists!')
        return
    url = "https://zenodo.org/records/15480470/files/test_2024-05-16.hdf5?download=1"
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)

def download_nas_dataset(path:str, filename:str="nas_data.tar") -> None:
    """ Download the NAS dataset
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
    """ Download the Optim dataset
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
    "Optim":download_optim_dataset
}

def download_dataset(dataset:str, path:str) -> None:
    """ Download a dataset into the given path.
    """
    print(f"Downloading {dataset} dataset...")
    downloader = DATASETS[dataset]
    downloader(path)
    print("Done!")