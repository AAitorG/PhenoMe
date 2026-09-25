import io
import os
import tarfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class NASProteinsDataset(Dataset):
    def __init__(
        self,
        archive_path: str, 
        transform: Optional[Callable] = None,
        conditions: Optional[List[str]] = None,
        channel: Optional[int] = 1, 
        proteins: Optional[List[str]] = "PSD95-Basson",
        **kwargs
    ) -> None:
        super(NASProteinsDataset, self).__init__()
        self.archive_path = archive_path
        self.transform = transform
        self.conditions = conditions
        self.proteins = proteins
        self.channel = channel

        self.data = []
        with tarfile.open(self.archive_path, "r") as handle:
            for member in tqdm(handle.getmembers(), desc="... Loading dataset from archive ..."):
                buffer = io.BytesIO() 
                buffer.write(handle.extractfile(member).read())
                buffer.seek(0)
                curr = np.load(buffer, allow_pickle=True)
                curr = {key: values for key, values in curr.items()}

                if self.conditions is not None and str(curr["condition"]) not in self.conditions:
                    continue 
                if self.proteins is not None and str(curr["proteins"]) not in self.proteins:
                    continue 

                self.data.append(curr)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        data = self.data[idx]

        img = data["image"]
        spots = data["spots"]
        foreground = data["foreground"]
        condition = data["condition"]
        proteins = data["proteins"]

        if self.channel is not None:
            image = torch.tensor(img[self.channel][np.newaxis, ...], dtype=torch.float32)
            spots = spots[self.channel]

        else:
            image = torch.tensor(img, dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        metadata = {
            "condition": str(condition),
            "spots": spots,
            "foreground": foreground,
            "proteins": str(proteins),
            "coords": np.array(data["coords"]),
            "fname": str(data["fname"]),
        }
        return image, metadata
