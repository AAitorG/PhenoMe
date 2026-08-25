
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Tuple, Any
import tarfile
import io

class SynapticDevelopmentDataset(Dataset):
    def __init__(
        self,
        tarpath: str,
        transform: Optional[Callable] = None,
        n_channels: int = 1,
        num_samples: int = None,
        num_classes: int = 2,
        classes: List = ["DIV5", "DIV14"],
    ) -> None:
        self.tarpath = tarpath
        self.archive_obj = tarfile.open(tarpath, "r")
        self.members = self.get_members()
        self.__cache = {}
        self.classes = classes
        self.transform = transform
        self.n_channels = n_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.names = [m.name for m in self.members]

    def get_members(self):
        members = list(sorted(self.archive_obj.getmembers(), key=lambda m: m.name)) 
        return members

    def get_item_from_archive(self, member):
        buffer = io.BytesIO()
        buffer.write(self.archive_obj.extractfile(member).read())
        buffer.seek(0)
        data = np.load(buffer, allow_pickle=True)
        data = {key : values for key, values in data.items()}
        return data

    def __getsizeof(self, obj: Any) -> int:
        """
        Implements a simple function to estimate the size of an object in memory.

        :param obj: The object to estimate the size of.

        :returns : The size of the object in bytes.
        """
        if isinstance(obj, dict):
            return sum([self.__getsizeof(o) for o in obj.values()])
        elif isinstance(obj, (list, tuple)):
            return sum([self.__getsizeof(o) for o in obj])
        elif isinstance(obj, str):
            return len(obj)
        else:
            return obj.size * obj.dtype.itemsize

    def __len__(self) -> int:
        return len(self.members) 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.get_item_from_archive(self.members[idx])
        image = data["image"]
       # metadata = data["metadata"].item()
        # print(metadata)
        dpi = str(data["dpi"]) 
        div = str(data["div"])
        batch = "26.2"
        condition = str(data["condition"])
        min_value = data["min_value"]
        max_value = data["max_value"]

        img = torch.tensor(image[np.newaxis, ...], dtype=torch.float32)
        img = self.transform(img) if self.transform is not None else img
        return img, {"label": div, "dataset-idx": idx, "batch": batch, "dpi": dpi, "protein": "PSD95", "min_value": min_value, "max_value": max_value}


def main():
    dataset = SynapticDevelopmentDataset(tarpath="path/to/PLKO-262-PSD95-test.tar")
    for img, metadata in dataset[0]:
        print(img.shape)
        div = metadata["label"]


if __name__ == "__main__":
    main()
