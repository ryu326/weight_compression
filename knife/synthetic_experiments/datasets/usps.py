import os
import pickle
from typing import Any, Callable, Optional, Tuple, cast

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url


class USPS(VisionDataset):
    """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.

    Args:
        root (string): Root directory of dataset to store``USPS`` data files.
        train (bool, optional): If True, creates dataset from ``usps.bz2``,
            otherwise from ``usps.t.bz2``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2",
            "ec16c51db3855ca6c91edd34d0e9b197",
        ],
        "test": [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2",
            "8ea070ee2aca1ac39742fdd1ef5ed118",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(USPS, self).__init__(root, transform=transform, target_transform=target_transform)
        if split == "test":
            url, filename, checksum = self.split_list["test"]
        else:
            url, filename, checksum = self.split_list["train"]
        full_path = os.path.join(self.root, filename)

        if download and not os.path.exists(full_path):
            download_url(url, self.root, filename, md5=checksum)

        import bz2

        with bz2.open(full_path) as fp:
            raw_data = [line.decode().split() for line in fp.readlines()]
            tmp_list = [[x.split(":")[-1] for x in data[1:]] for data in raw_data]
            imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
            imgs = ((cast(np.ndarray, imgs) + 1) / 2 * 255).astype(dtype=np.uint8)
            classes = [int(d[0]) - 1 for d in raw_data]

        n = 5104
        if split == "train":
            self.data = imgs[:n]
            self.targets = classes[:n]
        elif split == "valid":
            self.data = imgs[n:]
            self.targets = classes[n:]
        else:
            self.data = imgs
            self.targets = classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img, mode='L')
        img = np.repeat(img[:, :, None], 3, axis=2)
        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
