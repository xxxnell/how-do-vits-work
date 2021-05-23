import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

import ops.datasets as datasets


class ToTensor:

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        x = x / 255.0
        x = x.transpose(0, 3, 1, 2)
        x = torch.from_numpy(x).float()

        return x


class Normalize:

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, x):
        mean = self.mean.reshape((1, 3, 1, 1))
        std = self.std.reshape((1, 3, 1, 1))
        x = (x - mean) / std

        return x


class CIFAR10P(VisionDataset):
    """
    `CIFAR10P <https://github.com/hendrycks/robustness>`_ Dataset.

    Args:
        root (string): Root directory of dataset.
        ptype (string): Perturbation type.
        base_folder (string):
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = "CIFAR-10-P"
    root = "./data"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-P.tar"
    filename = "cifar-10-p-python.tar"
    tgz_md5 = "125d6775afc5846ea84584d7524dedff"


    def __init__(
            self,
            root: str,
            ptype: str,
            base_folder: str = "cifar-10-p",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        self.base_folder = base_folder

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or perturbated." +
                               " You can use download=True to download it")

        if ptype not in datasets.get_perturbations():
            raise ValueError("Perturbation type %s is not provided. " % ptype +
                             "You must choose one of the following types: " +
                             ", ".join(datasets.get_perturbations()))

        self.data: Any = []
        self.targets = []

        # now load data and target
        fpath = os.path.join(self.root, self.base_folder, "%s.npy" % ptype)
        self.data = np.load(fpath)
        self.targets = np.zeros(self.data.shape[:2]) - 1
        self.targets = self.targets.astype(np.long)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for pname in datasets.get_perturbations():
            fpath = os.path.join(self.root, self.base_folder, "%s.npy" % pname)
            if not check_integrity(fpath):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
