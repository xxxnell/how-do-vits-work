import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

import ops.datasets as datasets


class CIFAR10C(VisionDataset):
    """
    `CIFAR10C <https://github.com/hendrycks/robustness>`_ Dataset.

    Args:
        root (string): Root directory of dataset.
        ctype (string): Corruption type.
        intensity (int): Corruption intensity from 1 to 5
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = "CIFAR-10-C"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    root = "./data"
    filename = "cifar-10-c-python.tar"
    tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"

    def __init__(
            self,
            root: str,
            ctype: str,
            intensity: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10C, self).__init__(root,
                                       transform=transform,
                                       target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." +
                               " You can use download=True to download it")

        if ctype not in datasets.get_corruptions(extra=True):
            raise ValueError("Corruption type %s is not provided. " % ctype +
                             "You must choose one of the following types: " +
                             ", ".join(datasets.get_corruptions(extra=True)))

        self.data: Any = []
        self.targets = []

        # now load data and target
        fpath = os.path.join(self.root, self.base_folder, "%s.npy" % ctype)
        self.data = np.load(fpath)
        fpath = os.path.join(self.root, self.base_folder, "labels.npy")
        self.targets = np.load(fpath)

        self.data = self.data[10000*(intensity-1):10000*intensity]
        self.targets = self.targets[10000*(intensity-1):10000*intensity]
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
        for cname in datasets.get_corruptions(extra=True):
            fpath = os.path.join(self.root, self.base_folder, "%s.npy" % cname)
            if not check_integrity(fpath):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class CIFAR100C(CIFAR10C):
    """
    `CIFAR100C <https://github.com/hendrycks/robustness>`_ Dataset.
    This is a subclass of the `CIFAR10C` Dataset.
    """
    base_folder = "CIFAR-100-C"
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "cifar-100-c-python.tar"
    tgz_md5 = "11f0ed0f1191edbf9fa23466ae6021d3"
