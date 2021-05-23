import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import torchvision.transforms as transforms
import torchvision.datasets as tdatasets

import ops.datasets as datasets


class ImageNetC(tdatasets.ImageFolder):
    """
    `ImageNet-C <https://github.com/hendrycks/robustness>`_ Dataset.

    Args:
        root (string): Root directory of dataset.
        ctype (string): Corruption type.
        intensity (int): Corruption intensity from 1 to 5
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = "imagenetc"

    def __init__(
            self,
            root: str,
            ctype: str,
            intensity: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if ctype not in datasets.get_corruptions(extra=True):
            raise ValueError("Corruption type %s is not provided. " % ctype +
                             "You must choose one of the following types: " +
                             ", ".join(datasets.get_corruptions(extra=True)))

        path = os.path.join(root, self.base_folder, ctype, str(intensity))

        super().__init__(path,
                         transform=transform,
                         target_transform=target_transform)
