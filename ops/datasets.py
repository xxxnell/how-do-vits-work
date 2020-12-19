import numpy as np
import torchvision
import torchvision.transforms as transforms


def get_dataset(name, mean, std,
                crop_ratio=(0.875, 0.875), root="./data"):
    dataset_train, dataset_test = None, None
    if name in ["cifar", "cifar10", "cifar-10"]:
        pass
    elif name in ["cifar100", "cifar-100"]:
        dataset_train, dataset_test = get_cifar100(mean, std, crop_ratio=crop_ratio, root=root)
    else:
        raise ValueError("The dataset %s is not provided." % name)
    return dataset_train, dataset_test
    

def get_cifar100(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
                 crop_ratio=(0.875, 0.875), root="./data"):
    crop_size = (np.array((32, 32)) * (1 - np.array(crop_ratio))).astype(int)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=tuple(crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    return dataset_train, dataset_test
