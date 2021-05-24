import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import ops.cifarc as cifarc
import ops.cifarp as cifarp
import ops.imagenetc as imagenetc


def get_dataset(name, root="./data", download=False, **kwargs):
    if name in ["cifar", "cifar10", "cifar-10"]:
        dataset_train, dataset_test = get_cifar10(root=root, download=download, **kwargs)
    elif name in ["cifar100", "cifar-100"]:
        dataset_train, dataset_test = get_cifar100(root=root, download=download, **kwargs)
    elif name in ["imagenet"]:
        dataset_train, dataset_test = get_imagenet(root=root, **kwargs)
    else:
        raise NotImplementedError
    return dataset_train, dataset_test


def get_cifar10(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), padding=(4, 4),
                root="./data", download=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=tuple(padding)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform_test)

    return dataset_train, dataset_test


def get_cifar100(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), padding=(4, 4),
                 root="./data", download=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=tuple(padding)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset_train = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_test)

    return dataset_train, dataset_test


def get_imagenet(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 root="./data", base_folder='imagenet'):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(root, base_folder, 'train')
    test_dir = os.path.join(root, base_folder, 'val')

    dataset_train = datasets.ImageFolder(train_dir, transform_train)
    dataset_test = datasets.ImageFolder(test_dir, transform_test)

    return dataset_train, dataset_test


def get_corruptions(extra=False):
    corruption_list = [
        "gaussian_noise", "shot_noise", "impulse_noise",  # noise
        "defocus_blur", "motion_blur", "zoom_blur", "glass_blur",  # blur
        "snow", "frost", "fog",  # weather
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",  # digital
    ]
    etc_list = [
        "speckle_noise", "gaussian_blur", "spatter", "saturate",
    ]
    corruption_list = corruption_list + etc_list if extra else corruption_list

    return corruption_list


def get_dataset_c(name, ctype, intensity,
                  root="./data", download=False, **kwargs):
    if name in ["cifar", "cifar10", "cifar-10"]:
        dataset_c = get_cifar10c(ctype, intensity, root=root, download=download, **kwargs)
    elif name in ["cifar100", "cifar-100"]:
        dataset_c = get_cifar100c(ctype, intensity, root=root, download=download, **kwargs)
    elif name in ["imagenet"]:
        dataset_c = get_imagenetc(ctype, intensity, root=root, **kwargs)
    else:
        raise NotImplementedError
    return dataset_c


def get_cifar10c(ctype, intensity, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                 root="./data", download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return cifarc.CIFAR10C(root, ctype, intensity, transform=transform, download=download)


def get_cifar100c(ctype, intensity, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
                  root="./data", download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return cifarc.CIFAR100C(root, ctype, intensity, transform=transform, download=download)


def get_imagenetc(ctype, intensity, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), root="./data"):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return imagenetc.ImageNetC(root, ctype, intensity, transform)


def get_cifar10p(ptype, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                 root="./data", base_folder="cifar-10-p", download=False):
    transform = transforms.Compose([
        cifarp.ToTensor(),
        cifarp.Normalize(mean, std),
    ])

    return cifarp.CIFAR10P(root=root, ptype=ptype, base_folder=base_folder,
                    transform=transform, download=download)


def get_perturbations():
    perturbation_list = [
        "gaussian_noise", "shot_noise", "motion_blur", "zoom_blur",
        "spatter", "brightness", "translate", "rotate", "tilt", "scale",
    ]
    return perturbation_list