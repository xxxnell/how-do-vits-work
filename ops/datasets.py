import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import ops.cifarc as cifarc


def get_dataset(name, mean, std,
                crop_ratio=(0.875, 0.875), root="./data", download=False):
    dataset_train, dataset_test = None, None
    if name in ["cifar", "cifar10", "cifar-10"]:
        dataset_train, dataset_test = get_cifar10(mean, std, crop_ratio=crop_ratio, root=root, download=download)
    elif name in ["cifar100", "cifar-100"]:
        dataset_train, dataset_test = get_cifar100(mean, std, crop_ratio=crop_ratio, root=root, download=download)
    else:
        raise NotImplementedError
    return dataset_train, dataset_test


def get_cifar10(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                crop_ratio=(0.875, 0.875), root="./data", download=False):
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

    dataset_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform_test)

    return dataset_train, dataset_test


def get_cifar100(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
                 crop_ratio=(0.875, 0.875), root="./data", download=False):
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

    dataset_train = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_test)

    return dataset_train, dataset_test


def get_corruptions():
    return cifarc.CIFAR10C.corruption_list


def get_dataset_c(name, ctype, intensity, mean, std,
                  root="./data", download=False):
    if name in ["cifar", "cifar10", "cifar-10"]:
        dataset_c = get_cifar10c(ctype, intensity, mean, std, root=root, download=download)
    elif name in ["cifar100", "cifar-100"]:
        dataset_c = get_cifar100c(ctype, intensity, mean, std, root=root, download=download)
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

