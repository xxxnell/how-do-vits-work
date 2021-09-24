import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.data.transforms_factory as tff

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


def get_cifar10(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                padding=None,
                scale=None, ratio=None,
                hflip=0.5, vflip=0.0,
                color_jitter=0.0,
                auto_augment=None,
                interpolation='random',
                re_prob=0.0, re_mode='const', re_count=1, re_num_splits=0,
                root="./data", download=False):
    transform_trains = tff.transforms_imagenet_train(
        img_size=32, mean=mean, std=std,
        scale=scale, ratio=ratio,
        hflip=hflip, vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        re_prob=re_prob, re_mode=re_mode, re_count=re_count, re_num_splits=re_num_splits,
        separate=True
    )

    if padding is not None:
        padding = tuple(padding) if isinstance(padding, list) else padding
        tfl = [transforms.RandomCrop(32, padding=padding)]
        tfl.append(transforms.RandomHorizontalFlip(p=hflip)) if hflip > 0 else None
        tfl.append(transforms.RandomVerticalFlip(p=vflip)) if vflip > 0 else None
        transform_trains = (transforms.Compose(tfl), *transform_trains[1:])

    transform_train = transforms.Compose(transform_trains)

    transform_test = tff.transforms_imagenet_eval(
        img_size=32, mean=mean, std=std,
        crop_pct=1.0, interpolation='bilinear',
    )

    dataset_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform_test)

    return dataset_train, dataset_test


def get_cifar100(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
                 padding=None,
                 scale=None, ratio=None,
                 hflip=0.5, vflip=0.0,
                 color_jitter=0.0,
                 auto_augment=None,
                 interpolation='random',
                 re_prob=0.0, re_mode='const', re_count=1, re_num_splits=0,
                 root="./data", download=False):
    transform_trains = tff.transforms_imagenet_train(
        img_size=32, mean=mean, std=std,
        scale=scale, ratio=ratio,
        hflip=hflip, vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        re_prob=re_prob, re_mode=re_mode, re_count=re_count, re_num_splits=re_num_splits,
        separate=True
    )

    if padding is not None:
        padding = tuple(padding) if isinstance(padding, list) else padding
        tfl = [transforms.RandomCrop(32, padding=padding)]
        tfl.append(transforms.RandomHorizontalFlip(p=hflip)) if hflip > 0 else None
        tfl.append(transforms.RandomVerticalFlip(p=vflip)) if vflip > 0 else None
        transform_trains = (transforms.Compose(tfl), *transform_trains[1:])

    transform_train = transforms.Compose(transform_trains)

    transform_test = tff.transforms_imagenet_eval(
        img_size=32, mean=mean, std=std,
        crop_pct=1.0, interpolation='bilinear',
    )

    dataset_train = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
    dataset_test = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_test)

    return dataset_train, dataset_test


def get_imagenet(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 scale=None, ratio=None,
                 hflip=0.5, vflip=0.0,
                 color_jitter=0.0,
                 auto_augment=None,
                 interpolation='random',
                 re_prob=0.0, re_mode='const', re_count=1, re_num_splits=0,
                 root="./data", base_folder='imagenet'):
    transform_train = tff.transforms_imagenet_train(
        img_size=224, mean=mean, std=std,
        scale=scale, ratio=ratio,
        hflip=hflip, vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        re_prob=re_prob, re_mode=re_mode, re_count=re_count, re_num_splits=re_num_splits,
    )

    transform_test = tff.transforms_imagenet_eval(
        img_size=224, mean=mean, std=std,
    )

    train_dir = os.path.join(root, base_folder, 'train')
    test_dir = os.path.join(root, base_folder, 'val')

    dataset_train = datasets.ImageFolder(train_dir, transform_train)
    dataset_test = datasets.ImageFolder(test_dir, transform_test)

    return dataset_train, dataset_test


def subsample(dataset, ratio, random=True):
    """
    Get indices of subsampled dataset with given ratio.
    """
    idxs = list(range(len(dataset)))
    idxs_sorted = {}
    for idx, target in zip(idxs, dataset.targets):
        if target in idxs_sorted:
            idxs_sorted[target].append(idx)
        else:
            idxs_sorted[target] = [idx]

    for idx in idxs_sorted:
        size = len(idxs_sorted[idx])
        lenghts = (int(size * ratio), size - int(size * ratio))
        if random:
            idxs_sorted[idx] = torch.utils.data.random_split(idxs_sorted[idx], lenghts)[0]
        else:
            idxs_sorted[idx] = idxs_sorted[idx][:lenghts[0]]

    idxs = [idx for idxs in idxs_sorted.values() for idx in idxs]
    return idxs


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
                 root="./data", download=True, **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return cifarc.CIFAR10C(root, ctype, intensity, transform=transform, download=download)


def get_cifar100c(ctype, intensity, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
                  root="./data", download=True, **kwargs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return cifarc.CIFAR100C(root, ctype, intensity, transform=transform, download=download)


def get_imagenetc(ctype, intensity, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  root="./data", **kwargs):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return imagenetc.ImageNetC(root, ctype, intensity, transform)


def get_cifar10p(ptype, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                 root="./data", base_folder="cifar-10-p", download=False, **kwargs):
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
