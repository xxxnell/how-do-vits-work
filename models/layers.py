import types
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange


def conv1x1(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups)


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1)


def convnxn(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)


def relu():
    return nn.ReLU()


def relu6(mx=6.0, mn=0.0):
    return Lambda(lambda x: torch.clamp(x, mn, mx))


def bn(dim):
    return nn.BatchNorm2d(dim)


def ln1d(dim):
    return nn.LayerNorm(dim)


def ln2d(dim):
    return nn.Sequential(
        Rearrange("b c h w -> b h w c"),
        nn.LayerNorm(dim),
        Rearrange("b h w c -> b c h w"),
    )


def dense(in_features, out_features, bias=True):
    return nn.Linear(in_features, out_features, bias)


def blur(in_filters, sfilter=(1, 1), pad_mode="constant"):
    if tuple(sfilter) == (1, 1) and pad_mode in ["constant", "zero"]:
        layer = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    else:
        layer = Blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)
    return layer


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class SamePad(nn.Module):

    def __init__(self, filter_size, pad_mode="constant", **kwargs):
        super(SamePad, self).__init__()

        self.pad_size = [
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
        ]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)

        return x

    def extra_repr(self):
        return "pad_size=%s, pad_mode=%s" % (self.pad_size, self.pad_mode)


class Blur(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="replicate", **kwargs):
        super(Blur, self).__init__()

        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)

        self.filter_proto = torch.tensor(sfilter, dtype=torch.float, requires_grad=False)
        self.filter = torch.tensordot(self.filter_proto, self.filter_proto, dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])

        return x

    def extra_repr(self):
        return "pad=%s, filter_proto=%s" % (self.pad, self.filter_proto.tolist())


class Downsample(nn.Module):

    def __init__(self, strides=(2, 2), **kwargs):
        super(Downsample, self).__init__()

        if isinstance(strides, int):
            strides = (strides, strides)
        self.strides = strides

    def forward(self, x):
        shape = (-(-x.size()[2] // self.strides[0]), -(-x.size()[3] // self.strides[1]))
        x = F.interpolate(x, size=shape, mode='nearest')

        return x

    def extra_repr(self):
        return "strides=%s" % repr(self.strides)


class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)


class Lambda(nn.Module):

    def __init__(self, lmd):
        super().__init__()
        if not isinstance(lmd, types.LambdaType):
            raise Exception("`lmd` should be lambda ftn.")
        self.lmd = lmd

    def forward(self, x):
        return self.lmd(x)
