"""
This model is based on the implementation of https://github.com/Jongchan/attention-module.
"""
from functools import partial

import torch
import torch.nn as nn

from einops import reduce, rearrange

import models.layers as layers


class ChannelGate(nn.Module):

    def __init__(self, channel, reduction=16, max_pool=True):
        super().__init__()

        self.pools = []
        self.pools.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.pools.append(nn.AdaptiveMaxPool2d((1, 1)))
        self.pools = self.pools if max_pool else self.pools[:1]

        self.ff = nn.Sequential(
            layers.dense(channel, channel // reduction, bias=False),
            layers.relu(),
            layers.dense(channel // reduction, channel, bias=False),
        )
        self.prob = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        s = torch.cat([pool(x) for pool in self.pools], dim=-1)
        s = rearrange(s, "b c n m -> b (n m) c")
        s = self.ff(s)
        s = reduce(s, "b n c -> b c", "mean")
        s = self.prob(s)
        s = s.view(b, c, 1, 1)

        return x * s


class SpatialGate(nn.Module):

    def __init__(self, kernel_size=7, max_pool=True):
        super().__init__()

        self.pools = []
        self.pools.append(partial(torch.mean, dim=1, keepdim=True))
        self.pools.append(lambda x: partial(torch.max, dim=1, keepdim=True)(x)[0])
        self.pools = self.pools if max_pool else self.pools[:1]

        self.ff = nn.Sequential(
            layers.convnxn(len(self.pools), 1, kernel_size=7, stride=1, padding=(kernel_size - 1) // 2),
            layers.bn(1)
        )
        self.prob = nn.Sigmoid()

    def forward(self, x):
        s = torch.cat([pool(x) for pool in self.pools], dim=1)
        s = self.ff(s)
        s = self.prob(s)

        return x * s
