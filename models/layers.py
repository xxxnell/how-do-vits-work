import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Blur(nn.Module):

    def __init__(self, in_filters, filter=(1, 1), pad_mode='replicate', **kwargs):
        super(Blur, self).__init__()

        filter_size = len(filter)
        self.filter_proto = torch.tensor(filter, dtype=torch.float, requires_grad=False)
        self.pad_size = [
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
        ]
        self.pad_mode = pad_mode

        self.filter = torch.tensordot(self.filter_proto, self.filter_proto, dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)
        x = F.conv2d(x, self.filter, groups=x.size()[1])

        return x


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
