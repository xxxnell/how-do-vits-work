import torch
import torch.nn as nn

import models.layers as layers
import models.gates as gates

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, groups=1, width_per_group=64, reduction=16, **block_kwargs):
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = layers.conv3x3(in_channels, width, stride=stride)
        self.bn1 = layers.bn(width)
        self.relu1 = layers.relu()

        self.conv2 = layers.conv3x3(width, channels * self.expansion)
        self.bn2 = layers.bn(channels * self.expansion)
        self.relu2 = layers.relu()

        self.gate = gates.ChannelGate(channels * self.expansion, reduction, max_pool=False)

    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gate(x)

        x = skip + x
        x = self.relu2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, groups=1, width_per_group=64, reduction=16, **block_kwargs):
        super(Bottleneck, self).__init__()

        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(layers.conv1x1(in_channels, channels * self.expansion, stride=stride))
            self.shortcut.append(layers.bn(channels * self.expansion))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.conv1 = layers.conv1x1(in_channels, width)
        self.bn1 = layers.bn(width)
        self.relu1 = layers.relu()

        self.conv2 = layers.conv3x3(width, width, stride=stride, groups=groups)
        self.bn2 = layers.bn(width)
        self.relu2 = layers.relu()

        self.conv3 = layers.conv1x1(width, channels * self.expansion)
        self.bn3 = layers.bn(channels * self.expansion)
        self.relu3 = layers.relu()

        self.gate = gates.ChannelGate(channels * self.expansion, reduction, max_pool=False)

    def forward(self, x):
        skip = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gate(x)

        x = skip + x
        x = self.relu3(x)

        return x
