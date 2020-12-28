import torch
import torch.nn as nn
import models.layers as layers


class BasicBlock(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, **block_kwargs):
        super(BasicBlock, self).__init__()

        self.conv = layers.convnxn(in_channels, channels, kernel_size, stride=stride, padding=padding)
        self.relu = layers.relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x
