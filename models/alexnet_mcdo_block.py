import torch
import torch.nn as nn
import torch.nn.functional as F
import models.layers as layers


class BasicBlock(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0, rate=0.3, **block_kwargs):
        super(BasicBlock, self).__init__()

        self.rate = rate

        self.conv = layers.convnxn(in_channels, channels, kernel_size, stride=stride, padding=padding)
        self.relu = layers.relu()

    def forward(self, x):
        x = F.dropout(x, p=self.rate)
        x = self.conv(x)
        x = self.relu(x)

        return x
