import torch
import torch.nn as nn
import torch.nn.functional as F
import models.layers as layers


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, rate=0.3, **block_kwargs):
        super(BasicBlock, self).__init__()

        self.rate = rate
        self.conv = layers.conv3x3(in_channels, out_channels)
        self.bn = layers.bn(out_channels)
        self.relu = layers.relu()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.dropout(x, p=self.rate)

        return x

    def extra_repr(self):
        return "rate=%.3e" % self.rate
