import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **block_kwargs):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
