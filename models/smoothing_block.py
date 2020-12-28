import torch
import torch.nn as nn
import models.layers as layers


class SigmoidBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=2e0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(SigmoidBlurBlock, self).__init__()

        self.temp = temp
        self.layer0 = nn.Sigmoid()
        self.layer1 = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = 2 * self.layer0(x / self.temp) - 1
        x = self.layer1(x)

        return x


class SoftmaxBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(SoftmaxBlurBlock, self).__init__()

        self.temp = temp
        self.layer0 = nn.Softmax(dim=1)
        self.layer1 = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.layer0(x / self.temp)
        x = self.layer1(x)

        return x


class ReLuBlurBlock(nn.Module):

    def __init__(self, in_filters, thr=6.0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(ReLuBlurBlock, self).__init__()

        self.thr = thr
        self.layer1 = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.thr)
        x = self.layer1(x)

        return x


class ScalingBlock(nn.Module):

    def __init__(self, temp=2e0, **kwargs):
        super(ScalingBlock, self).__init__()

        self.temp = temp

    def forward(self, x):
        x = x / self.temp
        return x


class ReLuBlock(nn.Module):

    def __init__(self, thr=6.0, **kwargs):
        super(ReLuBlock, self).__init__()

        self.thr = thr

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.thr)

        return x


class TanhBlock(nn.Module):

    def __init__(self, temp=1e0, **kwargs):
        super(TanhBlock, self).__init__()

        self.temp = temp
        self.layer0 = nn.Tanh()

    def forward(self, x):
        x = self.layer0(x / self.temp)

        return x


class BlurBlock(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(BlurBlock, self).__init__()

        self.layer1 = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.layer1(x)

        return x
