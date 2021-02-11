import torch
import torch.nn as nn
import models.layers as layers


class SigmoidBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=5e-1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(SigmoidBlurBlock, self).__init__()

        self.temp = temp
        self.sigmoid = nn.Sigmoid()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = 4 * self.temp * (self.sigmoid(x / self.temp) - 0.5)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class BoundedSigmoidBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=2e0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(BoundedSigmoidBlurBlock, self).__init__()

        self.temp = temp
        self.sigmoid = nn.Sigmoid()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = 2 * self.sigmoid(x / self.temp) - 1
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class SoftmaxBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(SoftmaxBlurBlock, self).__init__()

        self.temp = temp
        self.softmax = nn.Softmax(dim=1)
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.softmax(x / self.temp)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class ReLuBlurBlock(nn.Module):

    def __init__(self, in_filters, thr=6.0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(ReLuBlurBlock, self).__init__()

        self.thr = thr
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.thr)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "thr=%.3e" % self.thr


class ScalingBlock(nn.Module):

    def __init__(self, temp=4*2e0, **kwargs):
        super(ScalingBlock, self).__init__()

        self.temp = temp

    def forward(self, x):
        x = x / self.temp
        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class ReLuBlock(nn.Module):

    def __init__(self, thr=6.0, **kwargs):
        super(ReLuBlock, self).__init__()

        self.thr = thr

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.thr)

        return x

    def extra_repr(self):
        return "thr=%.3e" % self.thr


class TanhBlock(nn.Module):

    def __init__(self, temp=1e0, **kwargs):
        super(TanhBlock, self).__init__()

        self.temp = temp
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x / self.temp)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class BlurBlock(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(BlurBlock, self).__init__()

        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.blur(x)

        return x
