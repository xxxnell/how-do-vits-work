import torch
import torch.nn as nn
import models.layers as layers

    
class TanhBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(TanhBlurBlock, self).__init__()

        self.temp = temp
        self.relu = layers.relu()
        self.tanh = nn.Tanh()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class BNTanhBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(BNTanhBlurBlock, self).__init__()

        self.bn = layers.bn(in_filters)
        self.temp = temp
        self.relu = layers.relu()
        self.tanh = nn.Tanh()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.bn(x)
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp

    
class TanhBlock(nn.Module):

    def __init__(self, temp=1e1, **kwargs):
        super(TanhBlock, self).__init__()

        self.temp = temp
        self.relu = layers.relu()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class SigmoidBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(SigmoidBlurBlock, self).__init__()

        self.temp = temp
        self.relu = layers.relu()
        self.sigmoid = nn.Sigmoid()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = 4 * self.temp * (self.sigmoid(x / self.temp) - 0.5)
        x = self.relu(x)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class SoftmaxBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(SoftmaxBlurBlock, self).__init__()

        self.temp = temp
        self.relu = layers.relu()
        self.softmax = nn.Softmax(dim=1)
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.softmax(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class ReLuBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=6.0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(ReLuBlurBlock, self).__init__()

        self.temp = temp
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = torch.clamp(x, 0.0, self.temp)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "thr=%.3e" % self.thr

    
class ScalingBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=5e0, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(ScalingBlurBlock, self).__init__()

        self.temp = temp
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = x / self.temp
        x = self.blur(x)
        
        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp


class ScalingBlock(nn.Module):

    def __init__(self, temp=5e0, **kwargs):
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

    
class BNBlurBlock(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(PreactBlurBlock, self).__init__()

        self.bn = layers.bn(in_filters)
        self.relu = layers.relu()
        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.blur(x)

        return x


class BlurBlock(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(BlurBlock, self).__init__()

        self.blur = layers.blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.blur(x)

        return x
