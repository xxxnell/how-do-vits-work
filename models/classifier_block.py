import torch
import torch.nn as nn
import torch.nn.functional as F
import models.layers as layers


class GAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GAPBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x

    
class BNGAPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(BNGAPBlock, self).__init__()

        self.bn = layers.bn(in_features)
        self.relu = layers.relu()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class MLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(MLPBlock, self).__init__()

        self.dense1 = layers.dense(in_features, 4096)
        self.relu1 = layers.relu()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = layers.dense(4096, 4096)
        self.relu2 = layers.relu()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = layers.dense(4096, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x


class GMaxPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GMaxPBlock, self).__init__()

        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.gmp(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class GMedPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GMedPBlock, self).__init__()

        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], -1)
        x = torch.topk(x, k=int(x.size()[2] / 2), dim=2)[0][:, :, -1]
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class GAPClipBlock(nn.Module):

    def __init__(self, in_features, num_classes, temp=2e0, **kwargs):
        super(GAPClipBlock, self).__init__()

        self.temp = temp
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = layers.dense(in_features, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = self.temp * (F.sigmoid(x / self.temp) - 0.5)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)

        return x


class GAPMLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(GAPMLPBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = layers.dense(in_features, 4096)
        self.relu1 = layers.relu()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = layers.dense(4096, 4096)
        self.relu2 = layers.relu()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = layers.dense(4096, num_classes)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x
