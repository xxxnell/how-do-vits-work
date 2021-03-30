import torch
import torch.nn as nn
import models.classifier_block as classifier
import models.smoothing_block as smoothing
import models.vggnet_dnn_block as vggnet_dnn
import models.vggnet_mcdo_block as vggnet_mcdo


class VGGNet(nn.Module):

    def __init__(self, block, num_blocks,
                 sblock=smoothing.TanhBlurBlock, num_sblocks=(0, 0, 0, 0, 0),
                 cblock=classifier.MLPBlock,
                 num_classes=10, name="vgg", **block_kwargs):
        super(VGGNet, self).__init__()

        self.name = name

        self.layer0 = self._make_layer(block, 3, 64, num_blocks[0], pool=False, **block_kwargs)
        self.layer1 = self._make_layer(block, 64, 128, num_blocks[1], pool=True, **block_kwargs)
        self.layer2 = self._make_layer(block, 128, 256, num_blocks[2], pool=True, **block_kwargs)
        self.layer3 = self._make_layer(block, 256, 512, num_blocks[3], pool=True, **block_kwargs)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[4], pool=True, **block_kwargs)

        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0], **block_kwargs)
        self.smooth1 = self._make_smooth_layer(sblock, 128, num_sblocks[1], **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2], **block_kwargs)
        self.smooth3 = self._make_smooth_layer(sblock, 512, num_sblocks[3], **block_kwargs)
        self.smooth4 = self._make_smooth_layer(sblock, 512, num_sblocks[4], **block_kwargs)

        self.classifier = []
        if cblock is classifier.MLPBlock:
            self.classifier.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, pool, **block_kwargs):
        layers, channels = [], in_channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(num_blocks):
            layers.append(block(channels, out_channels, **block_kwargs))
            channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks, **block_kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.smooth0(x)

        x = self.layer1(x)
        x = self.smooth1(x)

        x = self.layer2(x)
        x = self.smooth2(x)

        x = self.layer3(x)
        x = self.smooth3(x)

        x = self.layer4(x)
        x = self.smooth4(x)

        x = self.classifier(x)

        return x


# Deterministic

def dnn_11(num_classes=10, name="vgg_dnn_11", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [1, 1, 2, 2, 2],
                  num_classes=num_classes, name=name, **block_kwargs)


def dnn_13(num_classes=10, name="vgg_dnn_13", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [2, 2, 2, 2, 2],
                  num_classes=num_classes, name=name, **block_kwargs)


def dnn_16(num_classes=10, name="vgg_dnn_16", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [2, 2, 3, 3, 3],
                  num_classes=num_classes, name=name, **block_kwargs)


def dnn_19(num_classes=10, name="vgg_dnn_19", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [2, 2, 4, 4, 4],
                  num_classes=num_classes, name=name, **block_kwargs)


# MC dropout

def mcdo_11(num_classes=10, name="vgg_mcdo_11", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [1, 1, 2, 2, 2],
                  num_classes=num_classes, name=name, **block_kwargs)


def mcdo_13(num_classes=10, name="vgg_mcdo_13", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [2, 2, 2, 2, 2],
                  num_classes=num_classes, name=name, **block_kwargs)


def mcdo_16(num_classes=10, name="vgg_mcdo_16", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [2, 2, 3, 3, 3],
                  num_classes=num_classes, name=name, **block_kwargs)


def mcdo_19(num_classes=10, name="vgg_mcdo_19", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [2, 2, 4, 4, 4],
                  num_classes=num_classes, name=name, **block_kwargs)


# Deterministic + Smoothing

def dnn_smooth_11(num_classes=10, name="vgg_dnn_smoothing_11", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [1, 1, 2, 2, 2],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


def dnn_smooth_13(num_classes=10, name="vgg_dnn_smoothing_13", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [2, 2, 2, 2, 2],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


def dnn_smooth_16(num_classes=10, name="vgg_dnn_smoothing_16", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [2, 2, 3, 3, 3],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


def dnn_smooth_19(num_classes=10, name="vgg_dnn_smoothing_19", **block_kwargs):
    return VGGNet(vggnet_dnn.BasicBlock, [2, 2, 4, 4, 4],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


# MC dropout + Smoothing

def mcdo_smooth_11(num_classes=10, name="vgg_mcdo_smoothing_11", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [1, 1, 2, 2, 2],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


def mcdo_smooth_13(num_classes=10, name="vgg_mcdo_smoothing_13", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [2, 2, 2, 2, 2],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


def mcdo_smooth_16(num_classes=10, name="vgg_mcdo_smoothing_16", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [2, 2, 3, 3, 3],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)


def mcdo_smooth_19(num_classes=10, name="vgg_mcdo_smoothing_19", **block_kwargs):
    return VGGNet(vggnet_mcdo.BasicBlock, [2, 2, 4, 4, 4],
                  num_sblocks=[1, 1, 1, 1, 1],
                  num_classes=num_classes, name=name, **block_kwargs)
