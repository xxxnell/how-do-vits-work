import torch
import torch.nn as nn
import models.layers as layers
import models.resnet as resnet
import models.preresnet_dnn_block as preresnet_dnn
import models.preresnet_mcdo_block as preresnet_mcdo
import models.smoothing_block as smoothing
import models.classifier_block as classifier


class PreResNet(resnet.ResNet):

    def __init__(self,
                 block, num_blocks,
                 sblock=smoothing.TanhBlurBlock, num_sblocks=(0, 0, 0, 0),
                 cblock=classifier.BNGAPBlock,
                 num_classes=10, stem=True, name="resnet", **block_kwargs):
        super().__init__(
            block, num_blocks,
            sblock=sblock, num_sblocks=num_sblocks,
            cblock=cblock,
            num_classes=num_classes, stem=stem, name=name, **block_kwargs)
        
        layer0 = []
        if stem:
            layer0.append(layers.convnxn(3, 64, kernel_size=7, stride=2, padding=3))
            layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            layer0.append(layers.conv3x3(3, 64, stride=1))
        self.layer0 = nn.Sequential(*layer0)


# Deterministic

def dnn_18(num_classes=10, stem=True, name="preresnet_dnn_18", **block_kwargs):
    return PreResNet(preresnet_dnn.BasicBlock, [2, 2, 2, 2],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_34(num_classes=10, stem=True, name="preresnet_dnn_34", **block_kwargs):
    return PreResNet(preresnet_dnn.BasicBlock, [3, 4, 6, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_50(num_classes=10, stem=True, name="preresnet_dnn_50", **block_kwargs):
    return PreResNet(preresnet_dnn.Bottleneck, [3, 4, 6, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_101(num_classes=10, stem=True, name="preresnet_dnn_101", **block_kwargs):
    return PreResNet(preresnet_dnn.Bottleneck, [3, 4, 23, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_152(num_classes=10, stem=True, name="preresnet_dnn_152", **block_kwargs):
    return PreResNet(preresnet_dnn.Bottleneck, [3, 8, 36, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout

def mcdo_18(num_classes=10, stem=True, name="preresnet_mcdo_18", **block_kwargs):
    return PreResNet(preresnet_mcdo.BasicBlock, [2, 2, 2, 2],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_34(num_classes=10, stem=True, name="preresnet_mcdo_34", **block_kwargs):
    return PreResNet(preresnet_mcdo.BasicBlock, [3, 4, 6, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_50(num_classes=10, stem=True, name="preresnet_mcdo_50", **block_kwargs):
    return PreResNet(preresnet_mcdo.Bottleneck, [3, 4, 6, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_101(num_classes=10, stem=True, name="preresnet_mcdo_101", **block_kwargs):
    return PreResNet(preresnet_mcdo.Bottleneck, [3, 4, 23, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_152(num_classes=10, stem=True, name="preresnet_mcdo_152", **block_kwargs):
    return PreResNet(preresnet_mcdo.Bottleneck, [3, 8, 36, 3],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# Deterministic + Smoothing

def dnn_smooth_18(num_classes=10, stem=True, name="preresnet_dnn_smoothing_18", **block_kwargs):
    return PreResNet(preresnet_dnn.BasicBlock, [2, 2, 2, 2],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_34(num_classes=10, stem=True, name="preresnet_dnn_smoothing_34", **block_kwargs):
    return PreResNet(preresnet_dnn.BasicBlock, [3, 4, 6, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_50(num_classes=10, stem=True, name="preresnet_dnn_smoothing_50", **block_kwargs):
    return PreResNet(preresnet_dnn.Bottleneck, [3, 4, 6, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_101(num_classes=10, stem=True, name="preresnet_dnn_smoothing_101", **block_kwargs):
    return PreResNet(preresnet_dnn.Bottleneck, [3, 4, 23, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_152(num_classes=10, stem=True, name="preresnet_dnn_smoothing_152", **block_kwargs):
    return PreResNet(preresnet_dnn.Bottleneck, [3, 8, 36, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout + Smoothing

def mcdo_smooth_18(num_classes=10, stem=True, name="preresnet_mcdo_smoothing_18", **block_kwargs):
    return PreResNet(preresnet_mcdo.BasicBlock, [2, 2, 2, 2],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_34(num_classes=10, stem=True, name="preresnet_mcdo_smoothing_34", **block_kwargs):
    return PreResNet(preresnet_mcdo.BasicBlock, [3, 4, 6, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_50(num_classes=10, stem=True, name="preresnet_mcdo_smoothing_50", **block_kwargs):
    return PreResNet(preresnet_mcdo.Bottleneck, [3, 4, 6, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_101(num_classes=10, stem=True, name="preresnet_mcdo_smoothing_101", **block_kwargs):
    return PreResNet(preresnet_mcdo.Bottleneck, [3, 4, 23, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_152(num_classes=10, stem=True, name="preresnet_mcdo_smoothing_152", **block_kwargs):
    return PreResNet(preresnet_mcdo.Bottleneck, [3, 8, 36, 3],
                     num_sblocks=[1, 1, 1, 0],
                     num_classes=num_classes, stem=stem, name=name, **block_kwargs)
