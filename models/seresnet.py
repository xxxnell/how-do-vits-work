import models.resnet as resnet
import models.seresnet_dnn_block as resnet_dnn
import models.seresnet_mcdo_block as resnet_mcdo


# Deterministic

def dnn_18(num_classes=10, stem=True, name="seresnet_dnn_18", **block_kwargs):
    return resnet.ResNet(resnet_dnn.BasicBlock, [2, 2, 2, 2],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_34(num_classes=10, stem=True, name="seresnet_dnn_34", **block_kwargs):
    return resnet.ResNet(resnet_dnn.BasicBlock, [3, 4, 6, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_50(num_classes=10, stem=True, name="seresnet_dnn_50", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_101(num_classes=10, stem=True, name="seresnet_dnn_101", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_152(num_classes=10, stem=True, name="seresnet_dnn_152", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 8, 36, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout

def mcdo_18(num_classes=10, stem=True, name="seresnet_mcdo_18", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.BasicBlock, [2, 2, 2, 2],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_34(num_classes=10, stem=True, name="seresnet_mcdo_34", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.BasicBlock, [3, 4, 6, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_50(num_classes=10, stem=True, name="seresnet_mcdo_50", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_101(num_classes=10, stem=True, name="seresnet_mcdo_101", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_152(num_classes=10, stem=True, name="seresnet_mcdo_152", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 8, 36, 3],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# Deterministic + Smoothing

def dnn_smooth_18(num_classes=10, stem=True, name="seresnet_dnn_smoothing_18", **block_kwargs):
    return resnet.ResNet(resnet_dnn.BasicBlock, [2, 2, 2, 2],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_34(num_classes=10, stem=True, name="seresnet_dnn_smoothing_34", **block_kwargs):
    return resnet.ResNet(resnet_dnn.BasicBlock, [3, 4, 6, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_50(num_classes=10, stem=True, name="seresnet_dnn_smoothing_50", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_101(num_classes=10, stem=True, name="seresnet_dnn_smoothing_101", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_152(num_classes=10, stem=True, name="seresnet_dnn_smoothing_152", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 8, 36, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout + Smoothing

def mcdo_smooth_18(num_classes=10, stem=True, name="seresnet_mcdo_smoothing_18", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.BasicBlock, [2, 2, 2, 2],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_34(num_classes=10, stem=True, name="seresnet_mcdo_smoothing_34", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.BasicBlock, [3, 4, 6, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_50(num_classes=10, stem=True, name="seresnet_mcdo_smoothing_50", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_101(num_classes=10, stem=True, name="seresnet_mcdo_smoothing_101", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_152(num_classes=10, stem=True, name="seresnet_mcdo_smoothing_152", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 8, 36, 3],
                         num_sblocks=[1, 1, 1, 0],
                         num_classes=num_classes, stem=stem, name=name, **block_kwargs)
