import torch
import torch.nn as nn
import models.vggnet as vgg
import models.classifier_block as classifier
import models.smoothing_block as smoothing
import models.prevggnet_dnn_block as prevggnet_dnn
import models.prevggnet_mcdo_block as prevggnet_mcdo


# Deterministic

def dnn_11(num_classes=10, name="prevgg_dnn_11", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [1, 1, 2, 2, 2],
                      num_classes=num_classes, name=name, **block_kwargs)


def dnn_13(num_classes=10, name="prevgg_dnn_13", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [2, 2, 2, 2, 2],
                      num_classes=num_classes, name=name, **block_kwargs)


def dnn_16(num_classes=10, name="prevgg_dnn_16", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [2, 2, 3, 3, 3],
                      num_classes=num_classes, name=name, **block_kwargs)


def dnn_19(num_classes=10, name="prevgg_dnn_19", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [2, 2, 4, 4, 4],
                      num_classes=num_classes, name=name, **block_kwargs)


# MC dropout

def mcdo_11(num_classes=10, name="prevgg_mcdo_11", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [1, 1, 2, 2, 2],
                      num_classes=num_classes, name=name, **block_kwargs)


def mcdo_13(num_classes=10, name="prevgg_mcdo_13", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [2, 2, 2, 2, 2],
                      num_classes=num_classes, name=name, **block_kwargs)


def mcdo_16(num_classes=10, name="prevgg_mcdo_16", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [2, 2, 3, 3, 3],
                      num_classes=num_classes, name=name, **block_kwargs)


def mcdo_19(num_classes=10, name="prevgg_mcdo_19", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [2, 2, 4, 4, 4],
                      num_classes=num_classes, name=name, **block_kwargs)


# Deterministic + Smoothing

def dnn_smooth_11(num_classes=10, name="prevgg_dnn_smoothing_11", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [1, 1, 2, 2, 2],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


def dnn_smooth_13(num_classes=10, name="prevgg_dnn_smoothing_13", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [2, 2, 2, 2, 2],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


def dnn_smooth_16(num_classes=10, name="prevgg_dnn_smoothing_16", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [2, 2, 3, 3, 3],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


def dnn_smooth_19(num_classes=10, name="prevgg_dnn_smoothing_19", **block_kwargs):
    return vgg.VGGNet(prevggnet_dnn.BasicBlock, [2, 2, 4, 4, 4],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


# MC dropout + Smoothing

def mcdo_smooth_11(num_classes=10, name="prevgg_mcdo_smoothing_11", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [1, 1, 2, 2, 2],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


def mcdo_smooth_13(num_classes=10, name="prevgg_mcdo_smoothing_13", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [2, 2, 2, 2, 2],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


def mcdo_smooth_16(num_classes=10, name="prevgg_mcdo_smoothing_16", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [2, 2, 3, 3, 3],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)


def mcdo_smooth_19(num_classes=10, name="prevgg_mcdo_smoothing_19", **block_kwargs):
    return vgg.VGGNet(prevggnet_mcdo.BasicBlock, [2, 2, 4, 4, 4],
                      num_sblocks=[1, 1, 1, 1, 1],
                      num_classes=num_classes, name=name, **block_kwargs)
