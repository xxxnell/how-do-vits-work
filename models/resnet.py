import torch
import torch.nn as nn
import models.layers as layers
import models.resnet_dnn_block as resnet_dnn
import models.resnet_mcdo_block as resnet_mcdo
import models.smoothing_block as smoothing
import models.classifier_block as classifier


class ResNet(nn.Module):

    def __init__(self,
                 block, num_blocks,
                 sblock=smoothing.TanhBlurBlock, num_sblocks=(0, 0, 0, 0),
                 cblock=classifier.GAPBlock,
                 sd=0.0, num_classes=10, stem=True, name="resnet", **block_kwargs):
        super().__init__()
        self.name = name
        idxs = [[j for j in range(sum(num_blocks[:i]), sum(num_blocks[:i + 1]))] for i in range(len(num_blocks))]
        sds = [[sd * j / (sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = []
        if stem:
            self.layer0.append(layers.convnxn(3, 64, kernel_size=7, stride=2, padding=3))
            self.layer0.append(layers.bn(64))
            self.layer0.append(layers.relu())
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(layers.conv3x3(3, 64, stride=1))
            self.layer0.append(layers.bn(64))
            self.layer0.append(layers.relu())
        self.layer0 = nn.Sequential(*self.layer0)

        self.layer1 = self._make_layer(block, 64, 64,
                                       num_blocks[0], stride=1, sds=sds[0], **block_kwargs)
        self.layer2 = self._make_layer(block, 64 * block.expansion, 128,
                                       num_blocks[1], stride=2, sds=sds[1], **block_kwargs)
        self.layer3 = self._make_layer(block, 128 * block.expansion, 256,
                                       num_blocks[2], stride=2, sds=sds[2], **block_kwargs)
        self.layer4 = self._make_layer(block, 256 * block.expansion, 512,
                                       num_blocks[3], stride=2, sds=sds[3], **block_kwargs)

        self.smooth1 = self._make_smooth_layer(sblock, 64 * block.expansion,
                                               num_sblocks[0], **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 128 * block.expansion,
                                               num_sblocks[1], **block_kwargs)
        self.smooth3 = self._make_smooth_layer(sblock, 256 * block.expansion,
                                               num_sblocks[2], **block_kwargs)
        self.smooth4 = self._make_smooth_layer(sblock, 512 * block.expansion,
                                               num_sblocks[3], **block_kwargs)

        self.classifier = []
        if cblock is classifier.MLPBlock:
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * 512 * block.expansion, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(512 * block.expansion, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, stride, sds, **block_kwargs):
        stride_seq = [stride] + [1] * (num_blocks - 1)
        layer_seq, channels = [], in_channels
        for i in range(num_blocks):
            layer_seq.append(block(channels, out_channels, stride=stride_seq[i], sd=sds[i], **block_kwargs))
            channels = out_channels * block.expansion
        return nn.Sequential(*layer_seq)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks, **block_kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)

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

def dnn_18(num_classes=10, stem=True, name="resnet_dnn_18", **block_kwargs):
    return ResNet(resnet_dnn.BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_34(num_classes=10, stem=True, name="resnet_dnn_34", **block_kwargs):
    return ResNet(resnet_dnn.BasicBlock, [3, 4, 6, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_50(num_classes=10, stem=True, name="resnet_dnn_50", **block_kwargs):
    return ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_101(num_classes=10, stem=True, name="resnet_dnn_101", **block_kwargs):
    return ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_152(num_classes=10, stem=True, name="resnet_dnn_152", **block_kwargs):
    return ResNet(resnet_dnn.Bottleneck, [3, 8, 36, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout

def mcdo_18(num_classes=10, stem=True, name="resnet_mcdo_18", **block_kwargs):
    return ResNet(resnet_mcdo.BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_34(num_classes=10, stem=True, name="resnet_mcdo_34", **block_kwargs):
    return ResNet(resnet_mcdo.BasicBlock, [3, 4, 6, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_50(num_classes=10, stem=True, name="resnet_mcdo_50", **block_kwargs):
    return ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_101(num_classes=10, stem=True, name="resnet_mcdo_101", **block_kwargs):
    return ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_152(num_classes=10, stem=True, name="resnet_mcdo_152", **block_kwargs):
    return ResNet(resnet_mcdo.Bottleneck, [3, 8, 36, 3],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# Deterministic + Smoothing

def dnn_smooth_18(num_classes=10, stem=True, name="resnet_dnn_smoothing_18", **block_kwargs):
    return ResNet(resnet_dnn.BasicBlock, [2, 2, 2, 2],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_34(num_classes=10, stem=True, name="resnet_dnn_smoothing_34", **block_kwargs):
    return ResNet(resnet_dnn.BasicBlock, [3, 4, 6, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_50(num_classes=10, stem=True, name="resnet_dnn_smoothing_50", **block_kwargs):
    return ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_101(num_classes=10, stem=True, name="resnet_dnn_smoothing_101", **block_kwargs):
    return ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth_152(num_classes=10, stem=True, name="resnet_dnn_smoothing_152", **block_kwargs):
    return ResNet(resnet_dnn.Bottleneck, [3, 8, 36, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


# MC dropout + Smoothing

def mcdo_smooth_18(num_classes=10, stem=True, name="resnet_mcdo_smoothing_18", **block_kwargs):
    return ResNet(resnet_mcdo.BasicBlock, [2, 2, 2, 2],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_34(num_classes=10, stem=True, name="resnet_mcdo_smoothing_34", **block_kwargs):
    return ResNet(resnet_mcdo.BasicBlock, [3, 4, 6, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_50(num_classes=10, stem=True, name="resnet_mcdo_smoothing_50", **block_kwargs):
    return ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_101(num_classes=10, stem=True, name="resnet_mcdo_smoothing_101", **block_kwargs):
    return ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth_152(num_classes=10, stem=True, name="resnet_mcdo_smoothing_152", **block_kwargs):
    return ResNet(resnet_mcdo.Bottleneck, [3, 8, 36, 3],
                  num_sblocks=[1, 1, 1, 1],
                  num_classes=num_classes, stem=stem, name=name, **block_kwargs)
