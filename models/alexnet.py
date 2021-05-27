import torch
import torch.nn as nn
import models.alexnet_dnn_block as alexnet_dnn
import models.alexnet_mcdo_block as alexnet_mcdo
import models.smoothing_block as smoothing
import models.classifier_block as classifier


class AlexNet(nn.Module):

    def __init__(self, block,
                 sblock=smoothing.TanhBlurBlock, num_sblocks=(0, 0, 0),
                 cblock=classifier.MLPBlock,
                 num_classes=10, stem=True, name="alexnet", **block_kwargs):
        super(AlexNet, self).__init__()

        self.name = name

        self.layer0 = []
        if stem:
            self.layer0.append(block(3, 64, kernel_size=11, stride=4, padding=2, **block_kwargs))
        else:
            self.layer0.append(block(3, 64, kernel_size=3, stride=2, padding=1, **block_kwargs))
        self.layer0 = nn.Sequential(*self.layer0)

        self.layer1 = []
        kernel_size = 3 if stem else 2
        self.layer1.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2))
        self.layer1.append(block(64, 192, kernel_size=(5, 5), padding=2, **block_kwargs))
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = []
        kernel_size = 3 if stem else 2
        self.layer2.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2))
        self.layer2.append(block(192, 384, kernel_size=3, padding=1, **block_kwargs))
        self.layer2.append(block(384, 256, kernel_size=3, padding=1, **block_kwargs))
        self.layer2.append(block(256, 256, kernel_size=3, padding=1, **block_kwargs))
        self.layer2 = nn.Sequential(*self.layer2)

        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0], **block_kwargs)
        self.smooth1 = self._make_smooth_layer(sblock, 192, num_sblocks[1], **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2], **block_kwargs)

        self.classifier = []
        if cblock is classifier.MLPBlock:
            kernel_size, out_size = (3, 6) if stem else (2, 2)
            self.classifier.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2))
            self.classifier.append(nn.AdaptiveAvgPool2d((out_size, out_size)))
            self.classifier.append(cblock(out_size * out_size * 256, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(256, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)

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

        x = self.classifier(x)

        return x


def dnn(num_classes=10, stem=True, name="alexnet_dnn", **block_kwargs):
    return AlexNet(alexnet_dnn.BasicBlock, num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo(num_classes=10, stem=True, name="alexnet_mcdo", **block_kwargs):
    return AlexNet(alexnet_mcdo.BasicBlock, num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def dnn_smooth(num_classes=10, stem=True, name="alexnet_dnn_smoothing", **block_kwargs):
    return AlexNet(alexnet_dnn.BasicBlock,
                   num_sblocks=[1, 1, 1],
                   num_classes=num_classes, stem=stem, name=name, **block_kwargs)


def mcdo_smooth(num_classes=10, stem=True, name="alexnet_mcdo_smoothing", **block_kwargs):
    return AlexNet(alexnet_mcdo.BasicBlock,
                   num_sblocks=[1, 1, 1],
                   num_classes=num_classes, stem=stem, name=name, **block_kwargs)

