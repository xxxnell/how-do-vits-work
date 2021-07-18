import math
import torch.nn as nn

import models.layers as layers
import models.classifier_block as classifier


class Basic(nn.Module):

    def __init__(self, dim_in, dim_out, stride, expand_ratio, **block_kwargs):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(dim_in * expand_ratio)
        self.identity = stride == 1 and dim_in == dim_out

        self.conv1 = nn.Sequential(
            layers.conv3x3(dim_in, hidden_dim, stride=stride, groups=dim_in),
            layers.bn(hidden_dim),
            layers.relu6(),
        )
        self.conv2 = nn.Sequential(
            layers.conv1x1(hidden_dim, dim_out),
            layers.bn(dim_out)
        )

    def forward(self, x):
        skip = x

        x = self.conv1(x)
        x = self.conv2(x)

        x = (x + skip) if self.identity else x

        return x


class Bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride, expand_ratio, **block_kwargs):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(dim_in * expand_ratio)
        self.identity = stride == 1 and dim_in == dim_out

        self.conv1 = nn.Sequential(
            layers.conv1x1(dim_in, hidden_dim),
            layers.bn(hidden_dim),
            layers.relu6(),
        )
        self.conv2 = nn.Sequential(
            layers.conv3x3(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            layers.bn(hidden_dim),
            layers.relu6(),
        )
        self.conv3 = nn.Sequential(
            layers.conv1x1(hidden_dim, dim_out),
            layers.bn(dim_out)
        )

    def forward(self, x):
        skip = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = (x + skip) if self.identity else x

        return x


class MobileNet(nn.Module):

    def __init__(self, *,
                 num_classes=1000, channel=3, width_mult=1.0,
                 stem=True, block1=Basic, block2=Bottleneck,
                 cblock=classifier.GAPBlock,
                 name="mobilenet", **block_kwargs):
        super().__init__()
        self.name = name
        conf = self._conf(stem)
        min_value = 4 if width_mult < 0.1 else 8

        # building first layer
        dim_in = self._make_divisible(32 * width_mult, min_value)
        self.layers = []
        self.layers.append(nn.Sequential(
            layers.conv3x3(channel, dim_in, stride=2 if stem else 1),
            layers.bn(dim_in),
            layers.relu6(),
        ))

        # building inverted residual blocks
        for t, c, n, s in conf:
            block = block1 if t == 1 else block2
            dim_out = self._make_divisible(c * width_mult, min_value)
            for i in range(n):
                self.layers.append(block(dim_in, dim_out, stride=s if i == 0 else 1, expand_ratio=t, **block_kwargs))
                dim_in = dim_out

        # building last several layers
        dim_out = self._make_divisible(1280 * width_mult, min_value) if width_mult > 1.0 else 1280
        self.layers.append(nn.Sequential(
            layers.conv1x1(dim_in, dim_out),
            layers.bn(dim_out),
            layers.relu6(),
        ))

        self.features = nn.Sequential(*self.layers)
        self.classifier = cblock(dim_out, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    @staticmethod
    def _conf(stem=True):
        # expansion (t), dim_out (c), num_blocks (n), stride (s)
        conf = [
            (1, 16, 1, 1),
            (6, 24, 2, 2 if stem else 1),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        return conf

    @staticmethod
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def dnn(num_classes=1000, stem=True, name="mobilenet", **block_kwargs):
    return MobileNet(num_classes=num_classes, stem=stem, name=name, **block_kwargs)

