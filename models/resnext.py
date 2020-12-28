import models.resnet as resnet
import models.resnet_dnn_block as resnet_dnn
import models.resnet_mcdo_block as resnet_mcdo


# Deterministic

def dnn_50(num_classes=10, tiny=False, name="resnext_dnn_50", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 6, 3],
                         width_per_group=4, groups=32,
                         num_classes=num_classes, tiny=tiny, name=name, **block_kwargs)


def dnn_101(num_classes=10, tiny=False, name="resnext_dnn_101", **block_kwargs):
    return resnet.ResNet(resnet_dnn.Bottleneck, [3, 4, 23, 3],
                         width_per_group=8, groups=32,
                         num_classes=num_classes, tiny=tiny, name=name, **block_kwargs)


# MC dropout

def mcdo_50(num_classes=10, tiny=False, name="resnext_mcdo_50", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 6, 3],
                         width_per_group=4, groups=32,
                         num_classes=num_classes, tiny=tiny, name=name, **block_kwargs)


def mcdo_101(num_classes=10, tiny=False, name="resnext_mcdo_101", **block_kwargs):
    return resnet.ResNet(resnet_mcdo.Bottleneck, [3, 4, 23, 3],
                         width_per_group=8, groups=32,
                         num_classes=num_classes, tiny=tiny, name=name, **block_kwargs)
