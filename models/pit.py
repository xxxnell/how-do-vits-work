"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
from math import sqrt

import torch
from torch import nn

from einops import rearrange

from models.layers import Lambda
from models.embeddings import ConvEmbedding, CLSToken, AbsPosEmbedding
from models.attentions import Transformer


class DepthwiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out,
                      kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            # nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Pool(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cls_ff = nn.Linear(dim_in, dim_out)
        self.downsample = DepthwiseConv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        cls_token, spat_tokens = x[:, :1], x[:, 1:]
        _, s, _ = spat_tokens.shape
        h, w = int(sqrt(s)), int(sqrt(s))

        cls_token = self.cls_ff(cls_token)

        spat_tokens = rearrange(spat_tokens, 'b (h w) c -> b c h w', h=h, w=w)
        spat_tokens = self.downsample(spat_tokens)
        spat_tokens = rearrange(spat_tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, spat_tokens), dim=1)


class PiT(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, dims, depths, heads, dims_head, dims_mlp,
                 channel=3, dropout=0.0, emb_dropout=0.0, sd=0.0, stride=None,
                 embedding=None, classifier=None,
                 name="pit"):
        super().__init__()
        self.name = name

        if len(depths) != 3:
            msg = "`depths` must be a tuple of integers with len of 3, " + \
                  "specifying the number of blocks before each downsizing."
            raise Exception(msg)
        dims = self._to_tuple(dims, len(depths))
        dims = (dims[0], *dims)  # (stem_dim, *stage_dims)
        heads = self._to_tuple(heads, len(depths))
        dims_head = self._to_tuple(dims_head, len(depths))
        dims_mlp = self._to_tuple(dims_mlp, len(depths))
        idxs = [[j for j in range(sum(depths[:i]), sum(depths[:i + 1]))] for i in range(len(depths))]
        sds = [[sd * j / (sum(depths) - 1) for j in js] for js in idxs]
        pools = [False] + [True] * (len(depths) - 1)

        self.embedding = nn.Sequential(
            ConvEmbedding(patch_size, dims[0], channel=channel, stride=stride),
            CLSToken(dims[0]),
            AbsPosEmbedding(image_size, patch_size, dims[0], stride=stride),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity()
        ) if embedding is None else embedding

        self.transformers = []
        for i in range(len(depths)):
            if pools[i]:
                self.transformers.append(Pool(dims[i], dims[i+1]))
            for j in range(depths[i]):
                self.transformers.append(
                    Transformer(dims[i+1],
                                heads=heads[i], dim_head=dims_head[i], dim_mlp=dims_mlp[i],
                                dropout=dropout, sd=sds[i][j])
                )
        self.transformers = nn.Sequential(*self.transformers)

        self.classifier = nn.Sequential(
            Lambda(lambda x: x[:, 0]),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        ) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)

        return x

    @staticmethod
    def _to_tuple(v, l):
        return v if isinstance(v, tuple) or isinstance(v, list) else (v,) * l


def tiny(num_classes=1000, name="pit_ti",
         image_size=224, patch_size=16, channel=3, stride=8,
         dims=(64, 128, 256), depths=(2, 6, 4), heads=(2, 4, 8), dims_head=(32, 32, 32),
         dims_mlp=(256, 512, 1024), dropout=0.0, emb_dropout=0.0, sd=0.0,
         **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel, stride=stride,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, dims_head=dims_head,
        dims_mlp=dims_mlp, dropout=dropout, emb_dropout=emb_dropout, sd=sd,
        name=name, **block_kwargs
    )


def xsmall(num_classes=1000, name="pit_xs",
           image_size=224, patch_size=16, channel=3, stride=8,
           dims=(96, 192, 384), depths=(2, 6, 4), heads=(2, 4, 8), dims_head=(48, 48, 48),
           dims_mlp=(384, 768, 1024), dropout=0.0, emb_dropout=0.0, sd=0.0,
           **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel, stride=stride,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, dims_head=dims_head,
        dims_mlp=dims_mlp, dropout=dropout, emb_dropout=emb_dropout, sd=sd,
        name=name, **block_kwargs
    )


def small(num_classes=1000, name="pit_s",
          image_size=224, patch_size=16, channel=3, stride=8,
          dims=(144, 288, 576), depths=(2, 6, 4), heads=(3, 6, 12), dims_head=(48, 48, 48),
          dims_mlp=(576, 1152, 2304), dropout=0.0, emb_dropout=0.0, sd=0.0,
          **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel, stride=stride,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, dims_head=dims_head,
        dims_mlp=dims_mlp, dropout=dropout, emb_dropout=emb_dropout, sd=sd,
        name=name, **block_kwargs
    )


def base(num_classes=1000, name="pit_base",
         image_size=224, patch_size=16, channel=3, stride=7,
         dims=(256, 512, 1024), depths=(3, 6, 4), heads=(4, 8, 16), dims_head=(64, 64, 64),
         dims_mlp=(256, 512, 1024), dropout=0.0, emb_dropout=0.0, sd=0.0,
         **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel, stride=stride,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, dims_head=dims_head,
        dims_mlp=dims_mlp, dropout=dropout, emb_dropout=emb_dropout, sd=sd,
        name=name, **block_kwargs
    )
