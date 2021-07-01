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
                 image_size, patch_size, num_classes, dims, depths, heads, head_dims, mlp_dims,
                 channel=3, dropout=0.0, emb_dropout=0.0, stride=None,
                 embedding=None, classifier=None,
                 name="pit"):
        super().__init__()
        self.name = name

        if len(depths) is not 3:
            msg = "`depths` must be a tuple of integers with len of 3, " + \
                  "specifying the number of blocks before each downsizing."
            raise Exception(msg)
        dims = self._to_tuple(dims, len(depths))
        dims = (dims[0], *dims)  # (stem_dim, *stage_dims)
        heads = self._to_tuple(heads, len(depths))
        head_dims = self._to_tuple(head_dims, len(depths))
        mlp_dims = self._to_tuple(mlp_dims, len(depths))
        pools = [False, True, True]

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
            for _ in range(depths[i]):
                self.transformers.append(
                    Transformer(dims[i+1],
                                heads=heads[i], head_dim=head_dims[i], mlp_dim=mlp_dims[i], dropout=dropout)
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
         image_size=224, patch_size=16, channel=3,
         dims=(64, 128, 256), depths=(2, 6, 4), heads=(2, 4, 8), head_dims=(32, 32, 32),
         mlp_dims=(256, 512, 1024), dropout=0.1, emb_dropout=0.1,
         **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, head_dims=head_dims,
        mlp_dims=mlp_dims, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs
    )


def xsmall(num_classes=1000, name="pit_xs",
           image_size=224, patch_size=16, channel=3,
           dims=(96, 192, 384), depths=(2, 6, 4), heads=(2, 4, 8), head_dims=(48, 48, 48),
           mlp_dims=(384, 768, 1024), dropout=0.1, emb_dropout=0.1,
           **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, head_dims=head_dims,
        mlp_dims=mlp_dims, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs
    )


def small(num_classes=1000, name="pit_s",
         image_size=224, patch_size=16, channel=3,
         dims=(144, 288, 576), depths=(2, 6, 4), heads=(3, 6, 12), head_dims=(48, 48, 48),
         mlp_dims=(576, 1152, 2304), dropout=0.1, emb_dropout=0.1,
         **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, head_dims=head_dims,
        mlp_dims=mlp_dims, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs
    )


def base(num_classes=1000, name="pit_base",
         image_size=224, patch_size=16, channel=3, stride=7,
         dims=(256, 512, 1024), depths=(3, 6, 4), heads=(4, 8, 16), head_dims=(64, 64, 64),
         mlp_dims=(256, 512, 1024), dropout=0.1, emb_dropout=0.1,
         **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel, stride=stride,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, head_dims=head_dims,
        mlp_dims=mlp_dims, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs
    )
