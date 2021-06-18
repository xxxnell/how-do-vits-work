"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
from math import sqrt

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ConvEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, channel, dim_out, stride=None, dropout=0.0):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception('Image dimensions must be divisible by the patch size.')
        stride = patch_size // 2 if stride is None else stride

        patch_dim = channel * patch_size ** 2
        output_size = self._conv_output_size(image_size, patch_size, stride)
        num_patches = output_size ** 2

        self.patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=patch_size // 2),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim_out)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_out))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        return x

    @staticmethod
    def _conv_output_size(image_size, kernel_size, stride, padding=0):
        return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):

    def __init__(self, dim, heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x, attn = self.mhsa(x)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + skip

        return x


class DepthwiseConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias)
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

        self.embedding = ConvEmbedding(image_size, patch_size, channel, dims[0], stride, emb_dropout)

        self.transformers = []
        for i in range(len(depths)):
            if pools[i]:
                self.transformers.append(Pool(dims[i], dims[i+1]))
            for _ in range(depths[i]):
                self.transformers.append(Transformer(dims[i+1], heads[i], head_dims[i], mlp_dims[i], dropout))
        self.transformers = nn.Sequential(*self.transformers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, img):
        x = self.embedding(img)
        x = self.transformers(x)
        x = x[:, 0]
        x = self.mlp_head(x)

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
        name=name
    )


def xsmall(num_classes=1000, name="pit_xs",
           image_size=224, patch_size=16, channel=3,
           dims=(96, 192, 256), depths=(2, 6, 4), heads=(2, 4, 8), head_dims=(48, 48, 48),
           mlp_dims=(384, 768, 1024), dropout=0.1, emb_dropout=0.1,
           **block_kwargs):
    return PiT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depths=depths,
        dims=dims, heads=heads, head_dims=head_dims,
        mlp_dims=mlp_dims, dropout=dropout, emb_dropout=emb_dropout,
        name=name
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
        name=name
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
        name=name
    )
