"""
This ConViT is ViT with two-dimensional convolutional MSA (cf. [1]), NOT [2]!

[1] Baosong Yang, Longyue Wang, Derek F Wong, Lidia S Chao, and Zhaopeng Tu. "Convolutional self-attention 
networks". NAACL, 2019.
[2] d'Ascoli, Stéphane, et al. "Convit: Improving vision transformers with soft convolutional inductive biases."
arXiv preprint arXiv:2103.10697 (2021).
"""
from functools import partial

import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum

import models.layers as layers
from models.attentions import Transformer
from models.embeddings import AbsPosEmbedding


class ConvAttention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1,
                 kernel_size=1, dilation=1, padding=0, stride=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.conv_args = {
            "kernel_size": kernel_size,
            "dilation": dilation,
            "padding": padding,
            "stride": stride
        }

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        q, kv = self.to_q(x), self.to_kv(x).chunk(2, dim=1)

        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.heads)
        q = repeat(q, 'b h n d -> b h n w d', w=self.conv_args["kernel_size"] ** 2)
        k, v = map(lambda t: F.unfold(t, **self.conv_args), kv)
        k, v = map(lambda t: rearrange(t, 'b (h d w) n -> b h n w d', h=self.heads, d=q.shape[-1]), (k, v))

        dots = einsum('b h n w d, b h n w d -> b h n w', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h n w, b h n w d -> b h n d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn

    def extra_repr(self):
        return ", ".join(["%s=%s" % (k, v) for k, v in self.conv_args.items()])


class ConViT(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, depth, dim, heads, dim_mlp,
                 channel=3, dim_head=64, dropout=0.0, emb_dropout=0.0, sd=0.0,
                 k=1, kernel_size=3, dilation=1, padding=0, stride=1,
                 embedding=None, classifier=None,
                 name="convit", **block_kwargs):
        super().__init__()
        self.name = name

        self.embedding = nn.Sequential(
            nn.Conv2d(channel, dim, patch_size, stride=patch_size, padding=0),
            Rearrange('b c x y -> b (x y) c'),
            AbsPosEmbedding(image_size, patch_size, dim, cls=False),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity(),
            Rearrange('b (x y) c -> b c x y', y=image_size // patch_size),
        ) if embedding is None else embedding

        attn = partial(
            ConvAttention2d,
            k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
        )
        f = partial(nn.Conv2d, kernel_size=1, stride=1)
        self.transformers = []
        for i in range(depth):
            self.transformers.append(
                Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
                            attn=attn, f=f, norm=layers.ln2d,
                            dropout=dropout, sd=(sd * i / (depth - 1)))
            )
        self.transformers = nn.Sequential(*self.transformers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)

        return x


def tiny(num_classes=1000, name="convit_ti",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=192, heads=3, dim_head=64, dim_mlp=768,
         k=1, kernel_size=3, dilation=1, padding=0, stride=1,
         **block_kwargs):
    return ConViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride,
        name=name, **block_kwargs,
    )


def small(num_classes=1000, name="convit_s",
          image_size=224, patch_size=16, channel=3,
          depth=12, dim=384, heads=6, dim_head=64, dim_mlp=1536,
          k=1, kernel_size=3, dilation=1, padding=0, stride=1,
          **block_kwargs):
    return ConViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride,
        name=name, **block_kwargs,
    )


def base(num_classes=1000, name="convit_b",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=768, heads=12, dim_head=64, dim_mlp=3072,
         k=1, kernel_size=3, dilation=1, padding=0, stride=1,
         **block_kwargs):
    return ConViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride,
        name=name, **block_kwargs,
    )


def large(num_classes=1000, name="convit_l",
          image_size=224, patch_size=16, channel=3,
          depth=24, dim=1024, heads=16, dim_head=64, dim_mlp=4096,
          k=1, kernel_size=3, dilation=1, padding=0, stride=1,
          **block_kwargs):
    return ConViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride,
        name=name, **block_kwargs,
    )


def huge(num_classes=1000, name="convit_h",
         image_size=224, patch_size=16, channel=3,
         depth=32, dim=1280, heads=16, dim_head=80, dim_mlp=5120,
         k=1, kernel_size=3, dilation=1, padding=0, stride=1,
         **block_kwargs):
    return ConViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        k=k, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride,
        name=name, **block_kwargs,
    )
