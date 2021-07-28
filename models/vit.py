"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
import torch
from torch import nn

from models.layers import Lambda
from models.embeddings import PatchEmbedding, CLSToken, AbsPosEmbedding
from models.attentions import Transformer


class ViT(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, depth, dim, heads, dim_mlp,
                 channel=3, dim_head=64, dropout=0.0, emb_dropout=0.0, sd=0.0,
                 embedding=None, classifier=None,
                 name="vit", **block_kwargs):
        super().__init__()
        self.name = name

        self.embedding = nn.Sequential(
            PatchEmbedding(image_size, patch_size, dim, channel=channel),
            CLSToken(dim),
            AbsPosEmbedding(image_size, patch_size, dim, cls=True),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity()
        ) if embedding is None else embedding

        self.transformers = []
        for i in range(depth):
            self.transformers.append(
                Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
                            dropout=dropout, sd=(sd * i / (depth - 1)))
            )
        self.transformers = nn.Sequential(*self.transformers)

        self.classifier = nn.Sequential(
            Lambda(lambda x: x[:, 0]),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)

        return x


def tiny(num_classes=1000, name="vit_ti",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=192, heads=3, dim_head=64, dim_mlp=768,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        name=name, **block_kwargs,
    )


def small(num_classes=1000, name="vit_s",
          image_size=224, patch_size=16, channel=3,
          depth=12, dim=384, heads=6, dim_head=64, dim_mlp=1536,
          **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        name=name, **block_kwargs,
    )


def base(num_classes=1000, name="vit_b",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=768, heads=12, dim_head=64, dim_mlp=3072,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        name=name, **block_kwargs,
    )


def large(num_classes=1000, name="vit_l",
          image_size=224, patch_size=16, channel=3,
          depth=24, dim=1024, heads=16, dim_head=64, dim_mlp=4096,
          **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        name=name, **block_kwargs,
    )


def huge(num_classes=1000, name="vit_h",
         image_size=224, patch_size=16, channel=3,
         depth=32, dim=1280, heads=16, dim_head=80, dim_mlp=5120,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
        name=name, **block_kwargs,
    )
