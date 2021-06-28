"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
import torch
from torch import nn

from einops.layers.torch import Reduce

from models.layers import Lambda
from models.embeddings import PatchEmbedding, CLSToken, AbsPosEmbedding
from models.attentions import Attention, MiniAttention, Transformer, MiniTransformer
from models.gates import SpatialGate


class ViT(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, depth, dim, heads, mlp_dim,
                 channel=3, head_dim=64, dropout=0.0, emb_dropout=0.0,
                 embedding=None, classifier=None,
                 name="vit", **block_kwargs):
        super().__init__()
        self.name = name

        self.embedding = nn.Sequential(
            PatchEmbedding(image_size, patch_size, dim, channel=channel),
            CLSToken(dim),
            AbsPosEmbedding(image_size, patch_size, dim, cls=True),
            nn.Dropout(emb_dropout)
        ) if embedding is None else embedding

        self.transformers = [Transformer(dim, heads=heads, head_dim=head_dim, mlp_dim=mlp_dim, dropout=dropout, attn=Attention)
                             for _ in range(depth)]
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
         depth=12, dim=192, heads=3, head_dim=64, mlp_dim=768, dropout=0.1, emb_dropout=0.1,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, head_dim=head_dim,
        mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs,
    )


def small(num_classes=1000, name="vit_s",
          image_size=224, patch_size=16, channel=3,
          depth=12, dim=384, heads=6, head_dim=64, mlp_dim=1536, dropout=0.1, emb_dropout=0.1,
          **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, head_dim=head_dim,
        mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs,
    )


def base(num_classes=1000, name="vit_b",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=768, heads=12, head_dim=64, mlp_dim=3072, dropout=0.1, emb_dropout=0.1,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, head_dim=head_dim,
        mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs,
    )


def large(num_classes=1000, name="vit_l",
          image_size=224, patch_size=16, channel=3,
          depth=24, dim=1024, heads=16, head_dim=64, mlp_dim=4096, dropout=0.1, emb_dropout=0.1,
          **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, head_dim=head_dim,
        mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs,
    )


def huge(num_classes=1000, name="vit_h",
         image_size=224, patch_size=16, channel=3,
         depth=32, dim=1280, heads=16, head_dim=80, mlp_dim=5120, dropout=0.1, emb_dropout=0.1,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, head_dim=head_dim,
        mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout,
        name=name, **block_kwargs,
    )
