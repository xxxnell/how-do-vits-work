"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
import torch
from torch import nn
from functools import partial

from einops.layers.torch import Reduce

from models.embeddings import PatchEmbedding
from models.attentions import FeedForward


class Mixer(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, hidden_dim, spatial_dim, channel_dim, depth,
                 channel=3, dropout=0.0,
                 name="mixer"):
        super().__init__()
        self.name = name
        if (image_size % patch_size) != 0:
            raise Exception("Image must be divisible by patch size.")
        num_patches = (image_size // patch_size) ** 2

        self.embedding = PatchEmbedding(image_size, patch_size, hidden_dim, channel=channel)

        self.mlps = []
        f1, f2 = partial(nn.Conv1d, kernel_size=1), nn.Linear
        for _ in range(depth):
            self.mlps.append(nn.LayerNorm(hidden_dim))
            self.mlps.append(FeedForward(num_patches, spatial_dim, f=f1, dropout=dropout))
            self.mlps.append(nn.LayerNorm(hidden_dim))
            self.mlps.append(FeedForward(hidden_dim, channel_dim, f=f2, dropout=dropout))
        self.mlps = nn.Sequential(*self.mlps)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.mlps(x)
        x = self.classifier(x)

        return x


def tiny(num_classes=1000, name="mixer_ti",
         image_size=224, patch_size=16,
         depth=8, hidden_dim=256, spatial_dim=128, channel_dim=1024,
         channels=3, dropout=0.0):
    return Mixer(
        image_size=image_size, patch_size=patch_size, channel=channels,
        num_classes=num_classes, depth=depth,
        hidden_dim=hidden_dim, spatial_dim=spatial_dim, channel_dim=channel_dim,
        dropout=dropout,
        name=name
    )


def small(num_classes=1000, name="mixer_s",
          image_size=224, patch_size=16,
          depth=8, hidden_dim=512, spatial_dim=256, channel_dim=2048,
          channels=3, dropout=0.0):
    return Mixer(
        image_size=image_size, patch_size=patch_size, channel=channels,
        num_classes=num_classes, depth=depth,
        hidden_dim=hidden_dim, spatial_dim=spatial_dim, channel_dim=channel_dim,
        dropout=dropout,
        name=name
    )


def base(num_classes=1000, name="mixer_b",
         image_size=224, patch_size=16,
         depth=12, hidden_dim=768, spatial_dim=384, channel_dim=3072,
         channels=3, dropout=0.0):
    return Mixer(
        image_size=image_size, patch_size=patch_size, channel=channels,
        num_classes=num_classes, depth=depth,
        hidden_dim=hidden_dim, spatial_dim=spatial_dim, channel_dim=channel_dim,
        dropout=dropout,
        name=name
    )


def large(num_classes=1000, name="mixer_l",
          image_size=224, patch_size=16,
          depth=24, hidden_dim=1024, spatial_dim=512, channel_dim=4096,
          channels=3, dropout=0.0):
    return Mixer(
        image_size=image_size, patch_size=patch_size, channel=channels,
        num_classes=num_classes, depth=depth,
        hidden_dim=hidden_dim, spatial_dim=spatial_dim, channel_dim=channel_dim,
        dropout=dropout,
        name=name
    )


def huge(num_classes=1000, name="mixer_h",
         image_size=224, patch_size=16,
         depth=32, hidden_dim=1280, spatial_dim=640, channel_dim=5120,
         channels=3, dropout=0.0):
    return Mixer(
        image_size=image_size, patch_size=patch_size, channel=channels,
        num_classes=num_classes, depth=depth,
        hidden_dim=hidden_dim, spatial_dim=spatial_dim, channel_dim=channel_dim,
        dropout=dropout,
        name=name
    )
