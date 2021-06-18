"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, channel, dim_out, dropout=0.0):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel * patch_size ** 2

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim_out),
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


class ViT(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, depth, dim, heads, mlp_dim,
                 channel=3, head_dim=64, dropout=0.0, emb_dropout=0.0,
                 name="vit"):
        super().__init__()
        self.name = name

        self.embedding = PatchEmbedding(image_size, patch_size, channel, dim, emb_dropout)

        self.transformers = [Transformer(dim, heads, head_dim, mlp_dim, dropout)
                             for _ in range(depth)]
        self.transformers = nn.Sequential(*self.transformers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = x[:, 0]
        x = self.mlp_head(x)

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
        name=name,
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
        name=name,
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
        name=name,
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
        name=name,
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
        name=name,
    )
