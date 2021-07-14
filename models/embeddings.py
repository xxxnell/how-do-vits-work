"""
These modules are based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, dim_out, channel=3):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        patch_dim = channel * patch_size ** 2

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim_out),
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        return x


class CLSToken(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        return x


class AbsPosEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, dim, stride=None, cls=True):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(image_size, patch_size, stride)
        num_patches = output_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)

    def forward(self, x):
        x = x + self.pos_embedding

        return x

    @staticmethod
    def _conv_output_size(image_size, kernel_size, stride, padding=0):
        return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


class PatchUnembedding(nn.Module):

    def __init__(self, image_size, patch_size):
        super().__init__()
        h, w = image_size // patch_size, image_size // patch_size

        self.rearrange = nn.Sequential(
            Rearrange('b (h w) (p1 p2 d) -> b d (h p1) (w p2)',
                      h=h, w=w, p1=patch_size, p2=patch_size),
        )

    def forward(self, x):
        x = x[:, 1:]
        x = self.rearrange(x)

        return x


class ConvEmbedding(nn.Module):

    def __init__(self, patch_size, dim_out, channel=3, stride=None):
        super().__init__()
        stride = patch_size if stride is None else stride
        patch_dim = channel * patch_size ** 2

        self.patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=stride),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim_out)
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        return x
