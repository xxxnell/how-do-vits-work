from functools import partial

import numpy as np

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from models.layers import ln2d
from models.attentions import Attention1d, Transformer


class CyclicShift(nn.Module):

    def __init__(self, d, dims=(2, 3)):
        super().__init__()
        self.d = d
        self.dims = dims

    def forward(self, x):
        x = torch.roll(x, shifts=(self.d, self.d), dims=self.dims)
        return x


class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, pool):
        super().__init__()
        self.patch_merge = nn.Conv2d(in_channels, out_channels, kernel_size=pool, stride=pool)

    def forward(self, x):
        x = self.patch_merge(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=32, dropout=0.0, window_size=7, shifted=False):
        super().__init__()
        self.attn = Attention1d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout)
        self.window_size = window_size
        self.shifted = shifted
        self.d = window_size // 2

        self.shift = CyclicShift(-1 * self.d) if shifted else nn.Identity()
        self.backshift = CyclicShift(self.d) if shifted else nn.Identity()

        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]
        if self.shifted:
            mask = mask + self._upper_lower_mask(h // p, w // p, p, p, self.d, x.device)
            mask = mask + self._left_right_mask(h // p, w // p, p, p, self.d, x.device)
            mask = repeat(mask, "n h i j -> (b n) h i j", b=b)

        x = self.shift(x)
        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) (p1 p2) c", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) (p1 p2) c -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)
        x = self.backshift(x)

        return x, attn

    @staticmethod
    def _upper_lower_mask(n1, n2, i, j, d, device=None):
        m = torch.zeros(i ** 2, j ** 2, device=device)
        m[-d * i:, :-d * j] = float('-inf')
        m[:-d * i, -d * j:] = float('-inf')

        mask = torch.zeros(n1 * n2, 1, i ** 2, j ** 2, device=device)
        mask[-n2:] = mask[-n2:] + m

        return mask

    @staticmethod
    def _left_right_mask(n1, n2, i, j, d, device=None):
        m = torch.zeros(i ** 2, j ** 2, device=device)
        m = rearrange(m, '(i k) (j l) -> i k j l', i=i, j=j)
        m[:, -d:, :, :-d] = float('-inf')
        m[:, :-d, :, -d:] = float('-inf')
        m = rearrange(m, 'i k j l -> (i k) (j l)')

        mask = torch.zeros(n1 * n2, 1, i ** 2, j ** 2, device=device)
        mask[-n1 - 1::n1] += m

        return mask

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d


class Swin(nn.Module):

    def __init__(self, *,
                 num_classes, depths, dims, heads, dims_mlp,
                 channel=3, dim_head=32, window_size=7, pools=(4, 2, 2, 2),
                 dropout=0.0, sd=0.0,
                 classifier=None,
                 name="swin", **block_kwargs):
        super().__init__()
        self.name = name
        idxs = [[j for j in range(sum(depths[:i]), sum(depths[:i + 1]))] for i in range(len(depths))]
        sds = [[sd * j / (sum(depths) - 1) for j in js] for js in idxs]

        self.layer1 = self._make_layer(
            in_channels=channel, hidden_dimension=dims[0], depth=depths[0], window_size=window_size,
            pool=pools[0], num_heads=heads[0], dim_head=dim_head, dim_mlp=dims_mlp[0],
            dropout=dropout, sds=sds[0],
        )
        self.layer2 = self._make_layer(
            in_channels=dims[0], hidden_dimension=dims[1], depth=depths[1], window_size=window_size,
            pool=pools[1], num_heads=heads[1], dim_head=dim_head, dim_mlp=dims_mlp[1],
            dropout=dropout, sds=sds[1],
        )
        self.layer3 = self._make_layer(
            in_channels=dims[1], hidden_dimension=dims[2], depth=depths[2], window_size=window_size,
            pool=pools[2], num_heads=heads[2], dim_head=dim_head, dim_mlp=dims_mlp[2],
            dropout=dropout, sds=sds[2],
        )
        self.layer4 = self._make_layer(
            in_channels=dims[2], hidden_dimension=dims[3], depth=depths[3], window_size=window_size,
            pool=pools[3], num_heads=heads[3], dim_head=dim_head, dim_mlp=dims_mlp[3],
            dropout=dropout, sds=sds[3],
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        ) if classifier is None else classifier

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)

        return x

    @staticmethod
    def _make_layer(in_channels, hidden_dimension, depth, window_size,
                    pool, num_heads, dim_head, dim_mlp,
                    dropout, sds):
        attn1 = partial(WindowAttention, window_size=window_size, shifted=False)
        attn2 = partial(WindowAttention, window_size=window_size, shifted=True)

        seq = list()
        seq.append(PatchMerging(in_channels, hidden_dimension, pool))
        for i in range(depth // 2):
            wt = Transformer(
                hidden_dimension,
                heads=num_heads, dim_head=dim_head, dim_mlp=dim_mlp, norm=ln2d,
                attn=attn1, f=partial(nn.Conv2d, kernel_size=1),
                dropout=dropout, sd=sds[2 * i]
            )
            swt = Transformer(
                hidden_dimension,
                heads=num_heads, dim_head=dim_head, dim_mlp=dim_mlp, norm=ln2d,
                attn=attn2, f=partial(nn.Conv2d, kernel_size=1),
                dropout=dropout, sd=sds[2 * i + 1]
            )
            seq.extend([wt, swt])

        return nn.Sequential(*seq)


def swin_t(num_classes,
           dims=(96, 192, 384, 768), depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), dims_mlp=(384, 768, 1536, 3072),
           name="swin_t", **kwargs):
    return Swin(num_classes=num_classes, dims=dims, depths=depths, heads=heads, dims_mlp=dims_mlp,
                name=name, **kwargs)


def swin_s(num_classes,
           dims=(96, 192, 384, 768), depths=(2, 2, 18, 2), heads=(3, 6, 12, 24), dims_mlp=(384, 768, 1536, 3072),
           name="swin_s", **kwargs):
    return Swin(num_classes=num_classes, dims=dims, depths=depths, heads=heads, dims_mlp=dims_mlp,
                name=name, **kwargs)


def swin_b(num_classes,
           dims=(128, 256, 512, 1024), depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), dims_mlp=(512, 1024, 2048, 4096),
           name="swin_b", **kwargs):
    return Swin(num_classes=num_classes, dims=dims, depths=depths, heads=heads, dims_mlp=dims_mlp,
                name=name, **kwargs)


def swin_l(num_classes,
           dims=(192, 384, 768, 1536), depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), dims_mlp=(768, 1536, 3072, 6144),
           name="swin_l", **kwargs):
    return Swin(num_classes=num_classes, dims=dims, depths=depths, heads=heads, dims_mlp=dims_mlp,
                name=name, **kwargs)
