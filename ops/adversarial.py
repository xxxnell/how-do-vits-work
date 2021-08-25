"""
This code is based on the implementation of https://github.com/Harry24k/adversarial-attacks-pytorch
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM:

    def __init__(self, model, *, loss=nn.CrossEntropyLoss(), n_ff=1, eps=0.007, gpu=True):
        super().__init__()
        self.model = model
        self.eps = eps
        self.loss = loss
        self.n_ff = n_ff
        self.gpu = gpu

    def __call__(self, xs, ys):
        xs = xs.clone().detach()
        ys = ys.clone().detach()

        if self.gpu:
            self.model = self.model.cuda()
            xs = xs.cuda()
            ys = ys.cuda()

        with torch.set_grad_enabled(True):
            xs.requires_grad = True
            ys_pred = torch.stack([self.model(xs) for _ in range(self.n_ff)])
            ys_pred = torch.softmax(ys_pred, dim=-1)
            ys_pred = torch.mean(ys_pred, dim=0)
            ys_pred = ys_pred.log()

            loss = self.loss(ys_pred, ys)

            grad = torch.autograd.grad(loss, xs)
            grad = grad[0]

        xs_adv = xs + self.eps * grad.sign()
        xs_adv = xs_adv.detach()

        return xs_adv, ys


class PGD:

    def __init__(self, model, *,
                 loss=nn.CrossEntropyLoss(), n_ff=1,
                 eps=0.3, alpha=2 / 255, steps=40, random_start=True, gpu=True):
        super().__init__()
        self.model = model
        self.loss = loss
        self.n_ff = n_ff
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.gpu = gpu

    def __call__(self, xs, ys):
        xs = xs.clone().detach()
        ys = ys.clone().detach()

        if self.gpu:
            self.model = self.model.cuda()
            xs = xs.cuda()
            ys = ys.cuda()

        xs_adv = xs.clone().detach()

        if self.random_start:
            xs_adv = xs_adv + torch.empty_like(xs_adv).uniform_(-self.eps, self.eps)
            xs_adv = torch.clamp(xs_adv, min=0, max=1).detach()

        for _ in range(self.steps):
            with torch.set_grad_enabled(True):
                xs_adv.requires_grad = True
                ys_pred = torch.stack([self.model(xs_adv) for _ in range(self.n_ff)])
                ys_pred = torch.softmax(ys_pred, dim=-1)
                ys_pred = torch.mean(ys_pred, dim=0)
                ys_pred = ys_pred.log()

                cost = self.loss(ys_pred, ys)

                grad = torch.autograd.grad(cost, xs_adv)
                grad = grad[0]

                xs_adv = xs_adv.detach() + self.alpha * grad.sign()
                delta = torch.clamp(xs_adv - xs, min=-self.eps, max=self.eps)
                xs_adv = xs + delta
                xs_adv = xs_adv.detach()

        return xs_adv, ys


class Random:

    def __init__(self, model=None, *, eps=0.007, gpu=True):
        super().__init__()
        self.eps = eps
        self.gpu = gpu

    def __call__(self, xs, ys):
        xs = xs.clone().detach()
        ys = ys.clone().detach()

        if self.gpu:
            xs = xs.cuda()
            ys = ys.cuda()

        b, c, h, w = xs.shape
        random = torch.randn([b, c, h, w])
        random = random.to(xs.device)

        xs_adv = xs + self.eps * random.sign()
        xs_adv = xs_adv.detach()

        return xs_adv, ys


class FreqAttack:

    def __init__(self, attack, *, f, s=0.2):
        super().__init__()
        self.attack = attack
        self.f = f
        self.s = s

    def __call__(self, xs, ys):
        xs_adv, ys = self.attack(xs, ys)
        xs_adv = xs + self._fourier_mask(xs_adv - xs, self.f, self.s).real

        return xs_adv, ys

    def _fourier_mask(self, x, f, s=0.2):
        b, c, h, w = x.shape

        # A. FFT
        x = torch.fft.fft2(x)
        x = self._shift(x)
        x_abs = x.abs()
        x_ang = x.angle()

        # B. Mask
        mask1 = self._center_mask(int(((f + s) * h) / (2 * math.pi)) * 2, h)
        mask2 = self._center_mask(int(((f - s) * h) / (2 * math.pi)) * 2, h)
        mask = mask1 - mask2
        mask = mask.to(x_abs.device)
        x_abs = mask * x_abs

        # C. Inverse FFT
        unit = torch.complex(torch.zeros(b, c, h, w), torch.ones(b, c, h, w))
        unit = unit.to(x.device)
        x = x_abs * torch.exp(unit * x_ang)

        x = self._shift(x)
        x = torch.fft.ifft2(x)

        return x

    def _shift(self, x):
        b, c, h, w = x.shape
        x = torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))
        return x

    def _center_mask(self, w1, w2):
        w1 = w2 if w1 > w2 else w1
        w1 = 0 if w1 < 0 else w1
        mask = torch.ones([1, 3, w1, w1])
        mask = F.pad(mask, [int((w2 - w1) / 2)] * 4)

        return mask
