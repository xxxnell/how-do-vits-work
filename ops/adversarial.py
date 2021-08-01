"""
This code is based on the implementation of https://github.com/Harry24k/adversarial-attacks-pytorch
"""
import torch
import torch.nn as nn


class FGSM:

    def __init__(self, model, *, loss=nn.CrossEntropyLoss(), eps=0.007, gpu=True):
        super().__init__()
        self.model = model
        self.eps = eps
        self.loss = loss
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
            ys_pred = self.model(xs)
            loss = self.loss(ys_pred, ys)

            grad = torch.autograd.grad(loss, xs)
            grad = grad[0]

        xs_adv = xs + self.eps * grad.sign()
        xs_adv = xs_adv.detach()

        return xs_adv, ys


class PGD:

    def __init__(self, model, *,
                 loss=nn.CrossEntropyLoss(),
                 eps=0.3, alpha=2 / 255, steps=40, random_start=True, gpu=True):
        super().__init__()
        self.model = model
        self.loss = loss
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
                outputs = self.model(xs_adv)

                cost = self.loss(outputs, ys)

                grad = torch.autograd.grad(cost, xs_adv)
                grad = grad[0]

                xs_adv = xs_adv.detach() + self.alpha * grad.sign()
                delta = torch.clamp(xs_adv - xs, min=-self.eps, max=self.eps)
                xs_adv = xs + delta
                xs_adv = xs_adv.detach()

        return xs_adv, ys
