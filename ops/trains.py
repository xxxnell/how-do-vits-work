import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(model, name, **kwargs):
    sch_kwargs = kwargs.pop("scheduler", {})
    if name in ["SGD", "Sgd", "sgd"]:
        optimizer = optim.SGD(model.parameters(), **kwargs)
    elif name in ["Adam", "adam"]:
        optimizer = optim.Adam(model.parameters(), **kwargs)
    else:
        raise ValueError("The Optimizer %s is not provided." % name)

    sch_name = sch_kwargs.pop("name")
    if sch_name in ["MultiStepLR"]:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sch_kwargs)
    else:
        raise ValueError("The Scheduler %s is not provided." % name)

    return optimizer, train_scheduler


def train_epoch(optimizer, model, dataset,
                scheduler=None, gpu=True):
    model.train()
    nll_function = nn.CrossEntropyLoss()
    nlls = []

    for step, (xs, ys) in enumerate(dataset):
        if gpu:
            xs = xs.cuda()
            ys = ys.cuda()

        optimizer.zero_grad()
        logits = model(xs)
        nll = nll_function(logits, ys)
        nll.backward()
        optimizer.step()

        nlls.append(nll.item())

        if scheduler:
            scheduler.step()

    return np.mean(nlls)
