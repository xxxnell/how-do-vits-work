import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

import ops.meters
import ops.norm

def get_optimizer(model, name, **kwargs):
    sch_kwargs = kwargs.pop("scheduler", {})
    if name in ["SGD", "Sgd", "sgd"]:
        optimizer = optim.SGD(model.parameters(), **kwargs)
    elif name in ["Adam", "adam"]:
        optimizer = optim.Adam(model.parameters(), **kwargs)
    else:
        raise NotImplementedError

    sch_name = sch_kwargs.pop("name")
    if sch_name in ["MultiStepLR"]:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sch_kwargs)
    else:
        raise NotImplementedError

    return optimizer, train_scheduler


def train_epoch(optimizer, model, dataset,
                scheduler=None, gpu=True):
    model.train()
    nll_function = nn.CrossEntropyLoss()
    nll_function = nll_function.cuda() if gpu else nll_function

    loss_meter = ops.meters.AverageMeter("loss")
    nll_meter = ops.meters.AverageMeter("nll")
    l1_meter = ops.meters.AverageMeter("l1")
    l2_meter = ops.meters.AverageMeter("l2")

    for step, (xs, ys) in enumerate(dataset):
        if gpu:
            xs = xs.cuda()
            ys = ys.cuda()

        optimizer.zero_grad()
        logits = model(xs)
        loss = nll_function(logits, ys)
        nll_meter.update(loss.item())

        loss_meter.update(loss.item())
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

    l1_reg = ops.norm.l1(model, gpu)
    l1_meter.update(l1_reg.item())

    l2_reg = ops.norm.l2(model, gpu)
    l2_meter.update(l2_reg.item())

    return loss_meter.avg, nll_meter.avg, l1_meter.avg, l2_meter.avg


def add_train_metrics(writer, metrics, epoch):
    loss, nll, l1, l2 = metrics

    writer.add_scalar("train/loss", loss, global_step=epoch)
    writer.add_scalar("train/nll", nll, global_step=epoch)
    writer.add_scalar("train/l1", l1, global_step=epoch)
    writer.add_scalar("train/l2", l2, global_step=epoch)


def add_test_metrics(writer, metrics, epoch):
    nll_value, \
    cutoffs, cms, accs, uncs, ious, freqs, \
    topk_value, brier_value, \
    count_bin, acc_bin, conf_bin, ece_value = metrics

    writer.add_scalar("test/nll", nll_value, global_step=epoch)
    writer.add_scalar("test/acc", accs[0], global_step=epoch)
    writer.add_scalar("test/acc-90", accs[1], global_step=epoch)
    writer.add_scalar("test/unc-90", uncs[1], global_step=epoch)
    writer.add_scalar("test/iou", ious[0], global_step=epoch)
    writer.add_scalar("test/iou-90", ious[1], global_step=epoch)
    writer.add_scalar("test/freq-90", freqs[1], global_step=epoch)
    writer.add_scalar("test/top-5", topk_value, global_step=epoch)
    writer.add_scalar("test/brier", brier_value, global_step=epoch)
    writer.add_scalar("test/ece", ece_value, global_step=epoch)
