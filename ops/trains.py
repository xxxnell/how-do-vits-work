import os
import time
import copy
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.data import Mixup

import ops.tests as tests
import ops.meters as meters
import ops.norm as norm

import models


def get_optimizer(model, name, **kwargs):
    sch_kwargs = copy.deepcopy(kwargs.pop("scheduler", {}))
    if name in ["SGD", "Sgd", "sgd"]:
        optimizer = optim.SGD(model.parameters(), **kwargs)
    elif name in ["Adam", "adam"]:
        optimizer = optim.Adam(model.parameters(), **kwargs)
    elif name in ["AdamW", "adamw"]:
        optimizer = optim.AdamW(model.parameters(), **kwargs)
    elif name in ["RMSprop", "rmsprop"]:
        optimizer = optim.RMSprop(model.parameters(), **kwargs)
    else:
        raise NotImplementedError

    sch_name = sch_kwargs.pop("name")
    if sch_name in ["StepLR"]:
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, **sch_kwargs)
    elif sch_name in ["MultiStepLR"]:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sch_kwargs)
    elif sch_name in ["CosineAnnealingLR"]:
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sch_kwargs)
    elif sch_name in ["CosineAnnealingWarmRestarts"]:
        train_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **sch_kwargs)
    else:
        raise NotImplementedError

    return optimizer, train_scheduler


def train(model, optimizer,
          dataset_train, dataset_val,
          train_scheduler, warmup_scheduler,
          train_args, val_args, gpu,
          writer=None,
          snapshot=-1, root="models_checkpoints", dataset_name=None, uid=None,
          verbose=1):
    train_args = copy.deepcopy(train_args)
    val_args = copy.deepcopy(val_args)
    snapshot_cond = snapshot > 0 and dataset_name is not None and uid is not None

    epochs = train_args.pop("epochs")
    warmup_epochs = train_args.get("warmup_epochs", 0)
    smoothing = train_args.get("smoothing", 0.0)
    mixup_args = train_args.get("mixup", None)
    max_norm = train_args.get("max_norm", None)
    n_ff = val_args.pop("n_ff", 1)

    mixup_function = Mixup(
        **mixup_args,
        label_smoothing=smoothing,
    ) if mixup_args is not None else None

    models.save_snapshot(model, dataset_name, uid, "init", optimizer, root=root) if snapshot_cond else None

    model = model.cuda() if gpu else model.cpu()
    warmup_time = time.time()
    for epoch in range(warmup_epochs):
        batch_time = time.time()
        *train_metrics, = train_epoch(optimizer, model, dataset_train,
                                      smoothing=smoothing, mixup_function=mixup_function,
                                      max_norm=max_norm,
                                      scheduler=warmup_scheduler, gpu=gpu)
        batch_time = time.time() - batch_time
        template = "(%.2f sec/epoch) Warmup epoch: %d, Loss: %.4f, lr: %.3e"
        print(template % (batch_time,
                          epoch,
                          train_metrics[0],
                          [param_group["lr"] for param_group in optimizer.param_groups][0]))

        if writer is not None and (epoch + 1) % 1 == 0:
            *test_metrics, cal_diag = tests.test(model, n_ff, dataset_val, verbose=False, gpu=gpu)

    if warmup_epochs > 0:
        print("The model is warmed up: %.2f sec \n" % (time.time() - warmup_time))
        models.save_snapshot(model, dataset_name, uid, "warmup", optimizer, root=root) if snapshot_cond else None

    for epoch in range(epochs):
        batch_time = time.time()
        *train_metrics, = train_epoch(optimizer, model, dataset_train,
                                      smoothing=smoothing, mixup_function=mixup_function,
                                      max_norm=max_norm,
                                      gpu=gpu)
        train_scheduler.step()
        batch_time = time.time() - batch_time

        if writer is not None and (epoch + 1) % 1 == 0:
            add_train_metrics(writer, train_metrics, epoch)
            template = "(%.2f sec/epoch) Epoch: %d, Loss: %.4f, lr: %.3e"
            print(template % (batch_time,
                              epoch,
                              train_metrics[0],
                              [param_group["lr"] for param_group in optimizer.param_groups][0]))

        if writer is not None and (epoch + 1) % 1 == 0:
            *test_metrics, cal_diag = tests.test(model, n_ff, dataset_val, verbose=False, gpu=gpu)
            add_test_metrics(writer, test_metrics, epoch)

            cal_diag = torchvision.utils.make_grid(cal_diag)
            writer.add_image("test/calibration diagrams", cal_diag, global_step=epoch)

            if verbose > 1:
                for name, param in model.named_parameters():
                    name = name.split(".")
                    writer.add_histogram("%s/%s" % (name[0], ".".join(name[1:])), param, global_step=epoch)
        if snapshot_cond and (epoch + 1) % snapshot == 0:
            models.save_snapshot(model, dataset_name, uid, epoch, optimizer, root=root) if snapshot_cond else None


def train_epoch(optimizer, model, dataset,
                smoothing=0.0, mixup_function=None, max_norm=None, scheduler=None, gpu=True):
    model.train()
    if mixup_function is not None:
        loss_function = SoftTargetCrossEntropy()
    elif smoothing > 0.0:
        loss_function = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda() if gpu else loss_function

    loss_meter = meters.AverageMeter("loss")
    nll_meter = meters.AverageMeter("nll")
    l1_meter = meters.AverageMeter("l1")
    l2_meter = meters.AverageMeter("l2")

    for step, (xs, ys) in enumerate(dataset):
        if gpu:
            xs = xs.cuda()
            ys = ys.cuda()

        if mixup_function is not None:
            xs, ys = mixup_function(xs, ys)

        optimizer.zero_grad()
        logits = model(xs)
        loss = loss_function(logits, ys)

        nll_meter.update(loss.item())
        loss_meter.update(loss.item())

        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if scheduler:
            scheduler.step()

    l1_reg = norm.l1(model, gpu)
    l1_meter.update(l1_reg.item())

    l2_reg = norm.l2(model, gpu)
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
    count_bin, acc_bin, conf_bin, ece_value, ecse_value = metrics

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
    writer.add_scalar("test/ecse", ecse_value, global_step=epoch)
