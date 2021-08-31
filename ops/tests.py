import io
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

import ops.meters as meters


@torch.no_grad()
def test(model, n_ff, dataset,
         transform=None, smoothing=0.0,
         cutoffs=(0.0, 0.9), bins=np.linspace(0.0, 1.0, 11),
         verbose=False, period=10, gpu=True):
    model.eval()
    model = model.cuda() if gpu else model.cpu()
    xs, ys = next(iter(dataset))
    xs = xs.cuda() if gpu else xs.cpu()
    num_classes = model(xs).size()[-1]

    cm_shape = [num_classes, num_classes]
    cms = [[np.zeros(cm_shape), np.zeros(cm_shape)] for _ in range(len(cutoffs))]

    nll_meter = meters.AverageMeter("nll")
    brier_meter = meters.AverageMeter("brier")
    topk_meter = meters.AverageMeter("top5")

    cms_bin = [np.zeros(cm_shape) for _ in range(len(bins) - 1)]
    conf_acc_bin = [0.0 for _ in range(len(bins) - 1)]
    count_bin, acc_bin, conf_bin, metrics_str = [], [], [], []
    metrics = None

    for step, (xs, ys) in enumerate(dataset):
        if gpu:
            xs = xs.cuda()
            ys = ys.cuda()

        if transform is not None:
            xs, ys_t = transform(xs, ys)
        else:
            xs, ys_t = xs, ys

        if len(ys_t.shape) > 1:
            loss_function = SoftTargetCrossEntropy()
            ys = torch.max(ys_t, dim=-1)[1]
        elif smoothing > 0.0:
            loss_function = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda() if gpu else loss_function

        # A. Predict results
        ys_pred = torch.stack([F.softmax(model(xs), dim=1) for _ in range(n_ff)])
        ys_pred = torch.mean(ys_pred, dim=0)

        ys_t = ys_t.cpu()
        ys = ys.cpu()
        ys_pred = ys_pred.cpu()

        # B. Measure Confusion Matrices
        nll_meter.update(loss_function(torch.log(ys_pred), ys_t).numpy())
        topk_meter.update(topk(ys.numpy(), ys_pred.numpy()))
        brier_meter.update(brier(ys.numpy(), ys_pred.numpy()))

        for cutoff, cm_group in zip(cutoffs, cms):
            cm_certain = cm(ys.numpy(), ys_pred.numpy(), filter_min=cutoff)
            cm_uncertain = cm(ys.numpy(), ys_pred.numpy(), filter_max=cutoff)
            cm_group[0] = cm_group[0] + cm_certain
            cm_group[1] = cm_group[1] + cm_uncertain
        for i, (start, end) in enumerate(zip(bins, bins[1:])):
            cms_bin[i] = cms_bin[i] + cm(ys.numpy(), ys_pred.numpy(), filter_min=start, filter_max=end)
            confidence = np.amax(ys_pred.numpy(), axis=1)
            condition = np.logical_and(confidence >= start, confidence < end)
            conf_acc_bin[i] = conf_acc_bin[i] + np.sum(confidence[condition])

        nll_value = nll_meter.avg
        topk_value = topk_meter.avg
        brier_value = brier_meter.avg
        accs = [gacc(cm_certain) for cm_certain, cm_uncertain in cms]
        ious = [miou(cm_certain) for cm_certain, cm_uncertain in cms]
        uncs = [unconfidence(cm_certain, cm_uncertain) for cm_certain, cm_uncertain in cms]
        freqs = [frequency(cm_certain, cm_uncertain) for cm_certain, cm_uncertain in cms]
        count_bin = [np.sum(cm_bin) for cm_bin in cms_bin]
        acc_bin = [gacc(cm_bin) for cm_bin in cms_bin]
        conf_bin = [conf_acc / (count + 1e-7) for count, conf_acc in zip(count_bin, conf_acc_bin)]
        ece_value = ece(count_bin, acc_bin, conf_bin)
        ecse_value = ecse(count_bin, acc_bin, conf_bin)

        metrics = nll_value, \
                  cutoffs, cms, accs, uncs, ious, freqs, \
                  topk_value, brier_value, \
                  count_bin, acc_bin, conf_bin, ece_value, ecse_value
        if verbose and int(step + 1) % period == 0:
            print("%d Steps, %s" % (int(step + 1), repr_metrics(metrics)))

    print(repr_metrics(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    confidence_histogram(axes[0], count_bin)
    reliability_diagram(axes[1], acc_bin)
    fig.tight_layout()
    calibration_image = plot_to_image(fig)
    if not verbose:
        plt.close(fig)

    return (*metrics, calibration_image)


def repr_metrics(metrics):
    nll_value, \
    cutoffs, cms, accs, uncs, ious, freqs, \
    topk_value, brier_value, \
    count_bin, acc_bin, conf_bin, ece_value, ecse_value = metrics

    metrics_reprs = [
        "NLL: %.4f" % nll_value if nll_value > 0.01 else "NLL: %.4e" % nll_value,
        "Cutoffs: " + ", ".join(["%.1f %%" % (cutoff * 100) for cutoff in cutoffs]),
        "Accs: " + ", ".join(["%.3f %%" % (acc * 100) for acc in accs]),
        "Uncs: " + ", ".join(["%.3f %%" % (unc * 100) for unc in uncs]),
        "IoUs: " + ", ".join(["%.3f %%" % (iou * 100) for iou in ious]),
        "Freqs: " + ", ".join(["%.3f %%" % (freq * 100) for freq in freqs]),
        "Top-5: " + "%.3f %%" % (topk_value * 100),
        "Brier: " + "%.3f" % brier_value,
        "ECE: " + "%.3f %%" % (ece_value * 100),
        "ECE±: " + "%.3f %%" % (ecse_value * 100),
    ]

    return ", ".join(metrics_reprs)


@torch.no_grad()
def test_perturbation(dataset, model, n_ff):
    model = model.eval()
    model = model.cuda()

    cons_meter = meters.AverageMeter("cons")
    cec_meter = meters.AverageMeter("cec")
    for xs, ys in dataset:
        xs = xs.cuda()

        b, _, _, _, _ = xs.shape
        xs = xs.reshape([-1, 3, 32, 32])

        ys_pred = torch.stack([model(xs) for _ in range(n_ff)])
        ys_pred = torch.softmax(ys_pred, dim=-1)
        ys_pred = torch.mean(ys_pred, dim=0)

        xs = xs.reshape([b, -1, 3, 32, 32])
        ys_pred = ys_pred.reshape([b, -1, 10])

        # Consistency
        index = torch.argmax(ys_pred, dim=-1)
        cons = index[:, 1:] == index[:, :-1]
        cons = torch.mean(cons.float(), dim=-1)
        cons_meter.update(cons.cpu().numpy())

        # CEC
        cec = ys_pred[:, 1:] * torch.log(ys_pred[:, :-1])
        cec = - torch.mean(cec, dim=-1)
        cec_meter.update(cec.cpu().numpy())

    return cons_meter.avg, cec_meter.avg


@torch.no_grad()
def test_prediction_time(model, n_ff, input_size, n=100, gpu=True):
    model = model.eval()
    predict_times = meters.AverageMeter("predict_times", "%.3f")

    for _ in range(n):
        xs = torch.rand(input_size)
        xs = xs.cuda() if gpu else xs

        start_time = time.time()
        ys_pred = torch.stack([F.softmax(model(xs), dim=1) for _ in range(n_ff)])
        ys_pred = torch.mean(ys_pred, dim=0)
        torch.cuda.synchronize() if gpu else None
        predict_times.update(time.time() - start_time)

    print("Time: %.3f±%.3f ms" %
          (predict_times.avg * 1e3, predict_times.std * 1e3))

    return predict_times


def save_lists(metrics_dir, metrics_list):
    with open(metrics_dir, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for metrics in metrics_list:
            writer.writerow(metrics)


def save_metrics(metrics_dir, metrics_list):
    metrics_acc = []
    for metrics in metrics_list:
        *keys, \
        nll_value, \
        cutoffs, cms, accs, uncs, ious, freqs, \
        topk_value, brier_value, \
        count_bin, acc_bin, conf_bin, ece_value, ecse_value = metrics

        metrics_acc.append([
            *keys,
            nll_value, *cutoffs, *accs, *uncs, *ious, *freqs,
            topk_value, brier_value, ece_value, ecse_value
        ])

    save_lists(metrics_dir, metrics_acc)


def brier(ys, ys_pred):
    ys_onehot = np.eye(ys_pred.shape[1])[ys]
    return (np.square(ys_onehot - ys_pred)).sum(axis=1)


def topk(ys, ys_pred, k=5):
    ys_pred = ys_pred.argsort(axis=1)[:, -k:][:, ::-1]
    correct = np.logical_or.reduce(ys_pred == ys.reshape(-1, 1), axis=1)
    return correct


def cm(ys, ys_pred, filter_min=0.0, filter_max=1.0):
    """
    Confusion matrix.

    :param ys: numpy array [batch_size,]
    :param ys_pred: onehot numpy array [batch_size, num_classes]
    :param filter_min: lower bound of confidence
    :param filter_max: upper bound of confidence
    :return: cm for filtered predictions (shape: [num_classes, num_classes])
    """
    num_classes = ys_pred.shape[1]
    confidence = np.amax(ys_pred, axis=1)

    ys_pred = np.argmax(ys_pred, axis=1)
    condition = np.logical_and(confidence > filter_min, confidence <= filter_max)

    k = (ys >= 0) & (ys < num_classes) & condition
    cm = np.bincount(num_classes * ys[k] + ys_pred[k], minlength=num_classes ** 2)
    cm = np.reshape(cm, [num_classes, num_classes])

    return cm


def miou(cm):
    """
    Mean IoU
    """
    weights = np.sum(cm, axis=1)
    weights = [1 if weight > 0 else 0 for weight in weights]
    if np.sum(weights) > 0:
        _miou = np.average(ious(cm), weights=weights)
    else:
        _miou = 0.0
    return _miou


def ious(cm):
    """
    Intersection over unit w.r.t. classes.
    """
    num = np.diag(cm)
    den = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den != 0))


def gacc(cm):
    """
    Global accuracy p(accurate). For cm_certain, p(accurate|confident).
    """
    num = np.diag(cm).sum()
    den = np.sum(cm)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den != 0)).tolist()


def caccs(cm):
    """
    Accuracies w.r.t. classes.
    """
    accs = []
    for ii in range(np.shape(cm)[0]):
        if float(np.sum(cm, axis=1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(cm)[ii] / (float(np.sum(cm, axis=1)[ii]) + 1e-7)
        accs.append(acc)
    return accs


def unconfidence(cm_certain, cm_uncertain):
    """
    p(unconfident|inaccurate)
    """
    inaccurate_certain = np.sum(cm_certain) - np.diag(cm_certain).sum()
    inaccurate_uncertain = np.sum(cm_uncertain) - np.diag(cm_uncertain).sum()

    return inaccurate_uncertain / (inaccurate_certain + inaccurate_uncertain + 1e-7)


def frequency(cm_certain, cm_uncertain):
    return np.sum(cm_certain) / (np.sum(cm_certain) + np.sum(cm_uncertain) + 1e-7)


def ece(count_bin, acc_bin, conf_bin):
    count_bin = np.array(count_bin)
    acc_bin = np.array(acc_bin)
    conf_bin = np.array(conf_bin)
    freq = np.nan_to_num(count_bin / (sum(count_bin) + 1e-7))
    ece_result = np.sum(np.absolute(acc_bin - conf_bin) * freq)
    return ece_result


def ecse(count_bin, acc_bin, conf_bin):
    count_bin = np.array(count_bin)
    acc_bin = np.array(acc_bin)
    conf_bin = np.array(conf_bin)
    freq = np.nan_to_num(count_bin / (sum(count_bin) + 1e-7))
    ecse_result = np.sum((conf_bin - acc_bin) * freq)
    return ecse_result


def confidence_histogram(ax, count_bin):
    color, alpha = "tab:green", 0.8
    centers = np.linspace(0.05, 0.95, 10)
    count_bin = np.array(count_bin)
    freq = count_bin / (sum(count_bin) + 1e-7)

    ax.bar(centers * 100, freq * 100, width=10, color=color, edgecolor="black", alpha=alpha)
    ax.set_xlim(0, 100.0)
    ax.set_ylim(0, 100.0)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Frequency (%)")


def reliability_diagram(ax, accs_bins, colors="tab:red", mode=0):
    alpha, guideline_style = 0.8, (0, (1, 1))
    guides_x, guides_y = np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11)
    centers = np.linspace(0.05, 0.95, 10)
    accs_bins = np.array(accs_bins)
    accs_bins = np.expand_dims(accs_bins, axis=0) if len(accs_bins.shape) < 2 else accs_bins
    colors = [colors] if type(colors) is not list else colors
    colors = colors + [None] * (len(accs_bins) - len(colors))

    ax.plot(guides_x * 100, guides_y * 100, linestyle=guideline_style, color="black")
    for accs_bin, color in zip(accs_bins, colors):
        if mode == 0:
            ax.bar(centers * 100, accs_bin * 100, width=10, color=color, edgecolor="black", alpha=alpha)
        elif mode == 1:
            ax.plot(centers * 100, accs_bin * 100, color=color, marker="o", alpha=alpha)
        else:
            raise ValueError("Invalid mode %d." % mode)

    ax.set_xlim(0, 100.0)
    ax.set_ylim(0, 100.0)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Accuracy (%)")


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by "figure" to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)

    # Convert PNG buffer to TF image
    trans = transforms.ToTensor()
    image = buf.getvalue()
    image = Image.open(io.BytesIO(image))
    image = trans(image)

    return image

