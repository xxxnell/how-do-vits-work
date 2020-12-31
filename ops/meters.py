import math
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and std
    """

    def __init__(self, name, fmt=".3f"):
        self.name = name
        self.fmt = fmt

        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.sqsum = 0.0
        self.std = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.sqsum = 0.0
        self.std = 0.0

    def update(self, xs, n=1):
        if isinstance(xs, int) or isinstance(xs, float):
            xs = np.array([xs] * n)

        self.sum += np.sum(xs)
        self.sqsum += np.sum(np.square(xs))
        self.count += np.array(xs).size

        self.avg = self.sum / self.count
        self.std = math.sqrt(self.sqsum / self.count - self.avg * self.avg)

    def result(self):
        return self.avg

    def __str__(self):
        fmtstr = "AverageMeter(%s, %" + self.fmt + "±%" + self.fmt + ")"
        return fmtstr % (self.name, self.avg, self.std)
