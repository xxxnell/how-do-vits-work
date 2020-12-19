from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, iters, last_epoch=-1):
        self.iters = iters
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iters > 0:
            lr = [base_lr * self.last_epoch / self.iters for base_lr in self.base_lrs]
        else:
            lr = self.base_lrs

        return lr
