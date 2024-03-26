from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineAnnealingLR(_LRScheduler):
    """
        This scheduler adjusts the learning rate using a cosine annealing schedule with warmup.
        During warmup, the learning rate linearly increases to its maximum value. After warmup, it follows
        a cosine decay schedule to decrease the learning rate to its minimum value.

        Parameters:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to increase the learning rate from `warmup_start_lr` to `max_lr`.
        total_epochs (int): Total number of epochs including both warmup and cosine annealing phases.
        warmup_start_lr (float): Initial learning rate during the warmup phase.
        max_lr (float): Maximum learning rate after warmup.
        last_epoch (int): The index of the last epoch. Default: -1.
        """

    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        self.end_lr = warmup_start_lr / 100
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = (self.max_lr - self.warmup_start_lr) / self.warmup_epochs * self.last_epoch + self.warmup_start_lr
        else:
            cos_out = math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)) + 1
            lr = (self.max_lr - self.end_lr) / 2 * cos_out + self.end_lr
        return [lr for group in self.optimizer.param_groups]
