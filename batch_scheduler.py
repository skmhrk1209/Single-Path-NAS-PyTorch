import torch
from torch import optim
import math


class LambdaBatch(object):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, batch_sampler, batch_lambda, last_epoch=-1):
        self.batch_sampler = batch_sampler
        self.batch_lambda = batch_lambda
        self.epoch = last_epoch + 1

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1
        self.batch_sampler.batch_size *= self.batch_lambda(self.epoch)


class StepBatch(object):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, batch_sampler, step_size, gamma, last_epoch=-1):
        self.batch_sampler = batch_sampler
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = last_epoch + 1

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1
        if self.epoch and self.epoch % self.step_size == 0:
            self.batch_sampler.batch_size *= self.gamma


class MultiStepBatch(object):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, batch_sampler, milestones, gamma, last_epoch=-1):
        self.batch_sampler = batch_sampler
        self.milestones = milestones
        self.gamma = gamma
        self.epoch = last_epoch + 1

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1
        if self.epoch in self.milestones:
            self.batch_sampler.batch_size *= self.gamma


class ExponentialBatch(object):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, batch_sampler, gamma, last_epoch=-1):
        self.batch_sampler = batch_sampler
        self.gamma = gamma
        self.epoch = last_epoch + 1

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1
        self.batch_sampler.batch_size *= self.gamma


class CosineAnnealingBatch(object):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_{t+1} = \eta_{min} + (\eta_t - \eta_{min})\frac{1 +
        \cos(\frac{T_{cur+1}}{T_{max}}\pi)}{1 + \cos(\frac{T_{cur}}{T_{max}}\pi)},
        T_{cur} \neq (2k+1)T_{max};\\
        \eta_{t+1} = \eta_{t} + (\eta_{max} - \eta_{min})\frac{1 -
        \cos(\frac{1}{T_{max}}\pi)}{2},
        T_{cur} = (2k+1)T_{max}.\\

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, batch_sampler, T_max, batch_max, last_epoch=-1):
        self.batch_sampler = batch_sampler
        self.T_max = T_max
        self.batch_min = batch_sampler.batch_size
        self.batch_max = batch_max
        self.epoch = last_epoch + 1

    def step(self, epoch=None):
        self.epoch = epoch or self.epoch + 1
        self.batch_sampler.batch_size = self.batch_min + (self.batch_max - self.batch_min) * (1 + math.cos((self.epoch / self.T_max - 1) * math.pi)) / 2
