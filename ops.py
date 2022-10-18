import torch
from torch import nn


class SuperConv2d(nn.Module):

    def __init__(self, in_channels, out_channels=None, out_channels_list=[], kernel_size=None,  kernel_size_list=[],
                 padding=0, stride=1, dilation=1, groups=1, bias=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        max_out_channels = max(out_channels_list) if out_channels_list else out_channels
        max_kernel_size = max(kernel_size_list) if kernel_size_list else kernel_size

        channel_masks = []
        prev_out_channels = None
        for out_channels in out_channels_list:
            channel_mask = torch.ones(max_out_channels)
            channel_mask *= nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            if prev_out_channels:
                channel_mask *= nn.functional.pad(torch.zeros(prev_out_channels), [0, max_out_channels - prev_out_channels], value=1)
            channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            prev_out_channels = out_channels
            channel_masks.append(channel_mask)

        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if out_channels_list else None)
        self.register_parameter('channel_thresholds', nn.Parameter(torch.zeros(len(out_channels_list))) if out_channels_list else None)

        kernel_masks = []
        prev_kernel_size = None
        for kernel_size in kernel_size_list:
            kernel_mask = torch.ones(max_kernel_size, max_kernel_size)
            kernel_mask *= nn.functional.pad(torch.ones(kernel_size, kernel_size), [(max_kernel_size - kernel_size) // 2] * 4, value=0)
            if prev_kernel_size:
                kernel_mask *= nn.functional.pad(torch.zeros(prev_kernel_size, prev_kernel_size), [(max_kernel_size - prev_kernel_size) // 2] * 4, value=1)
            kernel_mask = kernel_mask.unsqueeze(0).unsqueeze(0)
            prev_kernel_size = kernel_size
            kernel_masks.append(kernel_mask)

        self.register_buffer('kernel_masks', torch.stack(kernel_masks, dim=0) if kernel_size_list else None)
        self.register_parameter('kernel_thresholds', nn.Parameter(torch.zeros(len(kernel_size_list))) if kernel_size_list else None)

        self.register_parameter('weight', nn.Parameter(torch.Tensor(max_out_channels, in_channels // groups, max_kernel_size, max_kernel_size)))
        self.register_parameter('bias', nn.Parameter(torch.Tensor(max_out_channels)) if bias else None)

        self.max_out_channels = max_out_channels
        self.max_kernel_size = max_kernel_size

    def forward(self, input):
        weight = self.weight
        if self.channel_masks is not None and self.channel_thresholds is not None:
            weight = weight * self.parametrized_mask(list(self.channel_masks), list(self.channel_thresholds))
        if self.kernel_masks is not None and self.kernel_thresholds is not None:
            weight = weight * self.parametrized_mask(list(self.kernel_masks), list(self.kernel_thresholds))
        return nn.functional.conv2d(input, weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

    def parametrized_mask(self, masks, thresholds):
        if not masks or not thresholds:
            return 0
        mask = masks.pop(0)
        threshold = thresholds.pop(0)
        norm = torch.norm(self.weight * mask)
        indicator = (norm > threshold).float() - torch.sigmoid(norm - threshold).detach() + torch.sigmoid(norm - threshold)
        return indicator * (mask + self.parametrized_mask(masks, thresholds))

    def freeze_weight(self):
        weight = self.weight
        if self.channel_masks is not None and self.channel_thresholds is not None:
            prev_out_channels = None
            for channel_mask, channel_threshold, out_channels in zip(self.channel_masks, self.channel_thresholds, self.out_channels_list):
                if prev_out_channels:
                    channel_norm = torch.norm(self.weight * channel_mask)
                    if channel_norm < channel_threshold:
                        weight = weight[:, :prev_out_channels, ...]
                        break
                prev_out_channels = out_channels
        if self.kernel_masks is not None and self.kernel_thresholds is not None:
            prev_kernel_size = None
            for kernel_mask, kernel_threshold, kernel_size in zip(self.kernel_masks, self.kernel_thresholds, self.kernel_size_list):
                if prev_kernel_size:
                    kernel_norm = torch.norm(self.weight * kernel_mask)
                    if kernel_norm < kernel_threshold:
                        cut = (self.max_kernel_size - prev_kernel_size) // 2
                        weight = weight[..., cut:-cut, cut:-cut]
                        break
                prev_kernel_size = kernel_size
        self.weight = weight


class ScheduledDropPath(nn.Module):

    def __init__(self, drop_prob_fn):
        super().__init__()
        self.drop_prob_fn = drop_prob_fn

    def forward(self, inputs):
        drop_prob = self.drop_prob_fn(self.epoch)
        if self.training and drop_prob > 0:
            keep_prob = 1 - drop_prob
            masks = inputs.new_full((inputs.size(0), 1, 1, 1), keep_prob).bernoulli()
            inputs = inputs * masks
            inputs = inputs / keep_prob
        return inputs

    def set_epoch(self, epoch):
        self.epoch = epoch


class Cutout(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, inputs):
        y_min = torch.randint(inputs.size(-2) - self.size[-2], (1,))
        x_min = torch.randint(inputs.size(-1) - self.size[-1], (1,))
        y_max = y_min + self.size[-2]
        x_max = x_min + self.size[-1]
        inputs[..., y_min:y_max, x_min:x_max] = 0
        return inputs


class CrossEntropyLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_prob = nn.functional.log_softmax(inputs, dim=1)
        targets = torch.zeros_like(log_prob).scatter(dim=1, index=targets.unsqueeze(1), value=1)
        targets = (1 - self.smoothing) * targets + self.smoothing / (targets.size(1) - 1) * (1 - targets)
        loss = -torch.sum(targets * log_prob, dim=1).mean()
        return loss
