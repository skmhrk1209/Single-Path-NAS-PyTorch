import torch
from torch import nn
import numpy as np


class SuperConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, padding=0, stride=1, dilation=1, groups=1, bias=True):

        super().__init__()

        max_kernel_size = max(kernel_sizes)

        self.super_weight = nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(out_channels, in_channels // groups, max_kernel_size, max_kernel_size)))
        self.bias = nn.init.zeros_(nn.Parameter(torch.Tensor(out_channels))) if bias else None
        self.thresholds = nn.init.zeros_(nn.Parameter(torch.Tensor(len(kernel_sizes))))

        masks = []
        for i, kernel_size in enumerate(kernel_sizes):
            mask = torch.ones(1, 1, max_kernel_size, max_kernel_size)
            mask *= nn.functional.pad(torch.ones(1, 1, kernel_size, kernel_size), [(max_kernel_size - kernel_size) // 2] * 4, value=0)
            mask *= nn.functional.pad(torch.zeros(1, 1, prev_kernel_size, prev_kernel_size), [(max_kernel_size - prev_kernel_size) // 2] * 4, value=1) if i else 1
            prev_kernel_size = kernel_size
            masks.append(mask)
        self.register_buffer('masks', torch.stack(masks, dim=0))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.freezed = False

    def forward(self, input):

        super_weight = torch.zeros_like(self.super_weight)
        for i, (mask, threshold) in enumerate(zip(self.masks, self.thresholds)):
            weight = self.super_weight * mask
            norm = torch.norm(weight)
            indicator = (norm > threshold) - torch.sigmoid(norm - threshold).detach() + torch.sigmoid(norm - threshold)
            super_weight += (indicator if i else 1) * weight

        return nn.functional.conv2d(
            input=input,
            weight=super_weight,
            bias=self.bias,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups
        )


class SuperDilatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride, dilation, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            SuperConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_sizes=kernel_sizes,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class SuperSeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, padding, stride,
                 affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            SuperConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_sizes=kernel_sizes,
                padding=padding,
                stride=stride,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                affine=affine
            ),
            nn.ReLU(),
            SuperConv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_sizes=kernel_sizes,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class DilatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                affine=affine
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class AvgPool2d(nn.Module):

    def __init__(self, kernel_size, padding, stride, **kwargs):
        super().__init__()
        self.module = nn.AvgPool2d(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, input):
        return self.module(input)


class MaxPool2d(nn.Module):

    def __init__(self, kernel_size, padding, stride, **kwargs):
        super().__init__()
        self.module = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, input):
        return self.module(input)


class Identity(nn.Module):

    def __init__(self, in_channels, out_channels, stride, affine, preactivation=True, **kwargs):
        super().__init__()
        self.module = nn.Identity() if stride == 1 and in_channels == out_channels else Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            affine=affine,
            preactivation=preactivation
        )

    def forward(self, input):
        return self.module(input)


class Zero(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input):
        return 0.0


class ScheduledDropPath(nn.Module):

    def __init__(self, drop_prob_fn):
        super().__init__()
        self.drop_prob_fn = drop_prob_fn

    def forward(self, input):
        drop_prob = self.drop_prob_fn(self.epoch)
        if self.training and drop_prob > 0:
            keep_prob = 1 - drop_prob
            mask = input.new_full((input.size(0), 1, 1, 1), keep_prob).bernoulli()
            input = input * mask
            input = input / keep_prob
        return input

    def set_epoch(self, epoch):
        self.epoch = epoch


class Cutout(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        y_min = torch.randint(input.size(-2) - self.size[-2], (1,))
        x_min = torch.randint(input.size(-1) - self.size[-1], (1,))
        y_max = y_min + self.size[-2]
        x_max = x_min + self.size[-1]
        input[..., y_min:y_max, x_min:x_max] = 0
        return input
