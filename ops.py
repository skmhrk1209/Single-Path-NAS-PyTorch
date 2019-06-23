import torch
from torch import nn


class SuperConv2d(nn.Module):

    def __init__(self, in_channels, out_channels=None, out_channels_list=[], kernel_size=None,  kernel_size_list=[],
                 padding=0, stride=1, dilation=1, groups=1, bias=True):

        super().__init__()

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

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

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
