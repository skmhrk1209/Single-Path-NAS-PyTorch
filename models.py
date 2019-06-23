import torch
from torch import nn
from ops import *


class MobileConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):

        super().__init__()

        hidden_channels = in_channels * expand_ratio

        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    groups=hidden_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            ),
        )

    def forward(self, input):
        output = self.module(input)
        if input.shape == output.shape:
            output += input
        return output


class SuperMobileConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio_list, kernel_size_list, stride):

        super().__init__()

        hidden_channels_list = [in_channels * expand_ratio for expand_ratio in expand_ratio_list]
        max_hidden_channels = max(hidden_channels_list)
        max_kernel_size = max(kernel_size_list)

        self.module = nn.Sequential(
            nn.Sequential(
                SuperConv2d(
                    in_channels=in_channels,
                    out_channels_list=hidden_channels_list,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(max_hidden_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                SuperConv2d(
                    in_channels=max_hidden_channels,
                    out_channels=max_hidden_channels,
                    groups=max_hidden_channels,
                    kernel_size_list=kernel_size_list,
                    padding=(max_kernel_size - 1) // 2,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(max_hidden_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=max_hidden_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            ),
        )

    def forward(self, input):
        output = self.module(input)
        if input.shape == output.shape:
            output += input
        return output


class MobileNetV2(nn.Module):

    def __init__(self, first_conv_param, middle_conv_params, last_conv_param, num_classes, drop_prob=0.2):

        super().__init__()

        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=first_conv_param.in_channels,
                    out_channels=first_conv_param.out_channels,
                    kernel_size=first_conv_param.kernel_size,
                    padding=(first_conv_param.kernel_size - 1) // 2,
                    stride=first_conv_param.stride,
                    bias=False
                ),
                nn.BatchNorm2d(first_conv_param.out_channels),
                nn.ReLU6()
            ),
            nn.Sequential(*[
                nn.Sequential(*[
                    MobileConvBlock(
                        in_channels=middle_conv_param.out_channels if i else middle_conv_param.in_channels,
                        out_channels=middle_conv_param.out_channels,
                        expand_ratio=middle_conv_param.expand_ratio,
                        kernel_size=middle_conv_param.kernel_size,
                        stride=1 if i else middle_conv_param.stride
                    ) for i in range(middle_conv_param.blocks)
                ]) for middle_conv_param in middle_conv_params
            ]),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_conv_param.in_channels,
                    out_channels=last_conv_param.out_channels,
                    kernel_size=last_conv_param.kernel_size,
                    padding=(last_conv_param.kernel_size - 1) // 2,
                    stride=last_conv_param.stride,
                    bias=False
                ),
                nn.BatchNorm2d(last_conv_param.out_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    in_channels=last_conv_param.out_channels,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=True
                )
            )
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, SuperConv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input):
        return self.module(input).squeeze()


class SuperMobileNetV2(nn.Module):

    def __init__(self, first_conv_param, middle_conv_params, last_conv_param, num_classes, drop_prob=0.2):

        super().__init__()

        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=first_conv_param.in_channels,
                    out_channels=first_conv_param.out_channels,
                    kernel_size=first_conv_param.kernel_size,
                    padding=(first_conv_param.kernel_size - 1) // 2,
                    stride=first_conv_param.stride,
                    bias=False
                ),
                nn.BatchNorm2d(first_conv_param.out_channels),
                nn.ReLU6()
            ),
            nn.Sequential(*[
                nn.Sequential(*[
                    SuperMobileConvBlock(
                        in_channels=middle_conv_param.out_channels if i else middle_conv_param.in_channels,
                        out_channels=middle_conv_param.out_channels,
                        expand_ratio_list=middle_conv_param.expand_ratio_list,
                        kernel_size_list=middle_conv_param.kernel_size_list,
                        stride=1 if i else middle_conv_param.stride
                    ) for i in range(middle_conv_param.blocks)
                ]) for middle_conv_param in middle_conv_params
            ]),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_conv_param.in_channels,
                    out_channels=last_conv_param.out_channels,
                    kernel_size=last_conv_param.kernel_size,
                    padding=(last_conv_param.kernel_size - 1) // 2,
                    stride=last_conv_param.stride,
                    bias=False
                ),
                nn.BatchNorm2d(last_conv_param.out_channels),
                nn.ReLU6()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Dropout(drop_prob),
                nn.Conv2d(
                    in_channels=last_conv_param.out_channels,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=True
                )
            )
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, SuperConv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input):
        return self.module(input).squeeze()

    def weights(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield parameter

    def named_weights(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield name, parameter

    def weight_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield parameter.grad

    def named_weight_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' not in name:
                yield name, parameter.grad

    def thresholds(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield parameter

    def named_thresholds(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield name, parameter

    def threshold_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield parameter.grad

    def named_threshold_gradients(self):
        for name, parameter in self.named_parameters():
            if 'threshold' in name:
                yield name, parameter.grad
