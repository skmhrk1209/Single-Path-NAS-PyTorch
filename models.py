import copy

import torch
from torch import nn
from torch import distributed
from ops import SuperConv2d


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

    def train_thresholds(self, val_images, val_labels, threshold_optimizer, criterion, config):

        val_logits = self(val_images)
        val_loss = criterion(val_logits, val_labels) / config.world_size

        threshold_optimizer.zero_grad()

        val_loss.backward()

        for threshold in self.thresholds():
            distributed.all_reduce(threshold.grad)

        threshold_optimizer.step()

        return val_logits, val_loss

    def train_thresholds_bilevel(self, train_images, train_targets, val_images, val_targets, weight_optimizer, threshold_optimizer, criterion, config):

        # Save current network parameters and optimizer.
        named_weights = copy.deepcopy(list(self.named_weights()))
        named_buffers = copy.deepcopy(list(self.named_buffers()))
        weight_optimizer_state_dict = copy.deepcopy(weight_optimizer.state_dict())

        # Approximate w*(α) by adapting w using only a single training step,
        # without solving the inner optimization completely by training until convergence.
        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) / config.world_size

        weight_optimizer.zero_grad()

        train_loss.backward()

        for weight in self.weights():
            distributed.all_reduce(weight.grad)

        weight_optimizer.step()

        # Apply chain rule to the approximate architecture gradient.
        # Backward validation loss, but don't update approximate parameter w'.
        val_logits = self(val_images)
        val_loss = criterion(val_logits, val_targets) / config.world_size

        weight_optimizer.zero_grad()
        threshold_optimizer.zero_grad()

        val_loss.backward()

        named_weight_gradients = copy.deepcopy([(name, weight.grad) for name, weight in self.named_weights()])
        weight_gradient_norm = torch.norm(torch.cat([weight_gradient.reshape(-1) for name, weight_gradient in named_weight_gradients]))

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for weight, (_, prev_weight), (_, prev_weight_gradient) in zip(self.weights(), named_weights, named_weight_gradients):
            weight.data = (prev_weight + prev_weight_gradient * config.epsilon / weight_gradient_norm).data

        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) * -(config.weight_optimizer.lr / (2 * config.epsilon / weight_gradient_norm)) / config.world_size

        train_loss.backward()

        # Avoid calculate hessian-vector product using the finite difference approximation.
        for weight, (_, prev_weight), (_, prev_weight_gradient) in zip(self.weights(), named_weights, named_weight_gradients):
            weight.data = (prev_weight - prev_weight_gradient * config.epsilon / weight_gradient_norm).data

        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) * (config.weight_optimizer.lr / (2 * config.epsilon / weight_gradient_norm)) / config.world_size

        train_loss.backward()

        # Finally, update architecture parameter.
        for threshold in self.thresholds():
            distributed.all_reduce(threshold.grad)

        threshold_optimizer.step()

        # Restore previous network parameters and optimizer.
        self.load_state_dict(dict(**dict(named_weights), **dict(named_buffers)), strict=True)
        weight_optimizer.load_state_dict(weight_optimizer_state_dict)

        return val_logits, val_loss

    def train_weights(self, train_images, train_targets, weight_optimizer, criterion, config):

        train_logits = self(train_images)
        train_loss = criterion(train_logits, train_targets) / config.world_size

        weight_optimizer.zero_grad()

        train_loss.backward()

        for weight in self.weights():
            distributed.all_reduce(weight.grad)

        weight_optimizer.step()

        return train_logits, train_loss

    def freeze_weight(self):
        for module in self.modules():
            if isinstance(module, SuperConv2d):
                module.freeze_weight()

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
