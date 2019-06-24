import torch
from torch import nn
from torch import optim
from torch import utils
from torch import backends
from torch import distributed
from torchvision import transforms
from tensorboardX import SummaryWriter
from models import *
from datasets import *
from utils import *
import numpy as np
import argparse
import copy
import json
import time
import os


def apply_dict(function, dictionary):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            dictionary[key] = apply_dict(function, value)
        dictionary = function(dictionary)
    return dictionary


def main(args):

    backends.cudnn.fastest = True
    backends.cudnn.benchmark = True

    distributed.init_process_group(backend='nccl')

    with open(args.config) as file:
        config = apply_dict(Dict, json.load(file))
    config.update(vars(args))
    config.update(dict(
        world_size=distributed.get_world_size(),
        global_rank=distributed.get_rank(),
        device_count=torch.cuda.device_count(),
        local_rank=distributed.get_rank() % torch.cuda.device_count()
    ))
    print(f'config: {config}')

    train_dataset = ImageNet(
        root=config.train_root,
        meta=config.train_meta,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    )
    train_datasets = [
        utils.data.Subset(train_dataset, indices)
        for indices in np.array_split(range(len(train_dataset)), 2)
    ]
    val_dataset = ImageNet(
        root=config.val_root,
        meta=config.val_meta,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    )

    train_samplers = [
        utils.data.distributed.DistributedSampler(train_dataset)
        for train_dataset in train_datasets
    ]
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_data_loaders = [
        utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.local_batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        ) for train_dataset, train_sampler in zip(train_datasets, train_samplers)
    ]
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    model = SuperMobileNetV2(
        first_conv_param=Dict(in_channels=3, out_channels=32, kernel_size=3, stride=2),
        middle_conv_params=[
            Dict(in_channels=32, out_channels=16, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=1, stride=1),
            Dict(in_channels=16, out_channels=24, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=2, stride=2),
            Dict(in_channels=24, out_channels=32, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=3, stride=2),
            Dict(in_channels=32, out_channels=64, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=4, stride=2),
            Dict(in_channels=64, out_channels=96, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=3, stride=1),
            Dict(in_channels=96, out_channels=160, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=3, stride=2),
            Dict(in_channels=160, out_channels=320, expand_ratio_list=[3, 6], kernel_size_list=[3, 5], blocks=1, stride=1),
        ],
        last_conv_param=Dict(in_channels=320, out_channels=1280, kernel_size=1, stride=1),
        drop_prob=config.drop_prob,
        num_classes=1000
    ).cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    config.global_batch_size = config.local_batch_size * config.world_size
    config.weight_optimizer.lr = config.weight_optimizer.lr * config.global_batch_size / config.global_batch_denom
    config.threshold_optimizer.lr = config.threshold_optimizer.lr * config.global_batch_size / config.global_batch_denom

    weight_optimizer = torch.optim.RMSprop(
        params=model.weights(),
        lr=config.weight_optimizer.lr,
        alpha=config.weight_optimizer.alpha,
        eps=config.weight_optimizer.eps,
        weight_decay=config.weight_optimizer.weight_decay,
        momentum=config.weight_optimizer.momentum
    )

    threshold_optimizer = torch.optim.Adam(
        params=model.thresholds(),
        lr=config.threshold_optimizer.lr,
        betas=config.threshold_optimizer.betas,
        eps=config.threshold_optimizer.eps,
        weight_decay=config.threshold_optimizer.weight_decay
    )

    last_epoch = -1
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        model.load_state_dict(checkpoint.model_state_dict)
        weight_optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        last_epoch = checkpoint.last_epoch
        global_step = checkpoint.global_step

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=weight_optimizer,
        T_max=config.num_epochs,
        last_epoch=last_epoch
    )

    if config.global_rank == 0:
        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)
        summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        for epoch in range(last_epoch + 1, config.num_epochs):

            for train_sampler in train_samplers:
                train_sampler.set_epoch(epoch)
            lr_scheduler.step(epoch)

            model.train()

            for local_step, ((train_images, train_labels), (val_images, val_labels)) in enumerate(zip(*train_data_loaders)):

                step_begin = time.time()

                train_images = train_images.cuda()
                train_labels = train_labels.cuda()

                val_images = val_images.cuda()
                val_labels = val_labels.cuda()

                # Save current network parameters and optimizer.
                named_weights = copy.deepcopy(list(model.named_weights()))
                named_buffers = copy.deepcopy(list(model.named_buffers()))
                weight_optimizer_state_dict = copy.deepcopy(weight_optimizer.state_dict())

                # Approximate w*(α) by adapting w using only a single training step,
                # without solving the inner optimization completely by training until convergence.
                # ----------------------------------------------------------------
                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) / config.world_size

                weight_optimizer.zero_grad()
                threshold_optimizer.zero_grad()

                train_loss.backward()

                for weight in model.weights():
                    distributed.all_reduce(weight.grad)

                weight_optimizer.step()
                # ----------------------------------------------------------------

                # Apply chain rule to the approximate architecture gradient.
                # Backward validation loss, but don't update approximate parameter w'.
                # ----------------------------------------------------------------
                val_logits = model(val_images)
                val_loss = criterion(val_logits, val_labels) / config.world_size

                weight_optimizer.zero_grad()
                threshold_optimizer.zero_grad()

                val_loss.backward()

                named_weight_gradients = copy.deepcopy(list(model.named_weight_gradients()))
                weight_gradient_norm = torch.norm(torch.cat([weight_gradient.reshape(-1) for name, weight_gradient in named_weight_gradients]))
                # ----------------------------------------------------------------

                # Avoid calculate hessian-vector product using the finite difference approximation.
                # ----------------------------------------------------------------
                for weight, (name, prev_weight), (name, prev_weight_gradient) in zip(model.weights(), named_weights, named_weight_gradients):
                    weight.data = (prev_weight + prev_weight_gradient * config.epsilon / weight_gradient_norm).data

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) * -(config.weight_optimizer.lr / (2 * config.epsilon / weight_gradient_norm)) / config.world_size

                train_loss.backward()

                for weight, (name, prev_weight), (name, prev_weight_gradient) in zip(model.weights(), named_weights, named_weight_gradients):
                    weight.data = (prev_weight - prev_weight_gradient * config.epsilon / weight_gradient_norm).data

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) * (config.weight_optimizer.lr / (2 * config.epsilon / weight_gradient_norm)) / config.world_size

                train_loss.backward()
                # ----------------------------------------------------------------
                # Finally, update architecture parameter.
                for threshold in model.thresholds():
                    distributed.all_reduce(threshold.grad)

                threshold_optimizer.step()
                # ----------------------------------------------------------------

                # Restore previous network parameters and optimizer.
                model.load_state_dict(dict(**dict(named_weights), **dict(named_buffers)), strict=False)
                weight_optimizer.load_state_dict(weight_optimizer_state_dict)

                # Update network parameter.
                # ----------------------------------------------------------------
                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) / config.world_size

                weight_optimizer.zero_grad()

                train_loss.backward()

                for weight in model.weights():
                    distributed.all_reduce(weight.grad)

                weight_optimizer.step()

                train_predictions = train_logits.topk(1)[1].squeeze()
                train_accuracy = torch.mean((train_predictions == train_labels).float()) / config.world_size

                for tensor in [train_loss, train_accuracy]:
                    distributed.all_reduce(tensor)

                step_end = time.time()

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(train=train_loss),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(train=train_accuracy),
                        global_step=global_step
                    )
                    print(f'[training] epoch: {epoch} global_step: {global_step} local_step: {local_step} '
                          f'loss: {train_loss:.4f} accuracy: {train_accuracy:.4f} [{step_end - step_begin:.4f}s]')

                global_step += 1

            if config.global_rank == 0:
                torch.save(dict(
                    model_state_dict=model.state_dict(),
                    weight_optimizer_state_dict=weight_optimizer.state_dict(),
                    threshold_optimizer_state_dict=threshold_optimizer.state_dict(),
                    last_epoch=epoch,
                    global_step=global_step
                ), f'{config.checkpoint_directory}/epoch_{epoch}')

            if config.validation:

                model.eval()

                with torch.no_grad():

                    average_loss = 0
                    average_accuracy = 0

                    for local_step, (val_images, val_labels) in enumerate(val_data_loader):

                        val_images = val_images.cuda()
                        val_labels = val_labels.cuda()

                        val_logits = model(val_images)
                        val_loss = criterion(val_logits, val_labels) / config.world_size

                        val_predictions = val_logits.topk(1)[1].squeeze()
                        val_accuracy = torch.mean((val_predictions == val_labels).float()) / config.world_size

                        for tensor in [val_loss, val_accuracy]:
                            distributed.all_reduce(tensor)

                        average_loss += val_loss
                        average_accuracy += val_accuracy

                    average_loss /= (local_step + 1)
                    average_accuracy /= (local_step + 1)

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(val=average_loss),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(val=average_accuracy),
                        global_step=global_step
                    )
                    print(f'[validation] epoch: {epoch} loss: {average_loss:.4f} accuracy: {average_accuracy:.4f}')

    elif config.validation:

        model.eval()

        with torch.no_grad():

            average_loss = 0
            average_accuracy = 0

            for local_step, (val_images, val_labels) in enumerate(val_data_loader):

                val_images = val_images.cuda()
                val_labels = val_labels.cuda()

                val_logits = model(val_images)
                val_loss = criterion(val_logits, val_labels) / config.world_size

                val_predictions = val_logits.topk(1)[1].squeeze()
                val_accuracy = torch.mean((val_predictions == val_labels).float()) / config.world_size

                for tensor in [val_loss, val_accuracy]:
                    distributed.all_reduce(tensor)

                average_loss += val_loss
                average_accuracy += val_accuracy

            average_loss /= (local_step + 1)
            average_accuracy /= (local_step + 1)

        if config.global_rank == 0:
            print(f'[validation] epoch: {last_epoch} loss: {average_loss:.4f} accuracy: {average_accuracy:.4f}')

    if config.global_rank == 0:
        summary_writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Single-Path-NAS')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
