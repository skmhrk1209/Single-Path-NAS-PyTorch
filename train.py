import torch
from torch import nn
from torch import distributed
from torch import optim
from torch import utils
from torch import backends
from torch import autograd
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from tensorboardX import SummaryWriter
from darts import *
from ops import *
from utils import *
import numpy as np
import skimage
import functools
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

    backends.cudnn.benchmark = True

    # python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main.py
    distributed.init_process_group(backend='mpi')

    with open(args.config) as file:
        config = apply_dict(Dict, json.load(file)).train
    config.update(vars(args))
    config.update(dict(
        world_size=distributed.get_world_size(),
        global_rank=distributed.get_rank(),
        device_count=torch.cuda.device_count(),
        local_rank=distributed.get_rank() % torch.cuda.device_count()
    ))
    print(f'config: {config}')

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    model = DARTS(
        operations=dict(
            super_sep_conv=functools.partial(SuperSeparableConv2d, kernel_sizes=[3, 5], padding=2),
            super_dil_conv=functools.partial(SuperDilatedConv2d, kernel_sizes=[3, 5], padding=4, dilation=2),
            avg_pool_3x3=functools.partial(AvgPool2d, kernel_size=3, padding=1),
            max_pool_3x3=functools.partial(MaxPool2d, kernel_size=3, padding=1),
            identity=functools.partial(Identity),
            zero=functools.partial(Zero)
        ),
        num_nodes=6,
        num_input_nodes=2,
        num_cells=20,
        reduction_cells=[6, 13],
        num_predecessors=2,
        num_channels=36,
        num_classes=10,
        drop_prob_fn=lambda epoch: config.drop_prob * epoch / config.num_epochs
    ).cuda()

    checkpoint = Dict(torch.load('checkpoints/search/epoch_49'))
    model.architecture.load_state_dict(checkpoint.architecture_state_dict)
    model.build_discrete_dag()
    model.build_discrete_network()
    model = model.cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    config.global_batch_size = config.local_batch_size * config.world_size
    config.network_lr = config.network_lr * config.global_batch_size / config.global_batch_denom
    config.network_lr_min = config.network_lr_min * config.global_batch_size / config.global_batch_denom

    network_optimizer = torch.optim.SGD(
        params=model.network.parameters(),
        lr=config.network_lr,
        momentum=config.network_momentum,
        weight_decay=config.network_weight_decay
    )

    # nn.parallel.DistributedDataParallel and apex.parallel.DistributedDataParallel don't support multiple backward passes.
    # This means `all_reduce` is executed when the first backward pass.
    # So, we manually reduce all gradients.
    # model = parallel.DistributedDataParallel(model, delay_allreduce=True)

    last_epoch = -1
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        model.network.load_state_dict(checkpoint.network_state_dict)
        model.architecture.load_state_dict(checkpoint.architecture_state_dict)
        network_optimizer.load_state_dict(checkpoint.network_optimizer_state_dict)
        last_epoch = checkpoint.last_epoch
        global_step = checkpoint.global_step

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=network_optimizer,
        T_max=config.num_epochs,
        eta_min=config.network_lr_min,
        last_epoch=last_epoch
    )

    train_dataset = datasets.CIFAR10(
        root='cifar10',
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768)
            ),
            Cutout(size=(16, 16))
        ]),
        download=True
    )
    val_dataset = datasets.CIFAR10(
        root='cifar10',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768)
            )
        ]),
        download=True
    )

    train_sampler = utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

    train_data_loader = utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    if config.global_rank == 0:
        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)
        summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        for epoch in range(last_epoch + 1, config.num_epochs):

            train_sampler.set_epoch(epoch)
            model.set_epoch(epoch)

            model.train()

            for local_step, (train_images, train_labels) in enumerate(train_data_loader):

                step_begin = time.time()

                train_images = train_images.cuda(non_blocking=True)
                train_labels = train_labels.cuda(non_blocking=True)

                # Update network parameter.
                # ----------------------------------------------------------------
                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) / config.world_size

                network_optimizer.zero_grad()

                train_loss.backward()

                for parameter in model.network.parameters():
                    distributed.all_reduce(parameter.grad)
                network_optimizer.step()
                # ----------------------------------------------------------------

                train_predictions = torch.argmax(train_logits, dim=1)
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
                    network_state_dict=model.network.state_dict(),
                    architecture_state_dict=model.architecture.state_dict(),
                    network_optimizer_state_dict=network_optimizer.state_dict(),
                    last_epoch=epoch,
                    global_step=global_step
                ), f'{config.checkpoint_directory}/epoch_{epoch}')

            lr_scheduler.step()

            if config.validation:

                model.eval()

                with torch.no_grad():

                    average_loss = 0
                    average_accuracy = 0

                    for local_step, (images, labels) in enumerate(val_data_loader):

                        images = images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)

                        logits = model(images)
                        loss = criterion(logits, labels) / config.world_size

                        predictions = torch.argmax(logits, dim=1)
                        accuracy = torch.mean((predictions == labels).float()) / config.world_size

                        for tensor in [loss, accuracy]:
                            distributed.all_reduce(tensor)

                        average_loss += loss
                        average_accuracy += accuracy

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

            for local_step, (images, labels) in enumerate(val_data_loader):

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels) / config.world_size

                predictions = torch.argmax(logits, dim=1)
                accuracy = torch.mean((predictions == labels).float()) / config.world_size

                for tensor in [loss, accuracy]:
                    distributed.all_reduce(tensor)

                average_loss += loss
                average_accuracy += accuracy

            average_loss /= (local_step + 1)
            average_accuracy /= (local_step + 1)

        if config.global_rank == 0:
            print(f'[validation] epoch: {last_epoch} loss: {average_loss:.4f} accuracy: {average_accuracy:.4f}')

    if config.global_rank == 0:
        summary_writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DARTS: Differentiable Architecture Search')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    main(args)
