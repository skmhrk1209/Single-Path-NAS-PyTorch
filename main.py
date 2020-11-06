import os
import argparse
import shutil
import json
import time

import torch
import numpy as np

from torch import optim
from torch import cuda
from torch import backends
from torch import utils
from torch import distributed as torch_distributed
from torchvision import transforms
from tensorboardX import SummaryWriter

from models import SuperMobileNetV2
from ops import CrossEntropyLoss
from datasets import ImageNet
from distributed import init_process_group
from utils import Dict, apply_dict


def main(args):

    init_process_group(backend='nccl')

    with open(args.config) as file:
        config = apply_dict(Dict, json.load(file))
    config.update(vars(args))
    config.update(dict(
        world_size=torch_distributed.get_world_size(),
        global_rank=torch_distributed.get_rank(),
        device_count=cuda.device_count(),
        local_rank=torch_distributed.get_rank() % cuda.device_count()
    ))
    print(f'config: {config}')

    backends.cudnn.benchmark = True
    backends.cudnn.fastest = True

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cuda.manual_seed(config.seed)
    cuda.set_device(config.local_rank)

    train_dataset = ImageNet(
        root=config.train_root,
        meta=config.train_meta,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    )
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

    for tensor in model.state_dict().values():
        torch_distributed.broadcast(tensor, 0)

    criterion = CrossEntropyLoss(config.label_smoothing)

    config.global_batch_size = config.local_batch_size * config.world_size
    config.lr = config.lr * config.global_batch_size / config.global_batch_denom

    optimizer = torch.optim.RMSprop(
        params=model.weights(),
        lr=config.lr,
        alpha=config.alpha,
        eps=config.eps,
        weight_decay=config.weight_decay,
        momentum=config.momentum
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.milestones,
        gamma=config.gamma
    )

    last_epoch = -1
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        last_epoch = checkpoint.last_epoch
        global_step = checkpoint.global_step
    elif config.global_rank == 0:
        if os.path.exists(config.checkpoint_directory):
            shutil.rmtree(config.checkpoint_directory)
        if os.path.exists(config.event_directory):
            shutil.rmtree(config.event_directory)
        os.makedirs(config.checkpoint_directory)
        os.makedirs(config.event_directory)

    if config.global_rank == 0:
        summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        for epoch in range(last_epoch + 1, config.num_epochs):

            train_sampler.set_epoch(epoch)
            lr_scheduler.step(epoch)

            model.train()

            for local_step, (images, targets) in enumerate(train_data_loader):

                step_begin = time.time()

                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                logits = model(images)
                loss = criterion(logits, targets) / config.world_size

                optimizer.zero_grad()

                loss.backward()

                for parameter in model.parameters():
                    torch_distributed.all_reduce(parameter.grad)

                optimizer.step()

                predictions = torch.argmax(logits, dim=1)
                accuracy = torch.mean((predictions == targets).float()) / config.world_size

                for tensor in [loss, accuracy]:
                    torch_distributed.all_reduce(tensor)

                step_end = time.time()

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(train=loss),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(train=accuracy),
                        global_step=global_step
                    )
                    print(f'[training] epoch: {epoch} global_step: {global_step} local_step: {local_step} '
                          f'loss: {loss:.4f} accuracy: {accuracy:.4f} [{step_end - step_begin:.4f}s]')

                global_step += 1

            if config.global_rank == 0:
                torch.save(dict(
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    last_epoch=epoch,
                    global_step=global_step
                ), f'{config.checkpoint_directory}/epoch_{epoch}')

            if config.validation:

                model.eval()

                with torch.no_grad():

                    average_loss = 0
                    average_accuracy = 0

                    for local_step, (images, targets) in enumerate(val_data_loader):

                        images = images.cuda(non_blocking=True)
                        targets = targets.cuda(non_blocking=True)

                        logits = model(images)
                        loss = criterion(logits, targets) / config.world_size

                        predictions = torch.argmax(logits, dim=1)
                        accuracy = torch.mean((predictions == targets).float()) / config.world_size

                        for tensor in [loss, accuracy]:
                            torch_distributed.all_reduce(tensor)

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

            for local_step, (images, targets) in enumerate(val_data_loader):

                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                logits = model(images)
                loss = criterion(logits, targets) / config.world_size

                predictions = torch.argmax(logits, dim=1)
                accuracy = torch.mean((predictions == targets).float()) / config.world_size

                for tensor in [loss, accuracy]:
                    torch_distributed.all_reduce(tensor)

                average_loss += loss
                average_accuracy += accuracy

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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
