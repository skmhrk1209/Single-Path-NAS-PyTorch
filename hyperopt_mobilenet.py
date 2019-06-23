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
import optuna


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
        local_rank=distributed.get_rank() % torch.cuda.device_count(),
        global_batch_size=config.local_batch_size * distributed.get_world_size()
    ))
    print(f'config: {config}')

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    model = MobileNetV2(
        first_conv_param=Dict(in_channels=3, out_channels=32, kernel_size=3, stride=2),
        middle_conv_params=[
            Dict(in_channels=32, out_channels=16, expand_ratio=1, kernel_size=3, blocks=1, stride=1),
            Dict(in_channels=16, out_channels=24, expand_ratio=6, kernel_size=3, blocks=2, stride=2),
            Dict(in_channels=24, out_channels=32, expand_ratio=6, kernel_size=3, blocks=3, stride=2),
            Dict(in_channels=32, out_channels=64, expand_ratio=6, kernel_size=3, blocks=4, stride=2),
            Dict(in_channels=64, out_channels=96, expand_ratio=6, kernel_size=3, blocks=3, stride=1),
            Dict(in_channels=96, out_channels=160, expand_ratio=6, kernel_size=3, blocks=3, stride=2),
            Dict(in_channels=160, out_channels=320, expand_ratio=6, kernel_size=3, blocks=1, stride=1),
        ],
        last_conv_param=Dict(in_channels=320, out_channels=1280, kernel_size=1, stride=1),
        drop_prob=config.drop_prob,
        num_classes=1000
    ).cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

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

    def sgd(trial, model, weight_decay):
        scale = config.global_batch_size / config.global_batch_denom
        return optim.SGD(
            params=model.parameters(),
            lr=trial.suggest_loguniform('sgd_lr', 1e-3, 1e-1) * scale,
            momentum=trial.suggest_loguniform('sgd_momentum', 0.9, 0.9),
            weight_decay=weight_decay
        )

    def rmsprop(trial, model, weight_decay):
        scale = config.global_batch_size / config.global_batch_denom
        return optim.RMSprop(
            params=model.parameters(),
            lr=trial.suggest_loguniform('rmsprop_lr', 1e-3, 1e-1) * scale,
            alpha=trial.suggest_uniform('rmsprop_alpha', 0.9, 0.9),
            momentum=trial.suggest_uniform('rmsprop_momentum', 0.9, 0.9),
            eps=trial.suggest_loguniform('rmsprop_eps', 1e-0, 1e-0),
            weight_decay=weight_decay
        )

    def suggest_optimizer(trial, model):
        optimizers = dict(sgd=sgd, rmsprop=rmsprop)
        optimizer = optimizers[trial.suggest_categorical('optimizer', list(optimizers))]
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
        return optimizer(trial, model, weight_decay)

    def cosine_annealing_lr(trial, optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=trial.suggest_int('cosine_annealing_lr_tmax', 100, 200)
        )

    def suggest_lr_scheduler(trial, optimizer):
        lr_schedulers = dict(cosine_annealing_lr=cosine_annealing_lr)
        lr_scheduler = lr_schedulers[trial.suggest_categorical('lr_scheduler', list(lr_schedulers))]
        return lr_scheduler(trial, optimizer)

    def objective(trial):

        if config.global_rank == 0:
            checkpoint_directory = os.path.join(config.checkpoint_directory, f'trial_{trial.number}')
            event_directory = os.path.join(config.event_directory, f'trial_{trial.number}')
            os.makedirs(checkpoint_directory, exist_ok=True)
            os.makedirs(event_directory, exist_ok=True)
            summary_writer = SummaryWriter(event_directory)

        last_epoch = -1
        global_step = 0
        if config.checkpoint:
            checkpoint = Dict(torch.load(config.checkpoint))
            model.load_state_dict(checkpoint.model_state_dict)
            optimizer = checkpoint.optimizer
            lr_scheduler = checkpoint.lr_scheduler
            last_epoch = checkpoint.last_epoch
            global_step = checkpoint.global_step
            config.checkpoint = ''
        else:
            # ...
            checkpoint = os.path.join(checkpoint_directory, f'epoch_{last_epoch}')
            if config.global_rank == 0:
                optimizer = suggest_optimizer(trial, model)
                lr_scheduler = suggest_lr_scheduler(trial, optimizer)
                torch.save(dict(
                    model_state_dict=model.state_dict(),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    last_epoch=last_epoch,
                    global_step=global_step
                ), checkpoint)
            # distributed.barrier()
            distributed.broadcast(torch.empty(0), 0)
            # distributed.broadcast([optimizer, lr_scheduler], 0)
            if config.global_rank != 0:
                checkpoint = torch.load(checkpoint)
                optimizer = checkpoint.optimizer
                lr_scheduler = checkpoint.lr_scheduler

        for epoch in range(last_epoch + 1, config.num_epochs):

            train_sampler.set_epoch(epoch)
            lr_scheduler.step(epoch)

            model.train()

            for local_step, (images, labels) in enumerate(train_data_loader):

                step_begin = time.time()

                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                loss = criterion(logits, labels) / config.world_size

                optimizer.zero_grad()
                loss.backward()
                for parameter in model.parameters():
                    distributed.all_reduce(parameter.grad)
                optimizer.step()

                predictions = logits.topk(1)[1].squeeze()
                accuracy = torch.mean((predictions == labels).float()) / config.world_size

                for tensor in [loss, accuracy]:
                    distributed.all_reduce(tensor)

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
                ), f'{checkpoint_directory}/epoch_{epoch}')

            model.eval()

            with torch.no_grad():

                average_loss = 0
                average_accuracy = 0

                for local_step, (images, labels) in enumerate(val_data_loader):

                    images = images.cuda()
                    labels = labels.cuda()

                    logits = model(images)
                    loss = criterion(logits, labels) / config.world_size

                    predictions = logits.topk(1)[1].squeeze()
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

            should_prune = torch.zeros(1)
            if config.global_rank == 0:
                trial.report(average_loss, epoch)
                if trial.should_prune():
                    should_prune += 1
            distributed.broadcast(should_prune, 0)
            if should_prune:
                if config.global_rank == 0:
                    summary_writer.close()
                    raise optuna.structs.TrialPruned()
                else:
                    return

        if config.global_rank == 0:
            summary_writer.close()

        return average_loss

    study = optuna.load_study(
        study_name=config.hyperopt.study_name,
        storage=config.hyperopt.storage,
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=config.hyperopt.pruner.min_resource,
            reduction_factor=config.hyperopt.pruner.reduction_factor,
            min_early_stopping_rate=config.hyperopt.pruner.min_early_stopping_rate
        )
    )
    study.optimize(objective, n_trials=config.hyperopt.n_trials)
    print(f'best_parameters: {study.best_params}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MobileNetV2')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    main(args)
