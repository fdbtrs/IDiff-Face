#!/usr/bin/env python
import argparse
import os
import logging
import shutil

import torch
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import ConcatDataset
import torch.utils.data.distributed
import torch.nn.functional as F

from utils.augmentation import get_conventional_aug_policy
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from utils.dataset import DataLoaderX
from utils.utils_logging import init_logging, AverageMeter
from utils.utils_callbacks import (
    CallBackLogging,
    CallBackVerification,
    CallBackModelCheckpointOld,
)
from utils.losses import ArcFace, CosFace, AdaFace
from backbones.iresnet import iresnet100, iresnet50, iresnet34, iresnet18
import config.config as cfg

from limited_image_folder_dataset import LimitedWidthAndDepthImageFolder


def main(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cudnn.benchmark = True

    if args.model != "None":
        cfg.model = args.model 

    if args.embedding_type != "None":
        cfg.embedding_type = args.embedding_type

    if args.architecture != "None":
        cfg.architecture = args.architecture

    if args.width != 0:
        cfg.width = args.width

    if args.depth != 0:
        cfg.depth = args.depth

    rec = os.path.join(cfg.dataset_folder, cfg.model, cfg.embedding_type)  # training dataset
    output_dir = os.path.join("output", cfg.architecture, cfg.model, cfg.embedding_type + f"_w{cfg.width}_d{cfg.depth}")

    os.makedirs(output_dir, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root, local_rank, output_dir, logfile="training.log")
    # copy config to output folder
    shutil.copyfile(
        r"config/config.py", os.path.join(output_dir, "config.py")
    )

    ###############################################
    ################ Data Loading #################
    ###############################################
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )
    
    if args.augment != "None":
        transform = get_conventional_aug_policy(cfg.augmentation)

    if cfg.width != 0 or cfg.depth != 0:
        trainset = LimitedWidthAndDepthImageFolder(rec, transform=transform, width=cfg.width, depth=cfg.depth)
    else:
        trainset = ImageFolder(rec, transform=transform)
    
    num_classes = len(trainset.classes)

    logging.info(f"Length of dataset: {trainset}")
    logging.info(f"Number of classes: {num_classes}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    ###############################################
    ####### Create Model + resume Training ########
    ###############################################
    logging.info("=> creating model '{}'".format(cfg.architecture))

    if cfg.architecture == "resnet18":
        model_class = iresnet18
    elif cfg.architecture == "resnet50":
        model_class = iresnet50
    else:
        raise NotImplementedError

    model = model_class(num_features=cfg.embedding_size, dropout=cfg.dropout_ratio).to(
        local_rank
    )
    
    start_epoch = 0
    if args.resume:
        try:
            backbone_pth = os.path.join(
                output_dir, str(cfg.global_step) + "backbone.pth"
            )
            model.load_state_dict(
                torch.load(backbone_pth, map_location=torch.device(local_rank))
            )
            start_epoch = cfg.start_epoch
            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    for ps in model.parameters():
        dist.broadcast(ps, 0)

    model = DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[local_rank]
    )
    model.train()

    if cfg.loss == "ArcFace":
        header = ArcFace(
            in_features=cfg.embedding_size,
            out_features=num_classes,
            s=cfg.s,
            m=cfg.m,
        ).to(local_rank)
    elif cfg.loss == "CosFace":
        header = CosFace(
            in_features=cfg.embedding_size,
            out_features=num_classes,
            s=cfg.s,
            m=cfg.m
        ).to(local_rank)
    elif cfg.loss == "AdaFace":
        header = AdaFace(
            embedding_size=cfg.embedding_size,
            classnum=num_classes,
        ).to(local_rank)
    else:
        print("Header not implemented")

    if args.resume:
        try:
            header_pth = os.path.join(
                output_dir, str(cfg.global_step) + "header.pth"
            )
            header.load_state_dict(
                torch.load(header_pth, map_location=torch.device(local_rank))
            )

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")

    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank]
    )
    header.train()

    ###############################################
    ######### loss function + optimizer ###########
    ###############################################
    criterion = CrossEntropyLoss()

    opt_backbone = torch.optim.SGD(
        params=[{"params": model.parameters()}],
        lr=cfg.learning_rate / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    opt_header = torch.optim.SGD(
        params=[{"params": header.parameters()}],
        lr=cfg.learning_rate / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func
    )
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func
    )

    if cfg.auto_schedule:
        scheduler_backbone = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_backbone, mode="max", factor=0.1, patience=4
        )
        scheduler_header = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_header, mode="max", factor=0.1, patience=4
        )


    ###############################################
    ################# Callbacks ###################
    ###############################################
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)
    callback_logging = CallBackLogging(
        cfg.print_freq, rank, total_step, cfg.batch_size, world_size
    )
    callback_verification = CallBackVerification(
        1, rank, cfg.val_targets, cfg.val_path, img_size=[112, 112]
    )
    callback_checkpoint = CallBackModelCheckpointOld(rank, output_dir)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global_step = 0
    e_acc = torch.zeros(2).to(rank)  # (epoch, maxAcc)
    avg_acc = torch.zeros(1).to(rank)

    ###############################################
    ################## Training ###################
    ###############################################
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        logging.info(f"learning rate: {round(opt_backbone.param_groups[0]['lr'], 12)}")

        # train for one epoch
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            features = F.normalize(model(images))

            if cfg.loss == "AdaFace":
                norm = torch.norm(features, 2, 1, True)
                output = torch.div(features, norm)
                thetas = header(output, norm, labels)
            else:
                thetas = header(features, labels)

            loss_v = criterion(thetas, labels)
            loss_v.backward()

            clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            clip_grad_norm_(header.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            losses.update(loss_v.item(), 1)
            acc = multi_class_acc(thetas, labels)
            top1.update(acc)
            top5.update(0)

            callback_logging(global_step, losses, top1, top5, epoch)
            global_step += 1

        #callback_checkpoint(global_step, model, header)
        ver_accs = callback_verification(global_step, model)
        avg_acc[0] = sum(ver_accs[:5]) / 5
        dist.broadcast(avg_acc, src=0)
        if cfg.auto_schedule:
            scheduler_backbone.step(avg_acc)
            scheduler_header.step(avg_acc)
        else:
            scheduler_backbone.step()
            scheduler_header.step()
        # update max accuracy
        if rank == 0 and avg_acc[0] > e_acc[1]:
            e_acc[0] = epoch
            e_acc[1] = avg_acc[0]
            if cfg.auto_schedule:
                callback_checkpoint(global_step, model, header)
        dist.broadcast(e_acc, src=0)

        # early stopping
        if cfg.auto_schedule and e_acc[0] <= epoch - 6:
            callback_checkpoint(global_step, model, header)
            logging.info(
                f"Avg validation accuracy did not improve for 6 epochs. Terminating at epoch {epoch} with max performance achieved at {e_acc[0]} with best average of {e_acc[1]} ..."
            )
            exit()


@torch.no_grad()
def multi_class_acc(pred, labels):
    a_max = torch.argmax(pred, dim=1)
    acc = (a_max == labels).sum().item() / labels.size(0)
    acc = round(acc * 100, 2)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--resume", type=int, default=0, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")

    parser.add_argument("--model", type=str, default="None", help="name of generative difusion model, e.g. unet-cond-ca-bs512-150K")
    parser.add_argument("--architecture", type=str, default="None", help="name of architecture, e.g. resnet18 or resnet50")
    parser.add_argument("--embedding_type", type=str, default="None", help="name of embedding type")

    parser.add_argument("--width", type=int, default=0, help="number of ids")
    parser.add_argument("--depth", type=int, default=0, help="samples per ids")

    parser.add_argument("--augment", type=str, default="None", help="")

    parser.add_argument("--savepath", type=str, default="None", help="path to save the model")
    args = parser.parse_args()
    main(args)
