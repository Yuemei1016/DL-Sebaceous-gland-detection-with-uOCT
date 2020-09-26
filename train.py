# This code is copied from a github respotry 2020.08.08 8:54 a.m.
# This code is copied from a github respotry 2020.08.10 10:00 a.m.
# this code is to add logging function and siplify the main file. By Ruibing 2020.08.11.15:11


import argparse
import os
import random
import shutil
import time
import warnings
from config import config, update_config
import logging
import pprint
import numpy as np

import _init_paths
import callback
import metric
import create_logger
import model_lib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def parse_args():
    parser = argparse.ArgumentParser(description='MOCT image classification network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    return args

best_acc1 = 0


def main():
    args = parse_args()
    curr_path = os.path.abspath(os.path.dirname(__file__))
    logger, final_output_path = create_logger.create_logger(curr_path, args.cfg, config)
    print('Using config:')
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger, final_output_path))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, final_output_path)


def main_worker(gpu, ngpus_per_node, config, logger, output_pt):
    global best_acc1

    if config.gpu is not None:
        print("Use GPU: {} for training".format(gpu))


    if config.distributed:
        config.gpu = gpu
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
    else:
        config.gpu = 0

    model, input_size = model_lib.initialize_model(config.arch, config.num_cls, use_pretrained=config.pretrained)
    model = nn.Sequential(model, nn.Sigmoid())

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
        model = model.cuda(0)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if config.arch.startswith('alexnet') or config.arch.startswith('vgg'):
            model[0].features = torch.nn.DataParallel(model[0].features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # optionally resume from a checkpoint
    if config.resume:
        resume_file = os.path.join(output_pt, config.arch + '_' + str(config.resume) + '_epoch' + '.pth.tar')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            logger.info("=> loading checkpoint '{}'".format(resume_file))
            if config.gpu is None:
                checkpoint = torch.load(resume_file)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:'+ str(config.gpu)
                checkpoint = torch.load(resume_file, map_location=loc)
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))
            logger.info("=> no checkpoint found at '{}'".format(resume_file))

    cudnn.benchmark = True

    # Data loading code
    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_root_pt = os.path.join(cur_path, 'data', 'MOCT', 'Data', 'jpeg_imgs')
    traindir = os.path.join(data_root_pt, 'train')
    valdir = os.path.join(data_root_pt, 'vis')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    if config.evaluate:
        validate(val_loader, model, criterion, config, logger, output_pt)
        return

    lr_scheduler = callback.lr_scheduler(config.lr_epoch, config.lr_inter, config.lr, config.lr_factor, config.warmup,
                                        config.warmup_lr, config.warmup_epoch)

    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)
        lr_scheduler(optimizer, epoch, logger)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config, logger, output_pt)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, config, logger, output_pt)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                    and config.rank % ngpus_per_node == 0):
            file_name = os.path.join(output_pt, config.arch + '_' + str(epoch+1) + '_epoch' + '.pth.tar')
            callback.save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename= file_name)
            logger.info('Save the model as {:}'.format(config.arch + '_' + str(epoch+1) + '_epoch' + '.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, config, logger, output_pt):
    batch_time = callback.AverageMeter('Time=', ':6.3f')
    data_time = callback.AverageMeter('Data=', ':6.5f')
    losses = callback.AverageMeter('Loss=', ':.4e')
    acc = callback.AverageMeter('Acc=', ':1.3f')
    prec = callback.AverageMeter('Pre=', ':1.3f')
    rec = callback.AverageMeter('Rec=', ':1.3f')

    progress = callback.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc, prec, rec],
        logger, prefix="Epoch: [{}]".format(epoch))

    # define roc and auc variables
    outputs = np.empty([0,1], dtype = np.float32)
    targets = np.empty([0,1], dtype = np.float32)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end, images.size(0))

        target = target.type(torch.FloatTensor)
        target_ = torch.unsqueeze(target, 1)

        if config.gpu is not None:
            gpu_id = int(config.gpu)
            images = images.cuda(gpu_id, non_blocking=True)
        if torch.cuda.is_available():
            target_ = target_.cuda(gpu_id, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target_)

        # measure accuracy and record loss
        acc_ = metric.accuracy(output, target_)
        pre_ = metric.precision(output, target_)
        rec_ = metric.recall(output, target_)

        losses.update(loss.item() * images.size(0), images.size(0))
        acc.update(acc_[0], acc_[1])
        prec.update(pre_[0], pre_[1])
        rec.update(rec_[0], rec_[1])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record results and labels for computing auc and roc
        outputs = np.concatenate([outputs, output.cpu().detach().numpy()])
        targets = np.concatenate([targets, target_.cpu().numpy()])

        # measure elapsed time
        batch_time.update(time.time() - end, images.size(0))
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i)

    F1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg + 1e-6)
    fpr, tpr, roc_auc = metric.roc(outputs, targets)

    print('Final Train-Loss:{losses.avg:.3f} \t Train-Acc:{acc.avg:.3f} \t Train-Prec:{prec.avg:.3f} \t Train-Recall:{rec.avg:.3f} \t Train-F1:{f1:.3f} \t Train-Auc:{auc:.3f}'
        .format(losses=losses, acc=acc, prec=prec, rec=rec, f1=F1, auc=roc_auc))
    logger.info('Final Train-Loss:{losses.avg:.3f} \t Train-Acc:{acc.avg:.3f} \t Train-Prec:{prec.avg:.3f} \t Train-Recall:{rec.avg:.3f} \t Train-F1:{f1:.3f} \t Train-Auc:{auc:.3f}'
        .format(losses=losses, acc=acc, prec=prec, rec=rec, f1=F1, auc=roc_auc))

def validate(val_loader, model, criterion, config, logger, output_pt):
    batch_time = callback.AverageMeter('Time=', ':6.3f')
    data_time = callback.AverageMeter('Data=', ':6.5f')
    losses = callback.AverageMeter('Loss=', ':.4e')
    acc = callback.AverageMeter('Acc=', ':1.3f')
    prec = callback.AverageMeter('Pre=', ':1.3f')
    rec = callback.AverageMeter('Rec=', ':1.3f')

    progress = callback.ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, acc, prec, rec],
        logger, prefix='Test: ')

    # define roc and auc variables
    outputs = np.empty([0,1], dtype = np.float32)
    targets = np.empty([0,1], dtype = np.float32)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            # measure data loading time
            data_time.update(time.time() - end, images.size(0))

            target = target.type(torch.FloatTensor)
            target_ = torch.unsqueeze(target, 1)

            if config.gpu is not None:
                gpu_id = int(config.gpu)
                images = images.cuda(gpu_id, non_blocking=True)
            if torch.cuda.is_available():
                target_ = target_.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target_)
            if config.vis:
                imshow(images, title=target[0])

            # measure accuracy and record loss
            acc_ = metric.accuracy(output, target_)
            pre_ = metric.precision(output, target_)
            rec_ = metric.recall(output, target_)

            losses.update(loss.item()*images.size(0), images.size(0))
            acc.update(acc_[0], acc_[1])
            prec.update(pre_[0], pre_[1])
            rec.update(rec_[0], rec_[1])

            # record results and labels for computing auc and roc
            outputs = np.concatenate([outputs, output.cpu().numpy()])
            targets = np.concatenate([targets, target_.cpu().numpy()])

            # measure elapsed time
            batch_time.update(time.time() - end, images.size(0))
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i)

        F1 = 2* prec.avg * rec.avg/(prec.avg + rec.avg + 1e-6)
        fpr, tpr, roc_auc = metric.roc(outputs, targets)
        print('Final Validation-Loss:{losses.avg:.3f} \t Validation-Acc:{acc.avg:.3f} \t Validation-Prec:{prec.avg:.3f} \t Validation-Recall:{rec.avg:.3f} \t Validation-F1:{f1:.3f} \t Validation-Auc:{auc:.3f}'
              .format(losses=losses, acc=acc, prec=prec, rec=rec, f1= F1, auc=roc_auc))
        logger.info('Final Validation-Loss:{losses.avg:.3f} \t Validation-Acc:{acc.avg:.3f} \t Validation-Prec:{prec.avg:.3f} \t Validation-Recall:{rec.avg:.3f} \t Validation-F1:{f1:.3f} \t Validation-Auc:{auc:.3f}'
                .format(losses=losses, acc=acc, prec=prec, rec=rec, f1= F1, auc=roc_auc))

    return acc.avg

def imshow(images, title=None):
    """Imshow for Tensor."""
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision

    inp = torchvision.utils.make_grid(images[0])
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        im_title = 'Probability:{:}'.format(title.cpu().numpy())
        plt.gca().text(inp.shape[0]/2-7, 10,
                       im_title,fontsize=12, color='white')

    plt.show()

if __name__ == '__main__':
    main()