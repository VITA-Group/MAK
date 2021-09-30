import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet import resnet18, resnet10, resnet50, resnet101, resnet152
from utils import *
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np
from data.augmentation import GaussianBlur

import time
from data.LT_Dataset import Unsupervised_LT_Dataset, LT_Dataset, Unsupervised_LT_Dataset_Mix
from pdb import set_trace


from inference.inference_save import inference_save


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='/hdd3/ziyu/SS_imbalance/checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset, [imagenet, imagenet-100, imagenet-FULL, imagenet-32-FULL, places, cifar, cifar100, iNatural18]')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_freq', default=100, type=int, help='save frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saved pretrained model for resuming')
parser.add_argument('--checkpoint_pretrain', default='', type=str, help='pretrained model')
parser.add_argument('--resume', action='store_true', help='if resume training')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type')
parser.add_argument('--lr', default=0.5, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--model', default='res18', type=str, help='model type')
parser.add_argument('--temperature', default=0.2, type=float, help='nt_xent temperature')
parser.add_argument('--output_ch', default=128, type=int, help='proj head output feature number')
parser.add_argument('--trainSplit', type=str, default='trainIdxList.npy', help="train split")

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

parser.add_argument('--colorStren', default=1.0, type=float, help='cifar augmentation, color jitter strength')

# additional dataset
parser.add_argument('--additional_dataset', type=str, default="", help="which additional dataset to be used")
parser.add_argument('--additional_dataset_root', type=str, default="", help="root for additional dataset")
parser.add_argument('--additional_dataset_split', type=str, default="", help="which split")

# inference settings
parser.add_argument('--inference', action='store_true', help='if do inference for sampling')
parser.add_argument('--inference_dataset', default='imagenet-900', type=str, help='dataset name')
parser.add_argument('--inference_dataset_root', default='', type=str, help='inference dataset root')
parser.add_argument('--inference_dataset_split', default='ImageNet_900_train', type=str, help='dataset name')
parser.add_argument('--inference_repeat_time', default=5, type=int, help='inference repeat time')
parser.add_argument('--inference_noAug', action='store_true', help='if use augmentation while inference')
parser.add_argument('--inference_no_save_before_proj_feature', action='store_true', help='not save features before projection head if specified')


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def main():
    global args
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")

    torch.cuda.set_device(args.local_rank)
    
    rank = torch.distributed.get_rank()
    logName = "log.txt"

    log = logger(path=save_dir, local_rank=rank, log_name=logName)
    log.info(str(args))

    setup_seed(args.seed + rank)
    
    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))

    print("rank is {}, world size is {}".format(rank, world_size))

    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size

    imagenet=True

    if 'imagenet' in args.dataset:
        num_class = 1000
        if 'imagenet-100' in args.dataset:
            num_class = 100
    else:
        assert False

    if args.model == 'res10':
        model = resnet10(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res18':
        model = resnet18(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res50':
        model = resnet50(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res101':
        model = resnet101(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res152':
        model = resnet152(pretrained=False, imagenet=imagenet, num_classes=num_class)
    else:
        assert False

    ch = model.fc.in_features

    from models.utils import proj_head
    model.fc = proj_head(ch, args.output_ch)

    model.cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    colorStren = args.colorStren
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * colorStren, 0.4 * colorStren, 0.4 * colorStren, 0.1 * colorStren)], p=0.8)
    log.info("employed color jittering strength is {}".format(colorStren))

    rnd_gray = transforms.RandomGrayscale(p=0.2)

    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
    ])

    tfs_test = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      ])

    # dataset process
    if args.dataset == 'imagenet-100':
        txt = "split/imagenet-100/ImageNet_100_train.txt"
        if args.trainSplit != '':
            txt = "split/imagenet-100/{}.txt".format(args.trainSplit)
        print("use imagenet-100 {}".format(args.trainSplit))
    else:
        if args.trainSplit != '':
            txt = "split/ImageNet_LT/{}.txt".format(args.trainSplit)
            print("use {}".format(txt))
        elif args.dataset == "imagenet":
            print("use imagenet long tail")
            txt = "split/ImageNet_LT/ImageNet_LT_train.txt"
        else:
            print("use imagenet full set")
            txt = "split/ImageNet_LT/ImageNet_train.txt"

    root = getImagenetRoot(args.data)

    train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=tfs_train)

    class_stat = [0 for _ in range(num_class)]
    for lbl in train_datasets.labels:
        class_stat[lbl] += 1
    log.info("class distribution in training set is {}".format(class_stat))

    if args.additional_dataset != "":
        train_datasets_additional = initalizeDataset(args.additional_dataset, args.additional_dataset_root,
                                                     args.additional_dataset_split, tfs_train)
        train_datasets = torch.utils.data.ConcatDataset([train_datasets, train_datasets_additional])

    if args.inference:
        if args.inference_noAug:
            print("no augmentation for inference")
            tfs_inference = tfs_test
            assert args.inference_repeat_time == 1; "when w/o augmentation, inference time should be 1."
        else:
            tfs_inference = tfs_train
        train_datasets_inference = initalizeDataset(args.inference_dataset, args.inference_dataset_root,
                                                    args.inference_dataset_split, tfs_inference, pairLoader=False, returnIdx=True)
        inference_dataset_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets_inference, shuffle=False)
        inference_dataset_loader = torch.utils.data.DataLoader(
            train_datasets_inference,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=inference_dataset_sampler,
            pin_memory=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=args.num_workers,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=False)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ], gamma=1)
    elif args.scheduler == 'cosine':
        training_iters = args.epochs * len(train_loader)
        warm_up_iters = 10 * len(train_loader)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    training_iters,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=warm_up_iters)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    if args.checkpoint_pretrain != '':
        checkpoint = torch.load(args.checkpoint_pretrain, map_location="cpu")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    start_epoch = 1
    if args.resume:
        if args.checkpoint != '':
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        elif os.path.isfile(os.path.join(save_dir, 'model.pt')):
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        else:
             checkpoint = None

        if checkpoint is not None:
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optim'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                raise ValueError("checkpoint broken")
        else:
            log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log.info("no available checkpoint, start from scratch or pretrain!!!!!!!!!!!")
            log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        assert args.checkpoint == ''

    if args.inference:
        inference_save(inference_dataset_loader, model, log, args.local_rank, world_size, args, args.inference_repeat_time,
                       inference_dataset_loader.dataset.txt, save_dir)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, scheduler, epoch, log, args.local_rank, rank, world_size, args=args)

        if rank == 0:

            save_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }

            if epoch % 1 == 0:
                save_checkpoint(save_dict, filename=os.path.join(save_dir, 'model.pt'))

            if epoch % args.save_freq == 0 and epoch > 0:
                save_checkpoint(save_dict, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))


def gather_features(features, world_size, rank):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[rank] = features
    features = torch.cat(features_list)
    return features


def train(train_loader, model, optimizer, scheduler, epoch, log, local_rank, rank, world_size, args=None):
    
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()

    for i, (inputs) in enumerate(train_loader):
        weightIns = None

        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda(non_blocking=True)

        model.train()
        features = model(inputs)
        features = gather_features(features, world_size, rank)

        loss = nt_xent(features, t=args.temperature)
        # normalize the loss
        loss = loss * world_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu() / world_size), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))

    return losses.avg


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


def initalizeDataset(dataset, root, split, transform, pairLoader=True, returnIdx=False):
    if dataset == "imagenet-900":
        root = getImagenetRoot(root)

        txt = "split/imagenet-900/{}.txt".format(split)

        if pairLoader:
            train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=transform, returnIdx=returnIdx)
        else:
            train_datasets = LT_Dataset(root=root, txt=txt, transform=transform)
    elif dataset == "imagenet-100-test":
        # for toy case of imagenet-100
        root = getImagenetRoot(root)

        txt = "split/imagenet-100/{}.txt".format('ImageNet_100_test')

        if pairLoader:
            train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=transform, returnIdx=returnIdx)
        else:
            train_datasets = LT_Dataset(root=root, txt=txt, transform=transform)

    elif dataset == "imagenet-100":
        # for toy case of imagenet-100
        root = getImagenetRoot(root)

        txt = "split/imagenet-100/{}.txt".format(split)

        if pairLoader:
            train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=transform, returnIdx=returnIdx)
        else:
            train_datasets = LT_Dataset(root=root, txt=txt, transform=transform)

    elif dataset == "imagenet":
        # for toy case of imagenet-100
        root = getImagenetRoot(root)

        txt = "split/ImageNet_LT/{}.txt".format(split)

        if pairLoader:
            train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=transform, returnIdx=returnIdx)
        else:
            train_datasets = LT_Dataset(root=root, txt=txt, transform=transform)

    elif dataset == "imagenet_places365_mix":
        root_imagenet = getImagenetRoot(root)
        root_places = getPlacesRoot(root)

        txt = "split/imagenet-places-mix/{}.txt".format(split)

        if pairLoader:
            train_datasets = Unsupervised_LT_Dataset_Mix([root_imagenet, root_places], ["imagenet", "places"], txt, transform=transform, returnIdx=returnIdx)
        else:
            train_datasets = Unsupervised_LT_Dataset_Mix([root_imagenet, root_places], ["imagenet", "places"], txt, pair=False, transform=transform)

    else:
        assert False

    return train_datasets


if __name__ == '__main__':
    main()


