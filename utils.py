import torch
import torch.nn as nn
import os
import numpy as np
import random
import torch.nn.functional as F
import re


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")


def gatherFeatures(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def pair_cosine_similarity(x, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps)


def nt_xent(x, t=0.5, sampleWiseLoss=False, return_prob=False):
    # print("device of x is {}".format(x.device))

    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))

    if return_prob:
        return x.reshape(len(x) // 2, 2).mean(-1)

    sample_loss = -torch.log(x)

    if sampleWiseLoss:
        return sample_loss.reshape(len(sample_loss) // 2, 2).mean(-1)

    return sample_loss.mean()


def nt_xent_inter_batch_multiple_time(out, t=0.5, batch_size=512, repeat_time=10, return_porbs=False):
    d = out.size()
    dataset_len = d[0] // 2
    out = out.view(dataset_len, 2, -1).contiguous()
    dataset_features_1 = out[:, 0]
    dataset_features_2 = out[:, 1]

    # doesn't give gradient
    losses_all = []

    with torch.no_grad():
        for cnt in range(repeat_time):
            losses_batch = []
            # order features
            random_order = torch.randperm(dataset_len, device=out.device)
            order_back = torch.argsort(random_order)

            # get the loss
            assert dataset_len >= batch_size
            for i in range(int(np.ceil(dataset_len / batch_size))):
                if (i + 1) * batch_size < dataset_len:
                    samplingIdx = random_order[i * batch_size: (i + 1) * batch_size]
                    offset = 0
                else:
                    samplingIdx = random_order[dataset_len - batch_size:]
                    offset = i * batch_size - (dataset_len - batch_size)

                # calculate loss
                out1 = dataset_features_1[samplingIdx]
                out2 = dataset_features_2[samplingIdx]

                out = torch.stack([out1, out2], dim=1).view((batch_size * 2, -1))
                out = F.normalize(out, dim=-1)
                losses_or_probs = nt_xent(out, t=t, sampleWiseLoss=True, return_prob=return_porbs)[offset:]
                losses_batch.append(losses_or_probs)

            # reset the order
            losses_batch = torch.cat(losses_batch, dim=0)
            losses_batch = losses_batch[order_back]
            losses_all.append(losses_batch)

        # average togather
        losses_all = torch.stack(losses_all).mean(0)

        return losses_all


def getStatisticsFromTxt(txtName, num_class=1000):
      statistics = [0 for _ in range(num_class)]
      with open(txtName, 'r') as f:
        lines = f.readlines()
      for line in lines:
            s = re.search(r" ([0-9]+)$", line)
            if s is not None:
              statistics[int(s[1])] += 1
      return statistics


def gather_tensor(tensor, local_rank, world_size):
    # gather features
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    tensor_list[local_rank] = tensor
    tensors = torch.cat(tensor_list)
    return tensors


def getImagenetRoot(root):
    if os.path.isdir(root):
        pass
    elif os.path.isdir("/ssd1/bansa01/imagenet_final"):
        root = "/ssd1/bansa01/imagenet_final"
    elif os.path.isdir("/mnt/imagenet"):
        root = "/mnt/imagenet"
    elif os.path.isdir("/hdd3/ziyu/imagenet"):
        root = "/hdd3/ziyu/imagenet"
    elif os.path.isdir("/home/xueq13/scratch/ziyu/ImageNet/ILSVRC/Data/CLS-LOC"):
        root = "/home/xueq13/scratch/ziyu/ImageNet/ILSVRC/Data/CLS-LOC"
    elif os.path.isdir("/hdd1/ziyu/ImageNet"):
        root = "/hdd1/ziyu/ImageNet"
    else:
        print("No dir for imagenet")
        assert False

    return root


def getPlacesRoot(root):
    if os.path.isdir(root):
        pass
    if os.path.isdir("/hdd2/ziyu/places365"):
        root = "/hdd2/ziyu/places365"
    elif os.path.isdir("/scratch/user/jiangziyu/places365"):
        root = "/scratch/user/jiangziyu/places365"
    elif os.path.isdir("/home/xueq13/scratch/ziyu/Places"):
        root = "/home/xueq13/scratch/ziyu/Places"
    else:
        raise NotImplementedError("no root")

    return root


def reOrderData(idxs, labels, features):
    # sort all losses and idxes
    labels_new = []
    features_new = []
    idxs_new = []

    # reorder
    for idx, label, feature in zip(idxs, labels, features):
        order = np.argsort(idx)
        idxs_new.append(idx[order])
        labels_new.append(label[order])
        features_new.append(feature[order])

    # check if equal
    for cnt in range(len(idxs_new) - 1):
        if not np.array_equal(idxs_new[cnt], idxs_new[cnt+1]):
            raise ValueError("idx for {} and {} should be the same".format(cnt, cnt+1))

    return idxs_new, labels_new, features_new

