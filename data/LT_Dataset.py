import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pdb import set_trace
# from data.utils import _gaussian_blur
import torch.distributed as dist
import torchvision.transforms as transforms


class LT_Dataset(Dataset):

  def __init__(self, root, txt, transform=None, balance=False, returnPath=False):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath
    self.txt = txt

    with open(txt) as f:
      for line in f:
        self.img_path.append(os.path.join(root, line.split()[0]))
        self.labels.append(int(line.split()[1]))

    self.targets = self.labels

    if balance:
      assert False
      num_class = 1000
      idxs = np.array(list(range(len(self.img_path)))).astype(np.long)
      targets = np.array(self.labels)
      # set_trace()

      idxList = []
      for i in range(num_class):
        # print("i is {}".format(i))
        idxList.append(idxs[targets == i])

      idxListLen = [len(i) for i in idxList]
      maxLenIdx = max(idxListLen)
      # set_trace()

      newIdxList = []
      for idxs in idxList:
        if len(idxs) < maxLenIdx:
          # print("i is {}".format(i))
          idxSampled = random.choices(idxs, k=maxLenIdx - len(idxs))
          newIdxList += idxSampled
        newIdxList += idxs.tolist()
      # set_trace()

      self.img_path = np.array(self.img_path)[newIdxList].tolist()
      self.labels = np.array(self.labels)[newIdxList].tolist()

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):

    path = self.img_path[index]
    label = self.labels[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)

    if not self.returnPath:
      return sample, label, index
    else:
      return sample, label, index, path.replace(self.root, '')


class Unsupervised_LT_Dataset(LT_Dataset):
  def __init__(self, returnIdx=False, returnLabel=False, **kwds):
    super().__init__(**kwds)
    self.returnIdx = returnIdx
    self.returnLabel = returnLabel

  def __getitem__(self, index):
    path = self.img_path[index]
    label = self.labels[index]

    if not os.path.isfile(path):
      path = path + ".gz"

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    samples = [self.transform(sample), self.transform(sample)]
    if self.returnIdx and (not self.returnLabel):
      return torch.stack(samples), index
    elif (not self.returnIdx) and self.returnLabel:
      return torch.stack(samples), label
    elif self.returnIdx and self.returnLabel:
      return torch.stack(samples), label, index
    else:
      return torch.stack(samples)


class Unsupervised_LT_Dataset_Mix(Dataset):
  def __init__(self, roots, identifiers, txt, pair=True, **kwds):
    super().__init__()
    self.roots = roots
    self.identifiers = identifiers
    self.txt = txt
    self.datasets = []
    self.identifierList = []
    for root in self.roots:
      if pair:
        self.datasets.append(Unsupervised_LT_Dataset(root=root, txt=txt, **kwds))
      else:
        self.datasets.append(LT_Dataset(root=root, txt=txt, **kwds))

    with open(txt) as f:
      for line in f:
        self.identifierList.append(line.split()[2])

    for id in self.identifierList:
      if id not in self.identifiers:
        raise ValueError("id: {} is not in provided identifiers: {}".format(id, self.identifiers))

  def __getitem__(self, index):
    identifier = self.identifierList[index]
    index_identifier = self.identifiers.index(identifier)
    # print("index_identifier is {}, identifier is {}".format(index_identifier, identifier))

    return self.datasets[index_identifier].__getitem__(index)

  def __len__(self):
    return len(self.identifierList)

