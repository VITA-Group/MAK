import torch
import torch.nn.functional as F
import math


def gaussian_kernel(size, sigma=2., dim=2, channels=3):
  # The gaussian kernel is the product of the gaussian function of each dimension.
  # kernel_size should be an odd number.

  kernel_size = 2 * size + 1

  kernel_size = [kernel_size] * dim
  sigma = [sigma] * dim
  kernel = 1
  meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

  for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
    mean = (size - 1) / 2
    kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

  # Make sure sum of values in gaussian kernel equals 1.
  kernel = kernel / torch.sum(kernel)

  # Reshape to depthwise convolutional weight
  kernel = kernel.view(1, 1, *kernel.size())
  kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

  return kernel


def _gaussian_blur(x, size, sigma):
  kernel = gaussian_kernel(size=size, sigma=sigma)
  kernel_size = 2 * size + 1

  x = x[None, ...]
  padding = int((kernel_size - 1) / 2)
  x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
  x = torch.squeeze(F.conv2d(x, kernel, groups=3))

  return x

