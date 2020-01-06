"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_images, rgb_mean = (0.5, 0.5, 0.5), rgb_std = (1.0, 1.0, 1.0)):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_images (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    assert (len(input_images.shape) == 4), 'tensor2im_image should be 4 dims due to mini-batch'
    if not isinstance(input_images, np.ndarray):
        if isinstance(input_images, torch.Tensor):  # get the data from a variable
            images_tensor = input_images.data
        else:
            return input_images  # who knows what is it  /(ㄒoㄒ)/~~
        image_tensor = images_tensor[0].cpu().float()  # only select the first one
        if image_tensor.shape[0] == 1:  # grayscale to RGB
            image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
        # normalize
        image_tensor = image_tensor * torch.tensor(rgb_std).view(3, 1, 1) + torch.tensor(rgb_mean).view(3, 1, 1)
        # clamp to [0,255]  or min max map to 0~255 is better?
        image_tensor = image_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        image_numpy = image_tensor.numpy()  # convert it into a numpy array.

    else:  # if it is a numpy array, do nothing
        image_numpy = input_images[0]
    return image_numpy


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
