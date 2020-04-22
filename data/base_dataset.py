"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_w = w
    new_h = h
    # if do resize or scale_width, first get new_w and new_h
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    rotate = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip, 'rotate': rotate}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, crop_size_scale=1):
    '''
    opt.preprocess:     [resize | scale_width | crop | resize_and_crop | scale_width_and_crop | none]

    :param opt:
    :param params:
    :param grayscale: whether to grayscale
    :param method:
    :param crop_size_scale: if you do crop, you can set bigger crop through this param factor
    :return: transforms.Compose(transform_list)
    '''
    rgb_mean = list(map(float, opt.normalize_means.split(',')))
    rgb_std = list(map(float, opt.normalize_stds.split(',')))
    if not grayscale:
        assert (len(rgb_mean) == 3 and len(rgb_std) == 3), 'please check out --normalize_means and --normalize_stds!'

    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            assert (crop_size_scale == 1), 'due to scale != 1, The content of the two images may not correspond'
            transform_list.append(transforms.RandomCrop(opt.crop_size * crop_size_scale))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, tuple([crop_size_scale*loc for loc in params['crop_pos']]), opt.crop_size * crop_size_scale)))

    # if none preprocess , make sure size some multiple of base. e.g. 4
    if opt.preprocess.lower() == 'none' and opt.multi_base > 0 and crop_size_scale == 1:  # only for LR now
        transform_list.append(transforms.Lambda(lambda img: make_power_2(img, base=opt.multi_base, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if not opt.no_rotate:
        if params is None:
            pass
        elif params['rotate']:
            transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['rotate'])))

    transform_list += [transforms.ToTensor()]

    if grayscale:
        transform_list += [transforms.Normalize((rgb_mean[0],), (rgb_std[0],))]
    else:
        transform_list += [transforms.Normalize(rgb_mean, rgb_std)]
    return transforms.Compose(transform_list)


def make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    ow_left = ow % base
    oh_left = oh % base
    if ow_left == 0 and oh_left == 0:
        return img
    # pad by 0 default (right,down) is more than (left, up) 1 if odd
    def pad_for_lr(x):
        if x == base:
            return 0, 0
        if x % 2 == 0:
            padx1 = x//2
            padx2 = padx1
        else:
            padx1 = x // 2
            padx2 = padx1 + 1
        return padx1, padx2

    w1, w2 = pad_for_lr(base - ow_left)
    h1, h2 = pad_for_lr(base - oh_left)
    return ImageOps.expand(img, border=(w1, h1, w2, h2), fill=0)  # left, top, right and bottom borders

    # h = int(round(oh / base) * base)
    # w = int(round(ow / base) * base)
    # if (h == oh) and (w == ow):
    #     return img
    #
    # __print_size_warning(ow, oh, w, h)
    # return img.resize((w, h), method)




def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __rotate(img, rotate):
    if rotate:
        return img.transpose(Image.ROTATE_90)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
