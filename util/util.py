"""This module contains simple helper functions """
from __future__ import print_function
import torch

import numpy as np
from PIL import Image
import os
import cv2


def tensor2im(input_image, rgb_mean = (0., 0., 0.), rgb_std = (1.0, 1.0, 1.0)):
    """"Converts a Tensor array into a numpy image array. [h,w,c] or [b,h,w,c](video)  [0,1]

    Parameters:
        input_image (tensor) --  the input image tensor array
        return               --  unit8 ndarray
    """
    assert isinstance(input_image, torch.Tensor), 'the input tensor should be torch.Tensor'
    image_tensor = input_image.data
    image_tensor = image_tensor.cpu().float()
    dim_len = len(image_tensor.shape)
    assert dim_len in (3, 4), 'dim_len should in (3,4)'

    # gray to RGB
    if image_tensor.shape[dim_len-3] == 1:
        image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), dim_len-3)

    if dim_len == 3:
        # normalize
        image_tensor = image_tensor * torch.tensor(rgb_std).view(3, 1, 1) + torch.tensor(rgb_mean).view(3, 1, 1)
        # clamp to [0,255]  or min max map to 0~255 is better?
        image_tensor = image_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
    elif dim_len == 4:
        # normalize
        image_tensor = image_tensor * torch.tensor(rgb_std).view(1, 3, 1, 1) + torch.tensor(rgb_mean).view(1, 3, 1, 1)
        # clamp to [0,255]
        image_tensor = image_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(torch.uint8)
    else:
        raise NotImplementedError("unknown image.shape")

    return image_tensor.numpy()


def save_image(image, image_path, factor=1, inverse=False):
    """Save a numpy(or PIL.image) image to the disk

    Parameters:
        image                     -- input numpy array or PIL.image
        image_path (str)          -- the path of the image
        factor                    -- factor for resize
        inverse                   -- if True down sample otherwise up sample
    """
    if isinstance(image, np.ndarray):
        assert len(image.shape) == 3
        image_pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError('image must be PIL.Image.Image or 3d np.ndarray!')

    w, h = image_pil.size

    assert factor >= 1, "factor should >=1"

    if factor > 1:
        if inverse:
            assert w % factor == 0 and h % factor == 0, "w,h should % SR_factor=0"
            image_pil = image_pil.resize((w//factor, h//factor), resample=Image.BICUBIC)
        else:
            image_pil = image_pil.resize((int(w*factor), int(h*factor)), resample=Image.BICUBIC)

    image_pil.save(image_path)


def save_video(video, video_path, factor=1, fps=2, inverse=True):
    '''
        Save a numpy video to the disk
    :param video_frames_list:  rgb numpy image(or PIL.image) list [..., [h,w,c], ...]  or single 4d ndarray [b,h,w,c]
    :param video_path: dst path
    :param SR_factor:
    :param fps:
    :return:  none
    '''
    if isinstance(video, list):
        for i in range(len(video)):
            if isinstance(video[i], np.ndarray):
                assert len(video[i].shape) == 3, 'not video!'
            elif isinstance(video[i], Image.Image):
                video[i] = np.array(video[i])
                assert len(video[i].shape) == 3, 'not video!'
        length = len(video)
    elif isinstance(video, np.ndarray):
        assert len(video.shape) == 4, 'not video!'
        length = video.shape[0]

    h, w, _ = video[0].shape
    # notice that in general task, idx 0 is all black...
    # print_numpy(video_frames_list[1])

    assert factor >= 1, "factor should >=1"

    if factor > 1:
        if inverse:
            assert w % factor == 0 and h % factor == 0, "w,h should % SR_factor=0"
            for i in range(length):
                video[i] = cv2.resize(video[i], (w//factor, h//factor), interpolation=cv2.INTER_CUBIC)
        else:
            for i in range(length):
                video[i] = cv2.resize(video[i], (int(w*factor), int(h*factor)), interpolation=cv2.INTER_CUBIC)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'I420'), fps, (w, h))
    for i in range(length):
        frame = video[i]
        out.write(frame[..., ::-1])
    out.release()


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


def check_whether_last_dir(path):
    """
    check whether the path is the last dir(thus don't include another dir)
    :param path: path to dir
    :return:
    """
    if not os.path.exists(path):
        return False

    if not os.path.isdir(path):
        return False

    for root, dirs, files in os.walk(path):
        if len(dirs) > 0:
            return False

    return True


def print_numpy(x, val=True, shp=True):
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


def moving_average(x, ma):
    """
    do moving average for x
    :param x: 2d ndarray , [:,0], [:,1], ... is the data
    :param ma: average num e.g. every 10
    :return: 2d ndarray
    """
    assert len(x.shape) == 2
    m, n = x.shape
    if ma > m:
        print('moving average size too large, change to m:{}'.format(m))
        ma = m
    result = np.zeros((m-ma+1, n))
    for i in range(n):
        result[:, i] = np.convolve(x[:, i], np.ones(ma), 'valid') / ma
    return result


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
