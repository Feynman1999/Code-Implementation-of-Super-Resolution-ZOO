"""This module contains simple helper functions """
from __future__ import print_function
import torch
import ntpath
import numpy as np
from PIL import Image
from data.image_folder import make_images_dataset
import os
import cv2


def tensor2im(input_image, rgb_mean = (0.5, 0.5, 0.5), rgb_std = (1.0, 1.0, 1.0)):
    """"Converts a Tensor array into a numpy image array. [h,w,c]

    Parameters:
        input_image (tensor) --  the input image tensor array,  without batchsize dim
    """
    assert (len(input_image.shape) == 3), 'input_image should be 3 dims'
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image  # who knows what is it  /(ㄒoㄒ)/~~
        image_tensor = image_tensor.cpu().float()
        if image_tensor.shape[0] == 1:  # grayscale to RGB
            image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
        # normalize
        image_tensor = image_tensor * torch.tensor(rgb_std).view(3, 1, 1) + torch.tensor(rgb_mean).view(3, 1, 1)
        # clamp to [0,255]  or min max map to 0~255 is better?
        image_tensor = image_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        image_numpy = image_tensor.numpy()  # convert it into a numpy array.

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((int(w * aspect_ratio)), h, Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((w, int(h / aspect_ratio)), Image.BICUBIC)
    image_pil.save(image_path)


def save_video(video_frames_list, video_path, aspect_ratio=1.0, fps=2):
    '''
        Save a numpy image list (video) to the disk
    :param video_frames_list:  rgb numpy image list
    :param video_path:
    :param aspect_ratio:
    :param fps:
    :return:  none
    '''
    h, w, _ = video_frames_list[0].shape
    # notice that in general task, idx 0 is all black...
    # print_numpy(video_frames_list[1])
    if aspect_ratio > 1.0:
        w = int(w * aspect_ratio)

    if aspect_ratio < 1.0:
        h = int(h / aspect_ratio)

    if aspect_ratio != 1.0:
        for i in range(len(video_frames_list)):
            video_frames_list[i] = cv2.resize(video_frames_list[i], (w, h), interpolation=cv2.INTER_CUBIC)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'I420'), fps, (w, h))
    for frame in video_frames_list:
        out.write(frame[..., ::-1])
    out.release()


def images2video(filepath, fps=12, Suffix = '.avi'):
    imagepathlist = make_images_dataset(filepath)
    framelist = []
    for imgpath in imagepathlist:
        framelist.append(cv2.imread(imgpath)[..., ::-1])
    dn = os.path.dirname(filepath)
    name = os.path.split(filepath)[-1] + Suffix
    save_video(framelist, os.path.join(dn, name), fps=fps)


def dataset_images2video(filepath, fps=12):
    for home, dirs, files in sorted(os.walk(filepath)):
        for dir_ in dirs:
            dir_ = os.path.join(home, dir_)
            print(dir_)
            images2video(dir_, fps=fps)

        # print("#######file list#######")
        # for filename in files:
        #     print(filename)
        #     fullname = os.path.join(home, filename)
        #     print(fullname)
        # print("#######file list#######")


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


def get_file_name(path):
    '''
        datasets/div2k/train/A/0001.jpg  ->  0001
    :param path: the path
    :return: the name
    '''
    short_path = ntpath.basename(path)  # get file name, it is name from domain A
    name = os.path.splitext(short_path)[0]  # Separating file name from extensions
    return name


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
