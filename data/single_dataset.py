"""
for image apply
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_images_dataset, get_images_size
from PIL import Image
from util import util, util_dataset
import math
import os


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        assert util_dataset.check_whether_last_dir(opt.dataroot), 'when SingleDataset, opt.dataroot:{} should be dir and contains only image files'.format(opt.dataroot)
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_images_dataset(self.dir_A, opt.max_dataset_size))  # get image paths

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing  [0,  sum(self.bucket_expect) )

        Returns a dictionary that contains A and A_paths
            A (tensor) - - an image in the input domain
            A_paths (str) - - image paths
        """
        # read a image given a integer index
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        transform = get_transform(self.opt, grayscale=(self.input_nc == 1))
        A = transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
