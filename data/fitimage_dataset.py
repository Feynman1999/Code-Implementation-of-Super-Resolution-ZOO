import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_images_dataset
from PIL import Image
from util import util, util_dataset
import random
import torch


class FitimageDataset(BaseDataset):
    """A dataset class for fitting image. for example
    "Implicit Neural Representations with Periodic Activation Functions"   arXiv:2006.09661v1 [cs.CV] 17 Jun 2020

    It assumes that the opt.dataroot is the path to image.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        self.path2image = opt.dataroot  # get the image path
        self.img = Image.open(self.path2image).convert('RGB')
        self.background_img = Image.new('RGB', self.img.size, (255, 255, 255))

        transform_params = get_params(self.opt, self.img.size)
        print(transform_params['crop_pos'])
        transform_params['flip'] = False
        transform_params['rotate'] = False
        transform = get_transform(self.opt, transform_params, grayscale=(opt.input_nc == 1))

        self.img = transform(self.img)
        self.background_img = transform(self.background_img)

        channels, rows, cols = self.img.shape
        sampled_pixel_count = int(rows * cols / opt.Reduction_factor)

        loc_list = []
        for i in range(rows):
            for j in range(cols):
                loc_list.append([i, j])

        self.sampled_loc_list = random.sample(loc_list, sampled_pixel_count)
        print("random sample 1/{} ok!".format(opt.Reduction_factor))

        for loc in self.sampled_loc_list:
            self.background_img[:, loc[0], loc[1]] = self.img[:, loc[0], loc[1]]

        if ('resize' in opt.preprocess or 'scale_width' in opt.preprocess) and 'crop' in opt.preprocess:
            assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

        self.input_nc = self.opt.input_nc  # The default is A->B
        self.output_nc = self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        A = self.sampled_loc_list[index]
        return {'A': torch.tensor(A, dtype=torch.float32), 'B': self.img[:, A[0], A[1]]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.sampled_loc_list)
