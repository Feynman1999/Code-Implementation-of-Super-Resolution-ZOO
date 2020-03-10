"""
for apply.py

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
        self.gap = opt.block_size

        assert util.check_whether_last_dir(opt.dataroot), 'when apply, opt.dataroot:{} should be dir and contains only image files'.format(opt.dataroot)
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_images_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.A_sizes = get_images_size(self.A_paths)
        self.bucket_expect = []
        for i, size in enumerate(self.A_sizes):
            w, h = size
            self.bucket_expect.append(math.ceil(w/self.gap) * math.ceil(h/self.gap))
            print("will divide {} to {} blocks".format(self.A_paths[i], self.bucket_expect[-1]))
        self.bucket = [0] * len(self.A_paths)  # bucket for record cal times
        self.now_do_id = 0

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.transform = get_transform(opt, grayscale=(self.input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing  [0,  sum(self.bucket_expect) )

        Returns a dictionary that contains A and A_paths
            A (tensor) - - an image in the input domain
            A_paths (str) - - image paths
        """
        # read a image given a integer index
        A_path = self.A_paths[self.now_do_id]
        block_idx = self.bucket[self.now_do_id]
        w, h = self.A_sizes[self.now_do_id]
        nw, nh = math.ceil(w/self.gap), math.ceil(h/self.gap)
        assert nw * nh == self.bucket_expect[self.now_do_id]

        A_img = Image.open(A_path).convert('RGB')
        left = (block_idx % nw) * self.gap
        up = (block_idx // nw) * self.gap

        if (block_idx + 1) % nw == 0:
            ow = w % self.gap
            if ow == 0:
                ow = self.gap
        else:
            ow = self.gap

        if (block_idx // nw +1) == nh:
            oh = h % self.gap
            if oh == 0:
                oh = self.gap
        else:
            oh = self.gap

        right = left + ow
        below = up + oh
        box = (left, up, right, below)
        A_img = A_img.crop(box)
        assert ow == A_img.size[0] and oh == A_img.size[1]

        A = self.transform(A_img)
        B = self.transform(A_img.resize((ow*self.SR_factor, oh*self.SR_factor), resample=Image.BICUBIC))


        file_name = util_dataset.get_file_name(A_path)
        dir_name = os.path.dirname(A_path)
        file_suffix = os.path.splitext(A_path)[1]
        A_path = os.path.join(dir_name, file_name + "__{}__{}{}".format(block_idx // nw, block_idx % nw, file_suffix))

        # tail deal
        self.bucket[self.now_do_id] += 1
        if self.bucket[self.now_do_id] == self.bucket_expect[self.now_do_id]:
            self.now_do_id += 1
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return sum(self.bucket_expect)
