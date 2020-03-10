import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_images_dataset
from PIL import Image
from util import util
from data.base_dataset import make_power_2


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train/A' contains image in domain A, and
    '/path/to/data/train/B' contains image in domain B.You should make sure that their
    filenames are one-to-one after sort.(The names can be different, but they should correspond to each other.
    Of course, we encourage the same filename.)
    During test time, you need to prepare a similar directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.only_HR = opt.only_HR
        self.SR_factor = opt.SR_factor
        if not self.only_HR:
            self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory e.g.      ./DIV2k/train/
            self.dir_A = os.path.join(self.dir_AB, 'A')
            self.dir_B = os.path.join(self.dir_AB, 'B')
            self.A_paths = sorted(make_images_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = sorted(make_images_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
            assert (len(self.A_paths) == len(self.B_paths))
            if ('resize' in opt.preprocess or 'scale_width' in opt.preprocess) and 'crop' in opt.preprocess:
                assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
            self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc  # The default is A->B
            self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        else:
            assert util.check_whether_last_dir(opt.dataroot), 'when only HR, opt.dataroot should be dir and contains only image files'
            self.dir_B = opt.dataroot
            self.B_paths = sorted(make_images_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
            self.input_nc = self.opt.input_nc
            self.output_nc = self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        if not self.only_HR:
            # read a image given a random integer index
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')
        else:
            B_path = self.B_paths[index]
            A_path = B_path
            B = Image.open(B_path).convert('RGB')
            w, h = B.size
            # B = make_power_2(B, self.opt.multi_base) for some train/test dataset if can't % =0 how to deal?
            assert w % self.SR_factor == 0 and h % self.SR_factor == 0, "file:{} w,h should % SR_factor=0".format(B_path)
            A = B.resize((w//self.SR_factor, h//self.SR_factor), resample=Image.BICUBIC)

        if self.opt.direction == 'BtoA' and (not self.only_HR):
            A, B = B, A
            A_path, B_path = B_path, A_path

        assert (B.size[0] >= A.size[0] and B.size[1] >= A.size[1]), 'By default, we think that in general tasks, the image size of target domain B is greater than or equal to source domain A'
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), crop_size_scale=self.opt.SR_factor)

        A = A_transform(A)
        B = B_transform(B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.B_paths)
