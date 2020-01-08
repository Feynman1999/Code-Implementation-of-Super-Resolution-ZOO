import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.video_folder import make_videos_dataset
import cv2

class AlignedVideoDataset(BaseDataset):
    """A dataset class for paired video dataset.

    It assumes that the directory '/path/to/data/train/A' contains video in domain A, and
    '/path/to/data/train/B' contains video in domain B.You should make sure that their
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory e.g.      ./DIV2k/train/
        self.dir_A = os.path.join(self.dir_AB, 'A')
        self.dir_B = os.path.join(self.dir_AB, 'B')
        self.A_paths = sorted(make_videos_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_videos_dataset(self.dir_B, opt.max_dataset_size))  # get video paths
        assert (len(self.A_paths) == len(self.B_paths))
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc  # The default is A->B
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an video in the input domain
            B (tensor) - - its corresponding video in the target domain
            A_paths (str) - - video paths
            B_paths (str) - - video paths
        """
        # read a video given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = read video   A_path   e.g. [120,3,720,540]
        B = pass

        if self.opt.direction == 'BtoA':
            A, B = B, A
            A_path, B_path = B_path, A_path

        assert (B.size[0] >= A.size[0] and B.size[1] >= A.size[1]), 'By default, we think that in general tasks, the image size of target domain B is greater than or equal to source domain A'
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), crop_size_scale=self.opt.SR_factor)

        A = A_transform(A)  # 问： 只能对pillow库中的image对象做?
        B = B_transform(B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
