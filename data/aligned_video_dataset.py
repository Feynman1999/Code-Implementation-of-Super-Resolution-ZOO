import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.video_folder import make_videos_dataset, read_video
import torch
from util import util, util_dataset


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
        self.SR_factor = opt.SR_factor

        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory e.g.      ./DIV2k/train/
        self.dir_A = os.path.join(self.dir_AB, 'A')
        self.dir_B = os.path.join(self.dir_AB, 'B')

        self.A_paths = sorted(make_videos_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_videos_dataset(self.dir_B, opt.max_dataset_size))  # get video paths
        assert (len(self.A_paths) == len(self.B_paths))
        if ('resize' in opt.preprocess or 'scale_width' in opt.preprocess) and 'crop' in opt.preprocess:
            assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc  # The default is A->B
        self.output_nc = self.opt.output_nc

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
        A = read_video(A_path)  # a list of PIL.image
        B = read_video(B_path)

        # some checks
        assert (len(A) == len(B))
        for i in range(len(A)):
            assert (B[i].size[0] >= A[i].size[0] and B[i].size[1] >= A[i].size[1]), 'By default, we think that in general tasks, the image size of target domain B is greater than or equal to source domain A'
        for i in range(len(A)):
            assert (B[i].size[0] == self.opt.SR_factor * A[i].size[0] and B[i].size[1] == self.opt.SR_factor * A[i].size[1]), 'the dataset should satisfy the sr_factor {}'.format(self.opt.SR_factor)

        # Capture the substring of video sequence
        if self.opt.imgseqlen > 0:
            assert self.opt.imgseqlen <= len(A), 'images sequence length for train should less than or equal to length of all images'
            start_id = random.randint(0, len(A)-self.opt.imgseqlen)
            A = A[start_id: start_id + self.opt.imgseqlen]
            B = B[start_id: start_id + self.opt.imgseqlen]

        # by default, we add an black image to the start of list A, B
        # black_img_A = Image.fromarray(np.zeros((A[0].size[1], A[0].size[0], self.input_nc), dtype=np.uint8))  # h w c
        # black_img_B = Image.fromarray(np.zeros((B[0].size[1], B[0].size[0], self.input_nc), dtype=np.uint8))  # h w c
        # A.insert(0, black_img_A)
        # B.insert(0, black_img_B)

        transform_params = get_params(self.opt, A[0].size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), crop_size_scale=self.opt.SR_factor)

        for i in range(len(A)):
            # print("doing transform..the {}th frame of {}th video".format(i, index))
            A[i] = A_transform(A[i])
            B[i] = B_transform(B[i])

        # list of 3dim to 4dim  e.g. ... [3,128,128] ... to [11,3,128,128]
        A = torch.stack(A, 0)
        B = torch.stack(B, 0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of videos in the dataset."""
        return len(self.B_paths)
