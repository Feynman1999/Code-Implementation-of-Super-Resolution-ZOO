import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torch
from data.image_folder import make_images_dataset
from PIL import Image
import pickle


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

        self.A_paths = sorted(os.listdir(self.dir_A))
        self.B_paths = sorted(os.listdir(self.dir_B))
        # max_dataset_size
        self.A_paths = self.A_paths[:min(opt.max_dataset_size, len(self.A_paths))]
        self.B_paths = self.B_paths[:min(opt.max_dataset_size, len(self.B_paths))]

        assert (len(self.A_paths) == len(self.B_paths))
        assert opt.imgseqlen <= opt.max_consider_len  # when train

        if ('resize' in opt.preprocess or 'scale_width' in opt.preprocess) and 'crop' in opt.preprocess:
            assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        def deal_scene_list(List, max_len):
            l = []
            if max_len > List[-1][1]+1:
                max_len = List[-1][1]+1
            for scene in List:
                if scene[1]+1 >= max_len:
                    l.append([scene[0], max_len-1])
                    break
                else:
                    l.append(scene)
            return l

        if opt.scenedetect:
            # 读取dataset文件夹中的分段信息(train/scene.json)，如果没有则报错
            # 列表 套 列表 套 列表
            pickle_path = os.path.join(self.dir_AB, 'scene.pickle')
            assert os.path.exists(pickle_path)
            with open(pickle_path, 'rb') as f:
                self.scene = pickle.load(f)
            # 根据max consider len进行修改
            for i in range(len(self.scene)):
                self.scene[i] = deal_scene_list(self.scene[i], opt.max_consider_len)

    def get_image_list(self, A_path, B_path, video_index):
        """

        :param short_path:
        :return: a list of PIL.Image
        """
        A_path = os.path.join(self.dir_A, A_path)
        B_path = os.path.join(self.dir_B, B_path)
        A_img_paths = make_images_dataset(A_path)
        B_img_paths = make_images_dataset(B_path)

        if self.opt.imgseqlen > 0:
            if self.opt.scenedetect:
                l = self.scene[video_index]
                scene_id = random.randint(0, len(l)-1)
                times = 0
                while l[scene_id][1] - l[scene_id][0]+1 < self.opt.imgseqlen:
                    times += 1
                    if times > 20:
                        raise ValueError("may be do not have scene satisfy imgseqlen, imgseqlen too large?")
                    scene_id = random.randint(0, len(l) - 1)
                # 在当前scene_id中选择一个片段
                start_id = random.randint(l[scene_id][0], l[scene_id][1]+1-self.opt.imgseqlen)
                A_img_paths = A_img_paths[start_id: start_id + self.opt.imgseqlen]
                B_img_paths = B_img_paths[start_id: start_id + self.opt.imgseqlen]

            else:
                assert self.opt.max_consider_len <= len(A_img_paths)
                A_img_paths = A_img_paths[:self.opt.max_consider_len]
                B_img_paths = B_img_paths[:self.opt.max_consider_len]
                start_id = random.randint(0, self.opt.max_consider_len-self.opt.imgseqlen)
                A_img_paths = A_img_paths[start_id: start_id + self.opt.imgseqlen]
                B_img_paths = B_img_paths[start_id: start_id + self.opt.imgseqlen]

        A = []
        B = []
        for path in A_img_paths:
            A.append(Image.open(path).convert('RGB'))
        for path in B_img_paths:
            B.append(Image.open(path).convert('RGB'))
        return A, B

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
        A, B = self.get_image_list(A_path, B_path, index)

        # some checks
        assert (len(A) == len(B))
        for i in range(len(A)):
            assert (B[i].size[0] >= A[i].size[0] and B[i].size[1] >= A[i].size[1]), 'By default, we think that in general tasks, the image size of target domain B is greater than or equal to source domain A'
        for i in range(len(A)):
            assert (B[i].size[0] == self.opt.SR_factor * A[i].size[0] and B[i].size[1] == self.opt.SR_factor * A[i].size[1]), 'the dataset should satisfy the sr_factor {}'.format(self.opt.SR_factor)


        # by default, we add an black image to the start of list A, B
        # black_img_A = Image.fromarray(np.zeros((A[0].size[1], A[0].size[0], self.input_nc), dtype=np.uint8))  # h w c
        # black_img_B = Image.fromarray(np.zeros((B[0].size[1], B[0].size[0], self.input_nc), dtype=np.uint8))  # h w c
        # A.insert(0, black_img_A)
        # B.insert(0, black_img_B)

        transform_params = get_params(self.opt, A[0].size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), domain="A")
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), domain="B")

        for i in range(len(A)):
            # print("doing transform..the {}th frame of {}th video".format(i, index))
            A[i] = A_transform(A[i])
            B[i] = B_transform(B[i])

        # list of 3dim to 4dim  e.g. ... [3,128,128] ... to [11,3,128,128]
        A = torch.stack(A, 0)
        B = torch.stack(B, 0)

        return {'A': A, 'B': B, 'A_paths': os.path.join(self.dir_A, A_path), 'B_paths': os.path.join(self.dir_B, B_path)}

    def __len__(self):
        """Return the total number of videos in the dataset."""
        return len(self.B_paths)
