import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torch
from data.image_folder import make_images_dataset
from PIL import Image
import pickle
import copy


class SingleVideoDataset(BaseDataset):
    """A dataset class for single video dataset when apply.

    It assumes that the directory '/path/to/data/' contains images dir for video in domain A.

    only support dir style now.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.block_size = opt.block_size.split("_")  # "2_3"
        self.block_size = list(map(int, self.block_size))
        self.SR_factor = opt.SR_factor
        self.dir_A = opt.dataroot
        self.A_paths = sorted(os.listdir(self.dir_A))
        # max_dataset_size
        self.A_paths = self.A_paths[:min(opt.max_dataset_size, len(self.A_paths))]

        self.length_for_videos = []
        for path in self.A_paths:
            A_path = os.path.join(self.dir_A, path)
            length = len(make_images_dataset(A_path))
            # length = len(make_images_dataset(A_path)) - opt.nframes + 1
            self.length_for_videos.append(length)

        self.now_deal_video_index = 0

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        if opt.scenedetect:
            # 读取dataset文件夹中的分段信息(train/scene.json)，如果没有则报错
            # 列表 套 列表 套 列表
            pickle_path = os.path.join(os.path.split(self.dir_A)[0], 'scene.pickle')
            assert os.path.exists(pickle_path)
            with open(pickle_path, 'rb') as f:
                self.scene = pickle.load(f)

    def get_image_list(self, A_path, index):
        assert index >= 0
        A_path = os.path.join(self.dir_A, A_path)
        A_img_paths = make_images_dataset(A_path)

        if self.opt.scenedetect:
            for one_scene in self.scene[self.now_deal_video_index]:
                if one_scene[0] <= index and index <= one_scene[1]:
                    A_img_paths = A_img_paths[one_scene[0]:one_scene[1]+1]
                    index = index - one_scene[0]
                    break

        # assert len(A_img_paths) >= self.opt.nframes // 2 + 1
        if len(A_img_paths) < self.opt.nframes // 2 + 1:
            gap = self.opt.nframes // 2 + 1 - len(A_img_paths)
            if gap % 2:
                A_img_paths = [A_img_paths[0], ] * (gap//2) + A_img_paths + [A_img_paths[-1], ] * (gap//2+1)
            else:
                A_img_paths = [A_img_paths[0], ] * (gap // 2) + A_img_paths + [A_img_paths[-1], ] * (gap // 2)

        A_img_paths_front = A_img_paths[1: 1+self.opt.nframes//2]
        A_img_paths_front.reverse()
        A_img_paths_back = A_img_paths[-1-self.opt.nframes//2: -1]
        A_img_paths_back.reverse()
        A_img_paths = A_img_paths_front + A_img_paths + A_img_paths_back

        A = []

        for path in A_img_paths[index: index + self.opt.nframes]:
            A.append(Image.open(path).convert('RGB'))

        assert len(A) == self.opt.nframes
        return A

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing   [ 0, sum(self.length_for_videos) )

        Returns a dictionary that contains A and A_paths
            A (tensor) - - an video in the input domain
            A_paths (str) - - video paths
            end_flag (bool) - - whether end frame for a video
        """
        # read a video given a random integer index
        A_path = self.A_paths[self.now_deal_video_index]
        frame_index = index//(self.block_size[0] * self.block_size[1])
        block_index = index % (self.block_size[0] * self.block_size[1])
        block_index_h = block_index // self.block_size[1]
        block_index_w = block_index % self.block_size[1]

        if block_index == 0:
            self.A = self.get_image_list(A_path, frame_index - sum(self.length_for_videos[0:self.now_deal_video_index]))
            w, h = self.A[0].size
            self.block_h = list(range(0, h+1, h//self.block_size[0]))
            if self.block_h[-1] != h:
                self.block_h[-1] = h
            self.block_w = list(range(0, w+1, w//self.block_size[1]))
            if self.block_w[-1] != w:
                self.block_w[-1] = w

        A = []
        box = (self.block_w[block_index_w], self.block_h[block_index_h], self.block_w[block_index_w + 1], self.block_h[block_index_h + 1])
        # blocking from self.A
        for frame in self.A:
            A.append(frame.crop(box))

        w, h = A[0].size
        gt_h_w = (h*self.SR_factor, w*self.SR_factor)

        trans = get_transform(self.opt, grayscale=(self.input_nc == 1))

        for i in range(len(A)):
            A[i] = trans(A[i])

        # list of 3dim to 4dim  e.g. ... [3,128,128] ... to [11,3,128,128]
        A = torch.stack(A, 0)

        end_flag = False
        if frame_index + 1 == sum(self.length_for_videos[0:self.now_deal_video_index+1]) and block_index+1 == (self.block_size[0] * self.block_size[1]):
            self.now_deal_video_index += 1
            end_flag = True

        return {'A': A, 'A_paths': os.path.join(self.dir_A, A_path), 'end_flag': end_flag, 'gt_h_w': gt_h_w}

    def __len__(self):
        """Return the total number of videos in the dataset."""
        return sum(self.length_for_videos) * self.block_size[0] * self.block_size[1]
