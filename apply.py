from options.apply_options import ApplyOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util_dataset import get_dataset_name
from tqdm import tqdm
import torch
import os


def cat_blocks(image_list, block_size):
    assert len(image_list[0].shape) == 4
    h = block_size[0]
    w = block_size[1]
    row_list = []
    id = 0
    for i in range(h):
        row_list.append(torch.cat(image_list[id:id + w], dim=-1))
        id += w
    return torch.cat(row_list, dim=-2)


if __name__ == '__main__':
    opt = ApplyOptions().parse()

    # hard-code some parameters for apply
    opt.dataset_mode = "single_video" if opt.video_flag else 'single'
    opt.batch_size = 1  # apply code only supports batch_size = 1
    opt.num_threads = 6  # apply code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.no_rotate = True
    opt.display_id = -1  # no visdom display; the apply code only saves the results to disk.
    opt.preprocess = 'none'  # we default do nothing other, just to tensor and normalize

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks(default load "latest"); create schedulers
    visualizer = Visualizer(opt, len(dataset))

    if opt.eval:
        model.eval()

    now_deal_frame = 0  # for video
    block_size = opt.block_size.split("_")  # "2_3"
    block_size = list(map(int, block_size))
    block_list = []

    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:
            break
        if len(data['A'].shape) == 4:  # image
            assert opt.dataset_mode == 'single'
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image/video results
            A_paths, _ = model.get_image_paths()  # get image/video paths
            file_name_with_suffix = os.path.basename(A_paths[0])  # xxxxxx__0__1.jpg     __0__1 help to locate block
            visualizer.save_for_apply(visuals, file_name_with_suffix, i)
        else:  # video
            assert opt.dataset_mode == 'single_video'
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()  # get image/video results
            block_list.append(visuals['HR_G'].cpu())
            if len(block_list) == block_size[0] * block_size[1]:
                A_paths, _ = model.get_image_paths()  # get image/video paths
                video_name = get_dataset_name(A_paths[0])
                visuals['HR_G'] = cat_blocks(block_list, block_size)
                block_list = []
                visualizer.display_and_save(visuals, os.path.join(video_name, '%.6d' % now_deal_frame))
                if data['end_flag']:
                    now_deal_frame = 0
                    print("video: {} is ok!".format(video_name))
                else:
                    now_deal_frame += 1
