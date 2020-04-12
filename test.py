"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to disk.

Example (You need to train models first or download pre-trained models from somewhere):

    Test a rbpn model:
        python test.py --dataroot ./datasets/Vid4  --name vimeo_rbpn  --suffix 04_05_13_46  --model rbpn --load_epoch epoch_65  --ensemble true  --only_Y  true

See options/base_options.py and options/test_options.py for more test options.
"""

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util_dataset import get_file_name
from util import ensemble
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling; thus every time test, input the same image
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.no_rotate = True  # no rotate; comment this line if results on rotated images are needed.
    opt.display_id = -1  # no visdom display; the test code only saves the results to disk.
    opt.preprocess = 'none'  # we default do nothing other, just to tensor and normalize
    opt.remove_first = 6
    opt.remove_last = 3

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks(default load "latest"); create schedulers
    visualizer = Visualizer(opt)


    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):  # data is a dict
        if i >= opt.num_test:
            break
        print("now dealing {}th data".format(i))
        # image
        if len(data['A'].shape) == 4:
            if opt.ensemble:  # only for image
                LR = data['A']
                LR_list = ensemble.ensemble(LR)  # torch tensor, [1,C,H,W] for image and [1,F,C,H,W] for video
                HR_list = []
                for i in range(len(LR_list)):
                    data['A'] = LR_list[i]
                    model.set_input(data)
                    model.test(compute_visual_flag=(i == 0))  # only for original LR, we compute additional visuals e.g. bicubic
                    HR_list.append(model.HR_G)  # [1,C,H,W] for image and video
                del LR_list
                HR_list = ensemble.ensemble_inverse(HR_list)
                HR = torch.cat(HR_list, dim=0)
                del HR_list
                HR = HR.mean(dim=0, keepdim=True)
                visuals = model.get_current_visuals()  # get image/video results
                visuals["HR_G"] = HR
                visuals["LR"] = LR

            else:
                model.set_input(data)  # unpack data from data loader
                model.test(compute_visual_flag=True)  # run inference
                visuals = model.get_current_visuals()  # get image/video results

            A_paths, B_paths = model.get_image_paths()  # get image/video paths
            file_name = get_file_name(A_paths[0])
            visualizer.display_and_save(visuals, file_name)
            visualizer.cal_iqa(visuals, file_name)

        # video
        elif len(data['A'].shape) == 5:
            if opt.ensemble:
                raise NotImplementedError("for video, we did not implement ensemble")
            else:
                remove_first = opt.remove_first
                remove_last = opt.remove_last
                LR = data['A']
                HR = data['B']
                HR_G_list = []
                HR_bicubic_list = []
                assert LR.shape[1] > remove_first + remove_last
                window_size = opt.nframes
                assert window_size % 2 == 1, 'window size should be odd'
                for target_frame_idx in range(remove_first, LR.shape[1] - remove_last):
                    data['A'] = LR[:, target_frame_idx - window_size//2: target_frame_idx + window_size//2 + 1, ...]
                    data['B'] = HR[:, target_frame_idx - window_size//2: target_frame_idx + window_size//2 + 1, ...]
                    model.set_input(data)
                    model.test(compute_visual_flag=True)
                    HR_G_list.append(model.HR_G)
                    HR_bicubic_list.append(model.HR_Bicubic)
                visuals = model.get_current_visuals()
                visuals["LR"] = LR[:, remove_first: -remove_last, ...]
                visuals["HR_GroundTruth"] = HR[:, remove_first: -remove_last, ...]
                visuals["HR_G"] = torch.stack(HR_G_list, dim=1)
                visuals["HR_Bicubic"] = torch.stack(HR_bicubic_list, dim=1)

            A_paths, B_paths = model.get_image_paths()  # get image/video paths
            file_name = get_file_name(A_paths[0])
            visualizer.display_and_save(visuals, file_name)
            visualizer.cal_iqa(visuals, file_name)
    visualizer.summary_iqa()
