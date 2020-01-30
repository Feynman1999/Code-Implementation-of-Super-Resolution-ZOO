"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to disk.

Example (You need to train models first or download pre-trained models from somewhere):

    Test a lwsr model:
        python test.py --dataroot ./datasets/DIV2k --name DIV_lwsr_L1_loss --model lwsr

See options/base_options.py and options/test_options.py for more test options.
"""

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import get_file_name


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling; thus every time test, input the same image
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code only saves the results to disk.
    opt.preprocess = 'none'  # we default do nothing other, just to tensor and normalize

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks(default load "latest"); create schedulers
    visualizer = Visualizer(opt)

    print_frq = 3

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image/video results
        A_paths, B_paths = model.get_image_paths()  # get image/video paths
        if i % print_frq == 0:
            print('processing (%04d)-th image... path: %s' % (i+1, A_paths))
        visualizer.display_and_save(visuals, get_file_name(A_paths[0]))
