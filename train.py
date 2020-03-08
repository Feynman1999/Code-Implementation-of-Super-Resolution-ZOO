"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, lwsr) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single). (more often, it was set by model)
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates dataset, model and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a lwsr model:
        python train.py --dataroot ./datasets/DIV2k --name DIV_lwsr --model lwsr

    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images / videos in the dataset.
    print('The number of training examples(images or videos) = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt, dataset_size)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_epoch_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time  # Look at the end of the for loop, you will know...
                t_data = t_data / opt.batch_size

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            '''visualize'''
            if total_iters % opt.display_freq == 0:   # display images on visdom
                model.compute_visuals()  # Calculate additional output images for visualization if you need
                visualizer.display_and_save(model.get_current_visuals(), epoch)

            '''loss'''
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_and_save_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        '''save model by epoch'''
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch: %d,  now total iters: %d' % (epoch, total_iters))
            save_prefix = 'epoch_%d' % epoch
            model.save_networks(save_prefix)
            visualizer.save_loss_image(save_prefix)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

    model.save_networks('latest')
