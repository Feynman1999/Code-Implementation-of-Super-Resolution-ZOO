import numpy as np
import os
import sys
import time
from . import util
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """This class includes several functions that can display/save images / videos and print/save logging information.

    It uses a Python library 'visdom' for display.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create images/videos dir for intermediate results display
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.name = opt.name
        self.rgb_mean = list(map(float, opt.normalize_means.split(',')))
        self.rgb_std = list(map(float, opt.normalize_stds.split(',')))

        if opt.phase == "test":
            self.aspect_ratio = opt.aspect_ratio
        else:
            self.aspect_ratio = 1.0

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if opt.phase == "train":
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        elif opt.phase == "test":
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.img_dir = os.path.join(opt.results_dir, opt.name, 'test_{}'.format(load_suffix))
        else:
            raise NotImplementedError("unknown opt.phase")

        print('create %s images/videos directory %s...' % (opt.phase, self.img_dir))
        util.mkdirs([self.img_dir])

        # create a logging file to store training losses
        if opt.phase == "train":
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.opt.display_port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_and_save(self, visuals, epoch):
        """Display current image or video results on visdom.

        :param visuals: dictionary of images to display and save
        :param epoch: the current epoch
        :return: no return
        """

        len_dim = len(list(visuals.values())[0].shape)
        if len_dim == 5:  # video
            if self.display_id > 0:
                 self.display_videos(visuals, epoch)
            self.save_videos(visuals, epoch, aspect_ratio=self.aspect_ratio)
        elif len_dim == 4:  # image
            if self.display_id > 0:
                self.display_images(visuals, epoch)
            self.save_images(visuals, epoch, aspect_ratio=self.aspect_ratio)
        else:
            raise NotImplementedError('visual dim length %d not implemented' % len_dim)

    def display_images(self, visuals, epoch, batch_idx=0):
        '''
            show each image in a separate visdom panel;
        '''
        idx = 1
        try:
            for label, image in visuals.items():
                assert len(image.shape) == 4, 'image dims length should be 4'
                image_numpy = util.tensor2im(image[batch_idx], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)  # [h,w,c]
                self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label+"_epoch_"+str(epoch)),
                               win=self.display_id + idx)  # c,h,w
                idx += 1
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def save_images(self, visuals, epoch_or_name, batch_idx=0, aspect_ratio=1.0):  # 复用一下 test也用
        for label, image in visuals.items():
            assert len(image.shape) == 4, 'image dims length should be 4'
            image_numpy = util.tensor2im(image[batch_idx], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)  # [h,w,c]
            if self.opt.phase == "train":
                img_path = os.path.join(self.img_dir, 'epoch%.4d_%s.png' % (epoch_or_name, label))
            elif self.opt.phase == "test":
                img_path = os.path.join(self.img_dir, '%s_%s.png' % (epoch_or_name, label))
            else:
                raise NotImplementedError("unknown opt.phase")
            util.save_image(image_numpy, img_path, aspect_ratio=aspect_ratio)

    def display_videos(self, visuals, epoch, batch_idx=0):
        '''
            show each video in a separate visdom panel;
        '''
        pass

    def save_videos(self, visuals, epoch_or_name, batch_idx=0, aspect_ratio=1.0):
        for label, video in visuals.items():
            assert len(video.shape) == 5, 'video dims length should be 5'
            video_frames_list = []
            for i in range(video.shape[1]):
                video_frames_list.append(util.tensor2im(video[batch_idx][i], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std))  # [h,w,c]
            if self.opt.phase == "train":
                vid_path = os.path.join(self.img_dir, 'epoch%.4d_%s.avi' % (epoch_or_name, label))
            elif self.opt.phase == "test":
                vid_path = os.path.join(self.img_dir, '%s_%s.avi' % (epoch_or_name, label))
            else:
                raise NotImplementedError("unknown opt.phase")
            util.save_video(video_frames_list, vid_path, aspect_ratio=aspect_ratio)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_and_save_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs  e.g.(loss_G_L1, 0.1111)
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time per data: %.3f, load per data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
