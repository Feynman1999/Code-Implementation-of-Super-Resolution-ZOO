import numpy as np
import os
import sys
import time
from . import util
from iqa import find_function_using_name
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
        self.dis_save_times = 0

        if opt.phase == "test":
            self.factor = opt.factor
            self.iqa_name_list = opt.iqa_list.split(',')
            self.iqa_values = []  # list in list
            for i in range(len(self.iqa_name_list)):
                self.iqa_values.append([])
            self.iqa_dict = dict()
            self.join_str = "   "
            self.iqa_results = []

            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            test_dataset_name = os.path.basename(self.opt.dataroot)
            self.img_dir_name = 'test_{}_{}'.format(test_dataset_name, load_suffix)
            self.result_path = os.path.join(opt.results_dir, opt.name, self.img_dir_name+"_results.txt")
            self.img_dir = os.path.join(opt.results_dir, opt.name, self.img_dir_name)
        elif opt.phase == "train":
            self.factor = 1
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')

        else:
            raise NotImplementedError("unknown opt.phase")

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

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
        """Display current image or video results on visdom.used both in train and test.

        :param visuals: dictionary of images to display and save
        :param epoch: the current epoch for train and the file name for test
        :return: no return
        """
        print("display and saving... the {}th visuals, epoch(train)_or_name(test): {}".format(self.dis_save_times+1, epoch))
        len_dim = len(list(visuals.values())[0].shape)
        if len_dim == 5:  # video
            if self.display_id > 0:
                self.display_videos(visuals, epoch)
            self.save_videos(visuals, epoch, factor=self.factor)
        elif len_dim == 4:  # image
            if self.display_id > 0:
                self.display_images(visuals, epoch)
            self.save_images(visuals, epoch, factor=self.factor)
        else:
            raise NotImplementedError('visual dim length %d not implemented' % len_dim)
        self.dis_save_times += 1

    def cal_iqa(self, visuals, file_name):
        """used in train and test.

        :param visuals: dictionary of images
        :param file_name: file name
        :return:no return
        """
        print("cal iqa for the {}th visuals, file_name: {}".format(self.dis_save_times, file_name))
        assert 'HR_G' in visuals.keys() and 'HR_GroundTruth' in visuals.keys(), 'please make sure that the model has the corresponding attr HR_G and HR_GroundTruth'

        temp_list = []
        for i in range(len(self.iqa_name_list)):
            func = find_function_using_name(self.iqa_name_list[i])
            HR_G = util.tensor2im(visuals['HR_G'][0], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)  # [h,w,c] for image and [b,h,w,c] for video
            HR_GroundTruth = util.tensor2im(visuals['HR_GroundTruth'][0], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)
            val = func(HR_G, HR_GroundTruth, only_Luminance=True, crop=self.opt.SR_factor)
            self.iqa_values[i].append(val)
            temp_list.append("{}: {:.2f}".format(self.iqa_name_list[i], val))
        self.iqa_dict[file_name] = self.join_str.join(temp_list)

    def summary_iqa(self):
        for i in range(len(self.iqa_name_list)):
            result = sum(self.iqa_values[i]) / len(self.iqa_values[i])
            self.iqa_results.append(result)
            fmat ='*'*10 + ' '*10
            print("{}{}: {:.2f}{}".format(fmat, 'average '+self.iqa_name_list[i], result, fmat[::-1]))
        with open(self.result_path, "w+") as log_file:
            now = time.strftime("%c")
            content = '================ Result with {} ({}) ================\n\n'.format(" / ".join(self.iqa_name_list), now)
            content += " "*15
            for i, res in enumerate(self.iqa_results):
                content += "average {}:{:.2f}   ".format(self.iqa_name_list[i], res)
            content += "\n\n"
            for key in sorted(self.iqa_dict.keys()):
                content += "{:^30}:   {}\n".format(key, self.iqa_dict[key])
            log_file.write(content)

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

    def save_images(self, visuals, epoch_or_name, batch_idx=0, factor=1):
        for label, image in visuals.items():
            assert len(image.shape) == 4, 'image dims length should be 4'
            image_numpy = util.tensor2im(image[batch_idx], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)  # [h,w,c]
            if self.opt.phase == "train":
                img_path = os.path.join(self.img_dir, '%.4d_epoch%.4d_%s.png' % (self.dis_save_times+1, epoch_or_name, label))
            elif self.opt.phase == "test":
                img_path = os.path.join(self.img_dir, '%s_%s.png' % (epoch_or_name, label))
            else:
                raise NotImplementedError("unknown opt.phase")
            util.save_image(image_numpy, img_path, factor=factor)

    def display_videos(self, visuals, epoch, batch_idx=0):
        '''
            show each video in a separate visdom panel;
        '''
        pass

    def save_videos(self, visuals, epoch_or_name, batch_idx=0, factor=1):
        for label, video in visuals.items():
            assert len(video.shape) == 5, 'video dims length should be 5'
            video_frames_list = []
            for i in range(video.shape[1]):
                video_frames_list.append(util.tensor2im(video[batch_idx][i], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std))  # [h,w,c]
            if self.opt.phase == "train":
                vid_path = os.path.join(self.img_dir, '%.4d_epoch%.4d_%s.avi' % (self.dis_save_times+1, epoch_or_name, label))
            elif self.opt.phase == "test":
                vid_path = os.path.join(self.img_dir, '%s_%s.avi' % (epoch_or_name, label))
            else:
                raise NotImplementedError("unknown opt.phase")
            util.save_video(video_frames_list, vid_path, factor=factor)

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

    def save_loss_json(self):
        pass