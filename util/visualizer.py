import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import time
from . import util, util_dataset
from iqa import find_function_using_name
from subprocess import Popen, PIPE
from collections import OrderedDict


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """This class includes several functions that can display/save images / videos and print/save logging information.
    and do some checks when training for better visualize.
    It uses a Python library 'visdom' for display.
    """

    def __init__(self, opt, dataset_size=1):
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
        self.output_factor = opt.factor
        self.dataset_size = dataset_size
        self.droped_data = dataset_size % opt.batch_size
        self.dataset_size = self.dataset_size - self.droped_data  # drop last

        if opt.phase == "train":
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')

            # create a logging file to store training losses
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
            # create a help file to store some help information
            self.help_name = os.path.join(opt.checkpoints_dir, opt.name, 'help_log.txt')
            with open(self.help_name, "a") as help_file:
                now = time.strftime("%c")
                help_file.write('================ HELP Information (%s) ================\n' % now)
            self.do_some_checks_for_training_and_print_save_help_info()

        elif opt.phase == "test":
            self.iqa_name_list = opt.iqa_list.split(',')
            self.iqa_values = []  # list in list
            for i in range(len(self.iqa_name_list)):
                self.iqa_values.append([])
            self.iqa_dict = dict()
            self.iqa_results = []

            options = ('ensemble', 'only_Y')
            img_dir_name = self.generate_filename_with_options(options)
            self.iqa_result_path = os.path.join(opt.results_dir, opt.name, img_dir_name+"-results.txt")
            self.img_dir = os.path.join(opt.results_dir, opt.name, img_dir_name)

        elif opt.phase == "apply":
            options = ('block_size', )
            img_dir_name = self.generate_filename_with_options(options)
            self.img_dir = os.path.join(opt.results_dir, opt.name, img_dir_name)

            self.now_deal_file_name_with_suffix = None
            self.last_deal_x = -1
        else:
            raise NotImplementedError("unknown opt.phase")


        print('create %s images/videos directory %s...' % (opt.phase, self.img_dir))
        util.mkdirs([self.img_dir])

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

    def generate_filename_with_options(self, options=()):
        assert self.opt.phase in ('test', 'apply')
        img_dir_name_item_list = []
        img_dir_name_item_list.append(self.opt.phase)  # test
        img_dir_name_item_list.append(util_dataset.get_dataset_name(self.opt.dataroot))  # Set5 for test and someimages for apply
        img_dir_name_item_list.append(self.opt.load_epoch)
        for option in options:
            # print(option)
            img_dir_name_item_list.append("{}_{}".format(option, vars(self.opt)[option]))
        img_dir_name = "-".join(img_dir_name_item_list)
        return img_dir_name

    def do_some_checks_for_training_and_print_save_help_info(self):
        # some checks for better visualize
        opt = self.opt
        assert opt.display_freq % opt.batch_size == 0 and opt.print_freq % opt.batch_size == 0, \
            'please make sure them % batchsize =0 for better understanding of iter(one sample),' \
            'iteration(one batchsize) and epoch(all sample)'

        pre_trained_epochs = opt.epoch_count - 1
        total_epochs = (opt.n_epochs + opt.n_epochs_decay)
        will_train_epochs = total_epochs - pre_trained_epochs

        total_iters = total_epochs * self.dataset_size
        will_train_iters = will_train_epochs * self.dataset_size

        assert will_train_iters / opt.display_freq < 1e4, 'please set opt.display_freq larger, otherwise too many ' \
                                                     'display/save result'

        assert will_train_iters / opt.print_freq < 1e5, 'please set opt.print_freq larger, otherwise too many loss log'

        assert will_train_epochs / opt.save_epoch_freq < 1e2, 'please set opt.save_epoch_freq larger, otherwise too ' \
                                                         'many times save models'

        assert opt.save_epoch_freq * self.dataset_size > opt.print_freq, 'please make sure opt.save_epoch_freq * ' \
                                                                         'self.dataset_size > opt.print_freq, so that when' \
                                                                         'save loss image we have loss data'

        content = ['\n-----------some training information-----------']
        content.append("training dataset size: {} samples(iters), have droped {} iters. batchsize: {} iters.".format(self.dataset_size, self.droped_data, opt.batch_size))
        content.append("total {} samples(iters)  and  {} epoch for this model".format(total_iters, total_epochs))
        content.append("pre-trained {} epochs, will train {} epochs for this time training".format(pre_trained_epochs, will_train_epochs))
        content.append("{:^50}: {} iters and {:.3f} epochs".format("display and save frequency", opt.display_freq, opt.display_freq / self.dataset_size))
        content.append("{:^50}: {} iters and {:.3f} epochs".format("print loss frequency", opt.print_freq, opt.print_freq / self.dataset_size))
        content.append("{:^50}: {} epochs".format("save model frequency", opt.save_epoch_freq))
        content.append('-----------------------------------------------\n')

        # save a help doc
        with open(self.help_name, "a") as help_file:
            help_file.write('%s\n' % "\n".join(content))  # save the message

        print("\n".join(content))

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.opt.display_port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_and_save(self, visuals, epoch):
        """Display current image or video results on visdom and save. used in train, test and apply.

        :param visuals: dictionary of images to display and save
        :param epoch: the current epoch for train and the file name for test or apply
        :return: no return
        """
        identification = "epoch" if self.opt.phase == "train" else "filename"
        print("display and saving the {}th visuals, {}: {}".format(self.dis_save_times+1, identification, epoch))
        len_dim = len(list(visuals.values())[0].shape)
        if len_dim == 5:  # video
            if self.display_id > 0:
                self.display_videos(visuals, epoch)
            self.save_videos(visuals, epoch, factor=self.output_factor)
        elif len_dim == 4:  # image
            if self.display_id > 0:
                self.display_images(visuals, epoch)
            self.save_images(visuals, epoch, factor=self.output_factor)
        else:
            raise NotImplementedError('visual dim length %d not implemented' % len_dim)
        self.dis_save_times += 1

    def save_for_apply(self, visuals, file_name_with_suffix, idx):
        """
        only support one thread now
        :param visuals:
        :param file_name_with_suffix:  xxxxxx__0__1.jpg
        :return:
        """
        def add_block_to_queue():
            for i, (label, image) in enumerate(visuals.items()):
                self.row_queue[i].append(image)

        def blocks_pop_queue():
            for i, item in enumerate(visuals.keys()):
                self.row_list[i].append(torch.cat(self.row_queue[i], dim=-1))
                self.row_queue[i] = []

        def init_lists():
            self.row_list = []  # cat on column at last
            self.row_queue = []  # cat on row to form a row and append to row_list
            for name in visuals.keys():
                self.row_list.append([])
                self.row_queue.append([])

        def make_result():
            blocks_pop_queue()
            visual_ret = OrderedDict()
            for i, name in enumerate(visuals.keys()):
                result = torch.cat(self.row_list[i], dim=-2)
                self.row_list[i] = []
                visual_ret[name] = result
            file_name = os.path.splitext(self.now_deal_file_name_with_suffix)[0]
            self.display_and_save(visuals=visual_ret, epoch=file_name)

        len_dim = len(list(visuals.values())[0].shape)
        if len_dim == 5:  # video
            pass
        elif len_dim == 4:  # image
            file_name = os.path.splitext(file_name_with_suffix)[0]
            suffix = os.path.splitext(file_name_with_suffix)[1]   # e.g.    .jpg
            file_name_split_by__list = file_name.split("__")
            x = int(file_name_split_by__list[-2])
            y = int(file_name_split_by__list[-1])
            origin_file_name_with_suffix = "__".join(file_name_split_by__list[:-2]) + suffix
            if origin_file_name_with_suffix != self.now_deal_file_name_with_suffix:  # change image, create list in list
                assert x == 0 and y == 0
                if self.now_deal_file_name_with_suffix is None:  # this time is the first block on this apply task
                    pass
                else:  # tail deal
                    make_result()
                init_lists()
                add_block_to_queue()
            else:
                if x != self.last_deal_x:  # change row
                    blocks_pop_queue()
                    add_block_to_queue()
                else:  # do not change row
                    add_block_to_queue()
            self.last_deal_x = x
            self.now_deal_file_name_with_suffix = origin_file_name_with_suffix
            if idx+1 == self.dataset_size:
                make_result()  # for the last image
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

    def save_images(self, visuals, epoch_or_name, batch_idx=0, factor=1):
        for label, image in visuals.items():
            assert len(image.shape) == 4, 'image dims length should be 4'
            image_numpy = util.tensor2im(image[batch_idx], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)  # [h,w,c]
            if self.opt.phase == "train":
                img_path = os.path.join(self.img_dir, '%.6d_epoch%.6d_%s.png' % (self.dis_save_times+1, epoch_or_name, label))
            elif self.opt.phase in ("test", "apply"):
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
            video = util.tensor2im(video[batch_idx], rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)  # [b,h,w,c]
            if self.opt.phase == "train":
                vid_path = os.path.join(self.img_dir, '%.6d_epoch%.6d_%s.avi' % (self.dis_save_times+1, epoch_or_name, label))
            elif self.opt.phase in ("test", "apply"):
                vid_path = os.path.join(self.img_dir, '%s_%s.avi' % (epoch_or_name, label))
            else:
                raise NotImplementedError("unknown opt.phase")
            util.save_video(video, vid_path, factor=factor)

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

        if self.opt.display_id > 0:
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

    def save_loss_image(self, save_prefix, moving_average=(10, 100)):
        """
        :param save_prefix: save_prefix is "latest" or "epoch_100"
        :param moving_average: do moving average
        :return:
        """
        save_filename = '%s_loss_moving_average_' % save_prefix  #
        assert hasattr(self, 'plot_data'), 'please make sure already have loss data when save loss image'

        # do moving_average
        X = np.array(self.plot_data['X'])
        Y = np.array(self.plot_data['Y'])
        for ma in moving_average:
            if ma >= Y.shape[0]:
                if self.opt.verbose:
                    print('moving average size{} too large, change to m:{}'.format(ma, max(1, Y.shape[0] - 1)))
                ma = max(1, Y.shape[0] - 1)
            ma_Y = util.moving_average(Y, ma=ma)
            X = np.linspace(X[0], X[-1], ma_Y.shape[0])
            maxn = 1e4
            gap = int(np.ceil(ma_Y.shape[0] / maxn))
            title = "loss from epoch {:.2f} to epoch {:.2f}  moving_average: {}".format(X[0], X[-1], ma)
            plt.figure(figsize=(len(X[::gap])/100+1, 8))
            for i in range(ma_Y.shape[1]):
                plt.plot(X[::gap], ma_Y[:, i][::gap], label=self.plot_data['legend'][i])  # promise that <= 10000
            plt.title(title)
            plt.xlabel('epochs')
            plt.ylabel('Losses')
            plt.legend()
            fig = plt.gcf()
            fig.savefig(os.path.join(self.save_dir, save_filename + str(ma) + '.svg'), dpi=600, bbox_inches='tight')
            # plt.show()
            plt.clf()
            plt.close()

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
            val = func(HR_G, HR_GroundTruth, only_Luminance=self.opt.only_Y, crop=self.opt.SR_factor)
            self.iqa_values[i].append(val)
            temp_list.append("{}: {:.4f}".format(self.iqa_name_list[i], val))
        self.iqa_dict[file_name] = "   ".join(temp_list)

    def summary_iqa(self):
        for i in range(len(self.iqa_name_list)):
            result = sum(self.iqa_values[i]) / len(self.iqa_values[i])
            self.iqa_results.append(result)
            fmat ='*'*10 + ' '*10
            print("{}{}: {:.4f}{}".format(fmat, 'average '+self.iqa_name_list[i], result, fmat[::-1]))
        with open(self.iqa_result_path, "w+") as log_file:
            now = time.strftime("%c")
            content = '================ Result with {} ({}) ================\n\n'.format(" / ".join(self.iqa_name_list), now)
            content += " "*15
            for i, res in enumerate(self.iqa_results):
                content += "average {}: {:.4f}   ".format(self.iqa_name_list[i], res)
            content += "\n\n"
            for key in sorted(self.iqa_dict.keys()):
                content += "{:^30}:   {}\n".format(key, self.iqa_dict[key])
            log_file.write(content)
