from .base_options import BaseOptions
from . import str2bool

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    important:
        --display_freq
        --print_freq
        please make sure them % batchsize =0 for better understanding of iter(one sample),iteration(one batchsize) and epoch(all sample)

        --save_epoch_freq
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # visdom visualization parameters
        parser.add_argument('--display_freq', type=int, default=6400, help='frequency of showing training results on visdom and save to disk, please make sure freq%batchsize =0 ')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom web display, set to >0 use visdom')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        # print loss
        parser.add_argument('--print_freq', type=int, default=6400, help='frequency of showing training results on console, please make sure freq%batchsize =0')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=2000, help='frequency of saving models at the end of epochs')
        parser.add_argument('--continue_train', type=str2bool, default=False, help='continue training: load the trained model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>+<save_epoch_freq>, ...')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=20000, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters epoch when opt.lr_policy == step')

        return parser
