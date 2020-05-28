from .base_options import BaseOptions
from . import str2bool

class ApplyOptions(BaseOptions):
    """This class includes apply options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='apply', help='train, val, test, etc')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--block_size', type=str, default='1_1', help='for save memory, we make blocks by 256*256, set lower for poor performance machine! for video we use 2_3')
        parser.add_argument('--video_flag', type=str2bool, default=True, help='deal for video or image')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', type=str2bool, default=False, help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1000000, help='how many test images(divided blocks) to run (upper_bound)')

        return parser
