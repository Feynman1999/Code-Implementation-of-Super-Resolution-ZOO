from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--factor', type=int, default=1, help='scale factor of result images/videos')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=1000000, help='how many test images to run')
        # iqa
        parser.add_argument('--iqa', action='store_true', help='do Image Quality Assessment')
        parser.add_argument('--iqa_list', type=str, default='psnr,ssim', help='if --iqa, then do the method in the iqa_list')
        self.isTrain = False
        return parser
