"""
python train.py --dataroot ./datasets/youku --name youku_frvsr --model frvsr
"""
import torch
from .base_model import BaseModel
from . import frvsr_networks
from . import base_networks

class FRVSRModel(BaseModel):
    """ This class implements the frvsr model

    The model training requires '--dataset_mode aligned_video' dataset.

    frvsr paper: https://arxiv.org/abs/1801.04590
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(dataset_mode='aligned_video')
        parser.set_defaults(batch_size=4)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(SR_factor=4)
        parser.set_defaults(normalize_means='0.5,0.5,0.5')
        parser.set_defaults(crop_size=64)
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0001)
        parser.set_defaults(init_type='xavier')
        parser.add_argument('--resblock_num', type=int, default=10, help='the number of ResidualBlock, 10 in paper')
        if is_train:
            parser.add_argument('--lambda_L2', type=float, default=1.0, help='weight for L2 loss')
            parser.add_argument('--imgseqlen', type=int, default=10, help='how long sub-string of video frames to train, default is 10')
        else:
            parser.add_argument('--imgseqlen', type=int, default=0, help='how long sub-string of video frames to test, default 0 means all of them')

        return parser

    def __init__(self, opt):
        """Initialize the frvsr class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['SR', 'flow']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = [] # ['F', 'SR']
        else:
            self.model_names = ['F', 'SR']

        # self.netSR = frvsr_networks.define_SR(opt)
        # self.netF = frvsr_networks.define_F(opt)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_SR = torch.optim.Adam(self.netSR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_SR)
            # self.optimizers.append(self.optimizer_F)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        by default, in video related task, the first frame is black image

        """
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

        # input['A']:   e.g. [4, 11, 3, 64, 64] for recurrent training

        self.LR = input['A']
        self.HR_GroundTruth = input['B']
        self.seq_idx_list = [i for i in range(0, self.LR.shape[1])]


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
           use recurrent training, thus recurrently run model and get a sequence of output and do bp.
        """
        # self.HR_G = self.netG(self.LR)  # G(A)
        # self.HR_copy = nn.functional.interpolate(self.LR, scale_factor=self.SR_factor, mode='bilinear')
        # assert self.HR_G.shape == self.HR_copy.shape
        self.HR_G = self.HR_GroundTruth


    def backward_G(self):
        """Calculate loss for the LWSR/G"""
        pass

    def backward_D(self):
        pass
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB = torch.cat((self.HR_copy, self.HR_G), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Real
    #     real_AB = torch.cat((self.HR_copy, self.HR_GroundTruth), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        # # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        # self.optimizer_G.zero_grad()        # set G's gradients to zero
        # self.backward_G()                   # calculate graidents for G
        # self.optimizer_G.step()             # udpate G's weights