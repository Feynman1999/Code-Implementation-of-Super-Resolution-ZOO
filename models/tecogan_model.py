"""
python train.py --dataroot ./datasets/videodataset --name videodataset_tecogan --model tecogan --display_ncols -1
"""
import torch
from .base_model import BaseModel
from options import str2bool
from . import tecogan_networks
from . import base_networks


class TECOGANModel(BaseModel):
    """ This class implements the TeCoGAN model

    The model training requires '--dataset_mode aligned_video' dataset.

    tecogan paper: https://arxiv.org/abs/1811.09393v3
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
        parser.set_defaults(crop_size=32)
        parser.set_defaults(norm='batch')
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0001)
        parser.set_defaults(init_type='xavier')
        parser.add_argument('--no_pingpong', type=str2bool, default=False, help='if specified, do not use the pingpong loss for data augmentation')
        parser.add_argument('--resblock_num', type=int, default=10, help='the number of ResidualBlock, 10 / 16 in paper')
        parser.add_argument('--maxvel', type=float, default=24.0, help='scales the flow network output to the normal velocity range')
        if is_train:
            parser.set_defaults(gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=100.0, help='weight for L2 loss')
            parser.add_argument('--imgseqlen', type=int, default=10, help='how long sub-string of video frames to train, default is 10')
        else:
            parser.add_argument('--imgseqlen', type=int, default=0, help='how long sub-string of video frames to test, default 0 means all of them')

        return parser

    def __init__(self, opt):
        """Initialize the TeCoGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor
        self.use_pingpong = not opt.no_pingpong

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_Gan', 'G_Dfeature', 'G_VGGfeature', 'G_PP', 'G_warp', 'D_real', 'D_fake']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'F', 'D']  # 'G', 'F', 'D'
        else:  # during test time, only load G
            self.model_names = ['G', 'F']

        self.netG = tecogan_networks.define_G(opt)
        self.netF = tecogan_networks.define_F(opt)

        if self.isTrain:
            self.netD = tecogan_networks.define_D(opt)

        if self.isTrain:
            pass
            # define loss functions
            self.criterionGAN = base_networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_F)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        by default, in video related task, the first frame is black image

        """
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

        # input['A']:   e.g. [4, 11, 3, 128, 128] recurrent training
        # self.LR_reverse = torch.flip(self.LR, dims=(1,))
        # self.LR = torch.cat((self.LR, self.LR_reverse[:, 1:, :, :, :]), 1).to(self.device)  # remove the first frame and cat
        # e.g. [4, 11, 3, 128, 128] -> [4, 21, 3, 128, 128]
        # del self.LR_reverse
        self.LR = input['A']
        self.HR_GroundTruth = input['B']

        self.seq_idx_list = [i for i in range(0, self.LR.shape[1])]
        if self.use_pingpong:
            self.seq_idx_list += [i for i in range(self.LR.shape[1]-2, -1, -1)]


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
           use recurrent training, thus recurrently run model and get a sequence of output and do bp.
        """
        # self.HR_G = self.netG(self.LR)  # G(A)
        # self.HR_copy = nn.functional.interpolate(self.LR, scale_factor=self.SR_factor, mode='bilinear')
        # assert self.HR_G.shape == self.HR_copy.shape
        pass


    def backward_G(self):
        """Calculate loss for the LWSR/G"""
        self.loss_G_L1 = self.criterionL1(self.HR_G, self.HR_GroundTruth) * self.opt.lambda_L1
        self.loss_G_L1.backward()

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