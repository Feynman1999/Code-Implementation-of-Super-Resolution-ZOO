import torch
from .base_model import BaseModel
from . import lwsr_networks


class LWSRModel(BaseModel):
    """ This class implements the LWSR model

    The model training requires '--dataset_mode aligned' dataset.

    LWSR paper: https://arxiv.org/pdf/1909.10774.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The training objective is: std L1 loss
        """
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(batch_size=16)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(SR_factor=4)
        parser.set_defaults(normalize_means='0.4488,0.4371,0.4040')
        parser.set_defaults(crop_size=48)
        parser.set_defaults(norm='batch')
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(netD='n_layers')
        parser.set_defaults(n_layers_D=2)
        parser.set_defaults(lr=0.0001)
        parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')
        if is_train:
            parser.set_defaults(gan_mode='')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=100.0, help='weight for L2 loss')

        return parser

    def __init__(self, opt):
        """Initialize the LWSR class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['G_L1']  # please use loss_G_L1 below

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        self.netG = lwsr_networks.define_lwsr_net(opt)

        # if self.isTrain:
        #     self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
        #                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.LR = input['A'].to(self.device)
        self.HR_GroundTruth = input['B'].to(self.device)
        # print(self.LR.shape, self.HR_GroundTruth.shape)
        self.A_paths = input['A_paths']  # list
        self.B_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.HR_G = self.netG(self.LR)  # G(A)
        # self.HR_copy = nn.functional.interpolate(self.LR, scale_factor=self.SR_factor, mode='bilinear')
        # assert self.HR_G.shape == self.HR_copy.shape

    def backward_G(self):
        """Calculate loss for the LWSR/G"""
        self.loss_G_L1 = self.criterionL1(self.HR_G, self.HR_GroundTruth) * self.opt.lambda_L1
        self.loss_G_L1.backward()

    # def backward_D(self):
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
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights