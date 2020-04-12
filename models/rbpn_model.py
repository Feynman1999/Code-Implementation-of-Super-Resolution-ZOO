"""
python train.py --dataroot ./datasets/Vid4 --name Vid4_rbpn --model rbpn  --display_freq  4  --print_freq  4

aimax:
    gpu:
    python3 train.py
        --dataroot          /opt/data/private/datasets/vimeo_septuplet
        --name              vimeo_rbpn
        --model             rbpn
        --display_freq      2400
        --print_freq        2400
        --save_epoch_freq   5
        --gpu_ids           0,1,2
        --batch_size        6
        --suffix            04_05_13_46
        --continue_train    True
        --load_epoch        epoch_40
        --epoch_count       41
"""
import torch
from .base_model import BaseModel
from . import rbpn_networks


class RBPNModel(BaseModel):
    """ This class implements the RBPN model

    The model training requires '--dataset_mode aligned_video' dataset.

    rbpn paper: arXiv:1903.10128v1 [cs.CV] 25 Mar 2019

    Here we only use the PF(past and future) strategy, so best to use odd numbers for imgseqlen (3,5,7).

    Here we do not use optical flow.

    vimeo90K train dataset size: 55025.

    for imgseqlen 5, will cost 8200 seconds for one epoch with 3 GPUs (TITAN_X_Pascal). (14 days for 150 epoch)
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
        parser.set_defaults(batch_size=4)  # 8 in paper  need 4 gpu
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(SR_factor=4)
        parser.set_defaults(crop_size=64)
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0001)
        parser.set_defaults(init_type='kaiming')
        parser.set_defaults(lr_policy='step')
        parser.set_defaults(lr_decay_iters=75)
        parser.set_defaults(n_epochs=150)
        parser.add_argument('--cl', type=int, default=256, help='the cl in paper')
        parser.add_argument('--cm', type=int, default=256, help='the cm in paper')
        parser.add_argument('--ch', type=int, default=64, help='the ch in paper')
        # used for assert, imgseqlen should set equal to this when train, normally not change
        parser.add_argument('--nframes', type=int, default=5, help='frames used by model')

        return parser

    def __init__(self, opt):
        """Initialize the RBPN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['SR']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G', 'HR_Bicubic']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']

        self.netG = rbpn_networks.define_G(opt)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        by default, in video related task, the first frame is black image

        """
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

        # input['A']:   e.g. [4, 10, 3, 64, 64] for recurrent training
        # input['B']:   e.g. [4, 10, 3, 256, 256]
        self.LR = input['A'].to(self.device, non_blocking=True)
        assert self.LR.shape[1] == self.opt.nframes, "input image length {} should equal to opt.nframes {}".format(self.LR.shape[1], self.opt.nframes)
        mid = self.opt.nframes // 2
        self.HR_GroundTruth = input['B'][:, mid, ...].to(self.device, non_blocking=True)
        # print(self.LR.shape)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.HR_G = self.netG(self.LR)

    def compute_visuals(self):
        mid = self.opt.nframes//2
        self.LR = self.LR[:, mid, ...]
        self.HR_Bicubic = torch.nn.functional.interpolate(self.LR, scale_factor=self.SR_factor, mode='bicubic', align_corners=False)

    def backward(self):
        """Calculate loss"""
        self.loss_SR = self.criterionL1(self.HR_G, self.HR_GroundTruth)
        self.loss_SR.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()
