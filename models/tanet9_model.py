"""
python train.py --dataroot ./datasets/Vid4 --name Vid4_tanet9 --model tanet9 --display_freq  40  --print_freq  4  --num_threads 2

aimax:
    gpu:
    python3 train.py
        --dataroot          /opt/data/private/datasets/vimeo_septuplet
        --name              vimeo_tanet9
        --model             tanet9
        --display_freq      4800
        --print_freq        4800
        --save_epoch_freq   5
        --gpu_ids           0,1,2,3,4
        --batch_size        30
        --suffix            05_04_17_10
        --crop_size         64
        --imgseqlen         5
        --nframes           5
        --seed              1
"""
import torch
from .base_model import BaseModel
from . import tanet9_networks
from util import remove_pad_for_tensor


class TANET9Model(BaseModel):
    """ This class implements the tanet9 model

    The model training requires '--dataset_mode aligned_video' dataset.

    Here we do not use optical flow.

    vimeo90K train dataset size: 55025.
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
        parser.set_defaults(batch_size=2)  # 8 in paper  need 4 gpu
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(SR_factor=4)
        parser.set_defaults(crop_size=64)  # 64
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0003)
        parser.set_defaults(init_type='kaiming')
        parser.set_defaults(lr_policy='step')
        parser.set_defaults(lr_decay_iters=20)
        parser.set_defaults(lr_gamma=0.65)
        parser.set_defaults(n_epochs=150)
        parser.set_defaults(multi_base=8)
        parser.add_argument('--cl', type=int, default=32, help='the cl in paper')
        parser.add_argument('--cm', type=int, default=32, help='the cm in paper')
        parser.add_argument('--ch', type=int, default=16, help='the ch in paper')
        parser.add_argument('--nframes', type=int, default=5, help='frames used by model')  # used for assert, imgseqlen should set equal to this when train

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

        self.netG = tanet9_networks.define_G(opt)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

        # input['A']:   e.g. [4, 10, 3, 64, 64] for recurrent training
        # input['B']:   e.g. [4, 10, 3, 256, 256]
        self.LR = input['A'].to(self.device, non_blocking=True)
        assert self.LR.shape[1] == self.opt.nframes, "input image length {} should equal to opt.nframes {}".format(self.LR.shape[1], self.opt.nframes)
        mid = self.opt.nframes // 2
        self.HR_GroundTruth = input['B'][:, mid, ...].contiguous().to(self.device, non_blocking=True)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.HR_G = self.netG(self.LR)

    def compute_visuals(self):
        mid = self.opt.nframes//2
        self.LR = self.LR[:, mid, ...]
        if self.opt.phase == "test":
            # remove pad for LR
            self.LR = remove_pad_for_tensor(tensor=self.LR,
                                            HR_GT_h_w=(self.HR_GroundTruth.shape[-2], self.HR_GroundTruth.shape[-1]),
                                            factor=self.SR_factor, LR_flag=True)
            # remove pad for HR_G
            self.HR_G = remove_pad_for_tensor(tensor=self.HR_G,
                                              HR_GT_h_w=(self.HR_GroundTruth.shape[-2], self.HR_GroundTruth.shape[-1]),
                                              factor=self.SR_factor, LR_flag=False)

        self.HR_Bicubic = torch.nn.functional.interpolate(self.LR, scale_factor=self.SR_factor, mode='bicubic', align_corners=False)

    def backward(self):
        """Calculate loss"""
        mid = self.opt.nframes//2
        self.loss_SR = self.criterionL1(self.HR_G, self.HR_GroundTruth)
        self.loss_SR.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()
