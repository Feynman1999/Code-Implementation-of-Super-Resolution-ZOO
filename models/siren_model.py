"""
aimax:
    gpu:
    v1:
    python3 train.py
        --dataroot          /opt/data/private/datasets/1593850115897.jpg
        --name              1593850115897_siren
        --model             siren
        --display_freq      320000
        --print_freq        32000
        --save_epoch_freq   1000
        --gpu_ids           0
        --batch_size        3200
        --suffix            07_05_13_14
        --crop_size         512
"""

import torch
from .base_model import BaseModel
from . import siren_networks


class SIRENModel(BaseModel):
    """ This class implements the siren model

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
        parser.set_defaults(dataset_mode='fitimage')
        parser.set_defaults(batch_size=3200)
        parser.set_defaults(SR_factor=1)
        parser.set_defaults(normalize_means='0.5,0.5,0.5')
        parser.set_defaults(crop_size=512)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0005)
        parser.set_defaults(init_type='kaiming')
        parser.set_defaults(lr_policy='step')
        parser.set_defaults(lr_decay_iters=1000)
        parser.set_defaults(lr_gamma=0.75)
        parser.set_defaults(n_epochs=5000)
        parser.set_defaults(num_threads=3)
        parser.add_argument('--Reduction_factor', type=int, default=16)
        return parser

    def __init__(self, opt):
        """Initialize the SIRENModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L2']  # please use loss_G_L1 below

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.opt.phase == "apply":
            self.visual_names = ['origin', 'restore', 'GT']
        else:
            self.visual_names = ['origin', 'restore', 'GT']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['siren']
        else:  # during test and apply, only load G
            self.model_names = ['siren']

        self.netsiren = siren_networks.define_siren_net(opt)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_siren = torch.optim.Adam(self.netsiren.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_siren)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.A = input['A'].to(self.device, non_blocking=True)  # [B,2]
        self.B = input['B'].to(self.device, non_blocking=True)  # [B,3]
        self.origin = input['C']
        self.GT = input['D']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.P = self.netsiren(self.A)  # G(A)

    def compute_visuals(self):
        loc_list = []
        h, w = self.GT.shape[1], self.GT.shape[2]
        for i in range(h):
            for j in range(w):
                loc_list.append([i, j])

        with torch.no_grad():
            rst = self.netsiren(torch.tensor(loc_list))  # [B,2]

        rst = rst.view(h, w, -1)
        self.restore = rst.permute(2, 0, 1)

    def backward(self):
        """Calculate loss for the G"""
        self.loss_L2 = self.criterionL2(self.P, self.B)
        self.loss_L2.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_siren.zero_grad()
        self.backward()
        self.optimizer_siren.step()