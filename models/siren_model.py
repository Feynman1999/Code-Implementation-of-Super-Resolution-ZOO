"""
python train.py --dataroot  C:\\Users\\76397\\Desktop\\1593850115897.jpg  --name   1593850115897_siren --model   siren  --display_freq  320000  --print_freq 32000  --save_epoch_freq   1000 --batch_size 3200

aimax:
    gpu:
    v1:
    python3 train.py
        --dataroot          /opt/data/private/datasets/siren/trump.jpg
        --name              trump_siren
        --model             siren
        --display_freq      819200
        --print_freq        81920
        --save_epoch_freq   1000
        --gpu_ids           0
        --batch_size        8192
        --suffix            07_05_23_34
        --crop_size         512
        --Reduction_factor  1
        --continue_train    True
        --load_epoch        epoch_5000
        --epoch_count       5001
        --n_epochs          5010
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
        parser.set_defaults(batch_size=8192)
        parser.set_defaults(SR_factor=4)
        parser.set_defaults(normalize_means='0.5,0.5,0.5')
        parser.set_defaults(crop_size=512)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0005)
        parser.set_defaults(init_type='kaiming')
        parser.set_defaults(lr_policy='step')
        parser.set_defaults(lr_decay_iters=1000)
        parser.set_defaults(lr_gamma=0.7)
        parser.set_defaults(n_epochs=5000)
        parser.set_defaults(num_threads=7)
        parser.set_defaults(normalize_means='0,0,0')
        parser.set_defaults(normalize_stds='0.00392156862745098,0.00392156862745098,0.00392156862745098')
        parser.add_argument('--Reduction_factor', type=int, default=10)
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
            self.visual_names = ['origin', 'restore', 'GT', 'SR']
        else:
            self.visual_names = ['origin', 'restore', 'GT', 'SR']

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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.phase == "train":
            self.P = self.netsiren(self.A)  # G(A)
        elif self.opt.phase == "apply":
            pass

    def compute_visuals(self, dataset=None):
        self.origin = dataset.dataset.background_img.unsqueeze(0)
        self.GT = dataset.dataset.img.unsqueeze(0)
        loc_list = []
        h, w = self.GT.shape[-2], self.GT.shape[-1]
        for i in range(h):
            for j in range(w):
                loc_list.append([i/h, j/w])

        with torch.no_grad():
            rst = self.netsiren(torch.tensor(loc_list, dtype=torch.float32))  # [B,2]

        rst = rst.view(h, w, -1)
        self.restore = rst.permute(2, 0, 1).unsqueeze(0)

        if self.opt.phase in ("apply", "train"):
            s = self.opt.SR_factor
            self.SR = torch.zeros((3, h*s, w*s), dtype=torch.float32)
            h_loc = torch.linspace(0-0.5, h-1+0.5, s*h) / h
            w_loc = torch.linspace(0-0.5, w-1+0.5, s*w) / w
            hh, ww = torch.meshgrid(h_loc, w_loc)
            h_w_loc = torch.stack([hh, ww], dim=2)  # [s*h, s*w, 2]
            with torch.no_grad():
                for i in range(s*h):
                    self.SR[:, i, :] = self.netsiren(h_w_loc[i, :, :]).permute(1, 0)
                self.SR = self.SR.unsqueeze(0)

    def backward(self):
        """Calculate loss for the G"""
        self.loss_L2 = self.criterionL2(self.P, self.B)
        self.loss_L2.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_siren.zero_grad()
        self.backward()
        self.optimizer_siren.step()
