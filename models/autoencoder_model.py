"""
python train.py --dataroot  C:\\Users\\昱翔\\Desktop\\train  --name  cartoonfaces_autoencoder --model autoencoder --display_freq 12800  --print_freq 12800 --save_epoch_freq 10  --gpu_ids 0  --batch_size 128 --suffix   06_25_17_09


aimax:
    gpu:

    v1:
    python3 train.py
        --dataroot          /opt/data/private/datasets/cartoonfaces
        --name              cartoonfaces_autoencoder
        --model             autoencoder
        --display_freq      25600
        --print_freq        25600
        --save_epoch_freq   10
        --gpu_ids           0
        --batch_size        256
        --suffix            06_25_17_09
        --crop_size         96
        --seed              1
        --max_dataset_size  50000
"""

import torch
from .base_model import BaseModel
from . import autoencoder_networks


class AUTOENCODERModel(BaseModel):
    """ This class implements the AUTOENCODER model

    The model training requires '--dataset_mode aligned' dataset.

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
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(batch_size=16)
        parser.set_defaults(SR_factor=1)
        parser.set_defaults(normalize_means='0.5,0.5,0.5')
        parser.set_defaults(crop_size=96)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0002)
        parser.set_defaults(init_type='kaiming')
        parser.set_defaults(lr_policy='step')
        parser.set_defaults(lr_decay_iters=30)
        parser.set_defaults(lr_gamma=0.75)
        parser.set_defaults(n_epochs=200)
        parser.add_argument('--block_size', type=str, default='1_1', help='for save memory, we make blocks')
        return parser

    def __init__(self, opt):
        """Initialize the AUTOENCODERModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.loss_names = ['L2']  # please use loss_G_L1 below

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.opt.phase == "apply":
            self.visual_names = ['origin', 'restore']
        else:
            self.visual_names = ['origin', 'restore']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['ae']
        else:  # during test and apply, only load G
            self.model_names = ['ae']

        self.netae = autoencoder_networks.define_ae_net(opt)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_ae = torch.optim.Adam(self.netae.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_ae)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.origin = input['A'].to(self.device, non_blocking=True)
        self.A_paths = input['A_paths']  # list       len = batchsize

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.restore = self.netae(self.origin)  # G(A)

    def compute_visuals(self):
        pass

    def backward(self):
        """Calculate loss for the G"""
        self.loss_L2 = self.criterionL2(self.restore, self.origin)
        self.loss_L2.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_ae.zero_grad()
        self.backward()
        self.optimizer_ae.step()