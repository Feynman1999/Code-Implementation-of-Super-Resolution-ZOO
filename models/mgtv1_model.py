"""
apply:
    v1: python apply.py --dataroot  ./datasets/mgtv/apply/A --name mgtv_mgtv1_05_24_22_39 --model mgtv1 --load_epoch epoch_1000
    v2: python apply.py --dataroot  ./datasets/mgtv/apply/A --name mgtv_mgtv1_48_32_100_05_27_00_25 --model mgtv1 --load_epoch epoch_1000

    nohup python3 -u apply.py --dataroot  /opt/data/private/datasets/mgtv/apply/A --name mgtv_mgtv1_05_24_22_39 --model mgtv1 --load_epoch epoch_1500 >> /opt/data/private/mgtv_epoch1500.log 2>&1 &

    dataset_images2video(datasetpath = "./results/mgtv_mgtv1_05_24_22_39/apply-A-epoch_1000-block_size_250", fps=25, suffix=".y4m")


aimax:
    gpu:

    v1:
    python3 train.py
        --dataroot          /opt/data/private/datasets/mgtv
        --name              mgtv_mgtv1
        --model             mgtv1
        --display_freq      480
        --print_freq        160
        --save_epoch_freq   500
        --gpu_ids           0
        --batch_size        16
        --suffix            05_24_22_39
        --crop_size         256
        --imgseqlen         5
        --seed              1
        --continue_train    True
        --load_epoch        epoch_1000
        --epoch_count       1001

    v2:
    python3 train.py
        --dataroot          /opt/data/private/datasets/mgtv
        --name              mgtv_mgtv1_48_32_100
        --model             mgtv1
        --display_freq      600
        --print_freq        120
        --save_epoch_freq   500
        --gpu_ids           0
        --batch_size        6
        --suffix            05_27_00_25
        --crop_size         256
        --imgseqlen         5
        --seed              1
        --max_consider_len  100
        --ch1               48
        --ch2               32
"""
import torch
from .base_model import BaseModel
from . import mgtv1_networks
from util import remove_pad_for_tensor


class MGTV1Model(BaseModel):
    """ This class implements the mgtv1 model

    The model training requires '--dataset_mode aligned_video' dataset.

    mgtv train dataset size: 600.
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
        parser.set_defaults(batch_size=16)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(SR_factor=1)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0004)
        parser.set_defaults(init_type='kaiming')
        parser.set_defaults(lr_policy='step')
        parser.set_defaults(lr_decay_iters=500)
        parser.set_defaults(lr_gamma=0.75)
        parser.set_defaults(n_epochs=5000)
        parser.set_defaults(multi_base=32)
        parser.set_defaults(max_consider_len=25)
        parser.add_argument('--ch1', type=int, default=12)
        parser.add_argument('--ch2', type=int, default=8)
        parser.add_argument('--nframes', type=int, default=5, help='frames used by model')  # used for assert, imgseqlen should set equal to this when train

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.SR_factor = opt.SR_factor

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['pd', 'st']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.opt.phase == "apply":
            self.visual_names = ['HR_G']
        else:
            self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']

        self.netG = mgtv1_networks.define_G(opt)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.A_paths = input['A_paths']
        # input['A']:   e.g. [4, 5, 3, 256, 256]
        # input['B']:   e.g. [4, 5, 3, 256, 256]
        self.LR = input['A'].to(self.device, non_blocking=True)
        assert self.LR.shape[1] == self.opt.nframes, "input image length {} should equal to opt.nframes {}".format(self.LR.shape[1], self.opt.nframes)

        if self.opt.phase in ('train', 'test'):
            self.B_paths = input['B_paths']
            self.HR_GroundTruth = input['B'].to(self.device, non_blocking=True)

        if self.opt.phase == "apply":
            self.HR_GT_h, self.HR_GT_w = input['gt_h_w']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.HR_Gs, self.HR_G = self.netG(self.LR)

    def compute_visuals(self):
        mid = self.opt.nframes//2
        self.LR = self.LR[:, mid, ...]

        if self.opt.phase in ("train", "test"):
            self.HR_GroundTruth = self.HR_GroundTruth[:, mid, ...]

        if self.opt.phase in ("test", "apply"):
            # remove pad for LR
            if self.opt.phase == "test":
                h, w = self.HR_GroundTruth.shape[-2], self.HR_GroundTruth.shape[-1]
            else:  # apply
                h, w = self.HR_GT_h, self.HR_GT_w
            self.LR = remove_pad_for_tensor(tensor=self.LR,
                                            HR_GT_h_w=(h, w),
                                            factor=self.SR_factor, LR_flag=True)
            # remove pad for HR_G
            self.HR_G = remove_pad_for_tensor(tensor=self.HR_G,
                                              HR_GT_h_w=(h, w),
                                              factor=self.SR_factor, LR_flag=False)


    def backward(self):
        """Calculate loss"""
        _, _, C, H, W = self.HR_Gs.shape
        self.loss_pd = self.criterionL2(self.HR_Gs.view(-1, C, H, W), self.HR_GroundTruth.view(-1, C, H, W))
        self.loss_st = self.criterionL2(self.HR_G, self.HR_GroundTruth[:, self.opt.nframes//2, ...])
        self.loss = self.loss_pd + self.nowepoch / 1000 * self.loss_st
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()
