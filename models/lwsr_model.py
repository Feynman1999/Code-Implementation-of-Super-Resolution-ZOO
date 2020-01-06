'''
python train.py --dataroot ./datasets/DIV2k --name DIV_lwsr --model lwsr --display_ncols -1
'''
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks

# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         self.weight.data.div_(std.view(3, 1, 1, 1))
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
#         self.bias.data.div_(std)
#         self.requires_grad = False


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResBlock2(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1.0):
        super(ResBlock2, self).__init__()
        self.conv = conv(n_feat*2, n_feat, 1, bias=bias)

        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        ori = self.conv(x)
        res = self.body(ori).mul(self.res_scale)
        res += ori

        return res


class LwsrGenerator(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(LwsrGenerator, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        scale = args.SR_factor

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        #rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = networks.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.input_nc, n_feats, kernel_size)]

        # self.add_mean = networks.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        # define body module

        self.feat2_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat2_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat2_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.feat3_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat3_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat3_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.feat4_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat4_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat4_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.feat5_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat5_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat5_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.feat6_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat6_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.conv1 = conv(6 * n_feats, n_feats, 1, bias=False)

        self.feat7_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat7_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.conv7 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        #   self.conv7 = ConvBlock(base_filter2, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat8_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat8_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.conv8 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        #   self.conv8 = ConvBlock(base_filter2, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat9_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat9_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        #   self.conv9 = nn.conv2d(base_filter2, base_filter, kernel_size = 1, stride = 1, padding = 0)
        self.conv9 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)

        self.feat10_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat10_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        #   self.conv10 = nn.conv2d(base_filter2, base_filter, kernel_size = 1, stride = 1, padding = 0)
        self.conv10 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)

        #      self.conv11 = torch.nn.Conv2d(base_filter*2, base_filter, 1, 1, 0)

        # define tail module
        modules_tail = [
            networks.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.input_nc, kernel_size)]

        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x1 = self.head(x)

        feat2_1 = self.feat2_1(x1)
        feat2_2 = self.feat2_2(feat2_1)
        feat2_3 = self.feat2_3(feat2_2)

        feat3_1 = self.feat3_1(torch.add(feat2_1, feat2_3))
        feat3_2 = self.feat3_2(feat3_1)
        feat3_3 = self.feat3_3(feat3_2)

        feat4_1 = self.feat4_1(torch.add(feat3_1, feat3_3))
        feat4_2 = self.feat4_2(feat4_1)
        feat4_3 = self.feat4_3(feat4_2)

        feat5_1 = self.feat5_1(torch.add(feat4_1, feat4_3))
        feat5_2 = self.feat5_2(feat5_1)
        feat5_3 = self.feat5_3(feat5_2)

        feat6_1 = self.feat6_1(torch.add(feat5_1, feat5_3))
        feat6_2 = self.feat6_2(feat6_1)

        bool = torch.cat([x1, feat3_1, feat4_1, feat5_1, feat6_1, feat6_2], 1)
        conv1 = self.conv1(bool)

        concat_7 = torch.cat([feat6_2, 0.5 * feat5_3 + 0.5 * conv1], 1)
        feat7_1 = self.feat7_1(concat_7)
        feat7_2 = self.feat7_2(feat7_1)
        conv7 = self.conv7(feat7_2)

        concat_8 = torch.cat([conv7, 0.5 * feat4_3 + 0.5 * conv1], 1)
        feat8_1 = self.feat8_1(concat_8)
        feat8_2 = self.feat8_2(feat8_1)
        conv8 = self.conv8(feat8_2)

        concat_9 = torch.cat([conv8, 0.5 * feat3_3 + 0.5 * conv1], 1)
        feat9_1 = self.feat9_1(concat_9)
        feat9_2 = self.feat9_2(feat9_1)
        conv9 = self.conv9(feat9_2)

        concat_10 = torch.cat([conv9, 0.5 * feat2_3 + 0.5 * conv1], 1)
        feat10_1 = self.feat10_1(concat_10)
        feat10_2 = self.feat10_2(feat10_1)
        conv10 = self.conv10(feat10_2)

        conv10 += x1

        x = self.tail(conv10)
        # x = self.add_mean(x)

        return x


def define_lwsr_net(opt):
    net = LwsrGenerator(opt)
    return networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


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

        The training objective is: xxxxxxxxxx
        """
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(batch_size=8)
        parser.set_defaults(preprocess='crop')
        parser.set_defaults(SR_factor=4)
        parser.set_defaults(normalize_means='0.4488,0.4371,0.4040')
        parser.set_defaults(crop_size=48)
        if is_train:
            parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=10.0, help='weight for L2 loss')

        return parser

    def __init__(self, opt):
        """Initialize the LWSR class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['L1', 'L2']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['LWSR']
        else:  # during test time, only load G
            self.model_names = ['LWSR']

        self.netLWSR = define_lwsr_net(opt)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netLWSR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.LR = input['A'].to(self.device)
        self.HR_GroundTruth = input['B'].to(self.device)
        # print(self.LR.shape, self.HR_GroundTruth.shape)
        self.A_paths = input['A_paths']
        self.B_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.HR_G = self.netLWSR(self.LR)  # G(A)

    def backward_LWSR(self):
        """Calculate loss for the LWSR"""
        self.loss_L1 = self.criterionL1(self.HR_G, self.HR_GroundTruth) * self.opt.lambda_L1
        self.loss_L2 = self.criterionL2(self.HR_G, self.HR_GroundTruth) * self.opt.lambda_L2
        self.loss = self.loss_L1 + self.loss_L2
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        # self.set_requires_grad(self.netLWSR, True)
        self.optimizer.zero_grad()
        self.backward_LWSR()
        self.optimizer.step()
