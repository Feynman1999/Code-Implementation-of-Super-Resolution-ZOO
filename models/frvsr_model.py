"""
python train.py --dataroot ./datasets/youku --name youku_frvsr --model frvsr

BitaHub:
python /code/Code-Implementation-of-Super-Resolution-ZOO-master/train.py --dataroot /data/feynman1999/Vid4/vid4 --name vid4_frvsr --model frvsr  --display_id 0 --checkpoints_dir /output/checkpoints
"""
import torch
from .base_model import BaseModel
from . import frvsr_networks


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
        parser.set_defaults(normalize_means='0,0,0')
        parser.set_defaults(crop_size=64)
        parser.set_defaults(beta1='0.9')
        parser.set_defaults(lr=0.0001)
        parser.set_defaults(init_type='xavier')
        parser.add_argument('--resblock_num', type=int, default=10, help='the number of ResidualBlock, 10 in paper')
        parser.add_argument('--maxvel', type=float, default=24.0, help='scales the flow network output to the normal velocity range')
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
        self.visual_names = ['LR', 'HR_GroundTruth', 'HR_G', 'low_res']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['F', 'SR']
        else:
            self.model_names = ['F', 'SR']

        self.netSR = frvsr_networks.define_SR(opt)
        self.netF = frvsr_networks.define_F(opt)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_SR = torch.optim.Adam(self.netSR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_SR)
            self.optimizers.append(self.optimizer_F)


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
        self.LR = input['A'].to(self.device)
        self.HR_GroundTruth = input['B'].to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
           use recurrent training, thus recurrently run model and get a sequence of output and do bp.
        """
        length = self.LR.shape[1]
        batchsize = self.LR.shape[0]
        size = self.LR.shape[-1]
        flows = []  # len will be 9
        warped_low_res = []

        for i in range(length-1):
            flows.append(self.netF(torch.cat((self.LR[:, i, ...], self.LR[:, i+1, ...]), 1)))
            warped_low_res.append(self.dense_image_warp(self.LR[:, i, ...], flows[-1]))

        # flows x4
        for i in range(len(flows)):
            flows[i] = torch.nn.functional.interpolate(flows[i] * self.SR_factor, scale_factor=self.SR_factor, mode='bilinear')

        HR_G_list = []
        black = torch.zeros((batchsize, 3*self.SR_factor*self.SR_factor, size, size), dtype=torch.float32).to(self.device)
        for i in range(length):
            if i == 0:  # first frame
                HR_G_list.append(self.netSR(torch.cat((self.LR[:, i, ...], black), 1)))
            else:
                warped_pre_G_frame = self.dense_image_warp(HR_G_list[-1], flows[i-1])
                warped_pre_G_frame = self.space_to_depth(warped_pre_G_frame, self.SR_factor)
                HR_G_list.append(self.netSR(torch.cat((self.LR[:, i, ...], warped_pre_G_frame), 1)))

        self.HR_G = torch.stack(HR_G_list, 1)
        self.low_res = torch.stack(warped_low_res, 1)

    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

    def dense_image_warp(self, image, flow):
        """
        :param image: 4-D float `Tensor` with shape `[batch, channels, height, width]`.
        :param flow: A 4-D float `Tensor` with shape `[batch, 2, height, width]`.
        :return: A 4-D float `Tensor` with shape`[batch, channels, height, width]`
            and same type as input image.
        """
        batch_size, channels, height, width = image.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
        stacked_grid = torch.stack([grid_x, grid_y], dim=0).type_as(flow).to(self.device)  # 2*h*w
        batched_grid = torch.unsqueeze(stacked_grid, dim=0)  # 1*2*h*w
        query_points_on_grid = batched_grid - flow  # batch * 2 * h * w
        normal = torch.tensor([(height-1)/2, (width-1)/2])
        normal = torch.reshape(normal, [1, 2, 1, 1]).to(self.device)
        query_points_on_grid = query_points_on_grid - normal
        query_points_on_grid = query_points_on_grid / normal  # try to [-1,1]
        query_points_on_grid = query_points_on_grid.permute(0, 3, 2, 1)
        interpolated = torch.nn.functional.grid_sample(image, query_points_on_grid, mode='bilinear', padding_mode="border")
        return interpolated


    def backward(self):
        """Calculate loss"""
        self.loss_SR = self.criterionL2(self.HR_G, self.HR_GroundTruth) * self.opt.lambda_L2
        self.loss_flow = self.criterionL2(self.low_res, self.LR[:, 1:, ...])
        self.loss = self.loss_SR + self.loss_flow
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update
        self.optimizer_SR.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward()
        self.optimizer_SR.step()
        self.optimizer_F.step()
