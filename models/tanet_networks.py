from . import base_networks
import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.modulated_deform_conv import ModulatedDeformConv, ModulatedDeformConvFunction
import logging

# change for rbpn
"""
* enhanced UPU and DPU
* id == mid 不continue
* add more init feature extrater
* add pcd align

do not use 

* add non local       OOM         
"""

logger = logging.getLogger('base')


class DCN(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset_mask.lr_mult = lr_mult
        self.conv_offset_mask.do_not_init_again = True
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, offset=None, mask=None):
        if offset is not None:
            out = self.conv_offset_mask(offset)
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))

        return ModulatedDeformConvFunction.apply(x, offset, mask, self.weight, self.bias,
                                                 self.stride, self.padding, self.dilation,
                                                 self.groups, self.deformable_groups, self.im2col_step)


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=128, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea


class CARBBlock(nn.Module):
    def __init__(self, channel_num):
        super(CARBBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))  # B,C,H,W -> B,C,1,1
        self.linear = nn.Sequential(
            nn.Linear(channel_num, channel_num // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num // 2, channel_num),
            nn.Sigmoid()
        )
        self.conv2 = nn.Conv2d(channel_num*2, channel_num, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)  # [B, C, H, W]
        w = self.global_average_pooling(x1)
        w = torch.squeeze(w)
        w = self.linear(w)
        w = torch.unsqueeze(w, dim=-1)
        w = torch.unsqueeze(w, dim=-1)
        x1 = torch.cat((x1, torch.mul(x1, w)), dim=1)  # [B, 2C, H, W]
        x1 = self.conv2(x1)  # [B, C, H, W]
        return x + x1


class Separate_non_local(nn.Module):
    def __init__(self, channel_num):
        super(Separate_non_local, self).__init__()
        self.A1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1)
        self.B1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1)
        self.D1 = nn.Conv2d(channel_num, channel_num, kernel_size=1, padding=0, stride=1)
        self.conv1_1 = nn.Conv2d(2*channel_num, channel_num, kernel_size=1, padding=0, stride=1)

    def forward(self, ref, nei):
        """
        :param ref: [B, C, H, W]
        :param nei: [B, C, H, W]
        :return: [B, C, H, W]
        """
        B, C, H, W = ref.shape
        x = torch.cat((ref, nei), dim=2)  # [B, C, 2H, W]

        A1 = self.A1(x).view(B, C, 2*H*W).permute(0, 2, 1)  # [B, 2*H*W, C]
        B1 = self.B1(x).view(B, C, 2*H*W)
        D1 = self.D1(x).view(B, C, 2*H*W).permute(0, 2, 1)
        attention1 = F.softmax(torch.matmul(A1, B1), dim=-1)
        E1 = torch.matmul(attention1, D1).permute(0, 2, 1).contiguous().view(B, C, 2*H, W)
        o1, o2 = torch.chunk(E1, 2, dim=2)
        E1 = self.conv1_1(torch.cat((o1, o2), dim=1))
        return ref + E1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='prelu'):
        super(ResBlock, self).__init__()

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.act = nn.PReLU()
        else:
            raise NotImplementedError("not implemented activation")

        m = []
        m.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), stride=1))
        m.append(self.act)
        m.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), stride=1))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.act(res)


class ResBlocks(nn.Module):
    def __init__(self, channel_num, resblock_num, kernel_size=3, activation='prelu'):
        super(ResBlocks, self).__init__()
        self.model = nn.Sequential(
            self.make_layer(ResBlock, channel_num, resblock_num, kernel_size, activation),
        )

    def make_layer(self, block, ch_out, num_blocks, kernel_size, activation):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_out, ch_out, kernel_size, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EUPU(nn.Module):
    """
        Enhanced up-projection units
    """
    def __init__(self, channels, kernel_size, stride, padding, use_pixel_shuffle=False):
        super(EUPU, self).__init__()
        if use_pixel_shuffle:
            assert stride == 4, "now support 4 only"
            self.deconv1 = nn.Sequential(
                nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
            )
            self.deconv2 = nn.Sequential(
                nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            self.deconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)
            self.deconv2 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)

        self.PRelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PRelu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.PRelu3 = nn.PReLU(num_parameters=1, init=0.25)
        self.PRelu4 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv1_1_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv1_1_2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        x2 = self.deconv1(x1)
        x2 = self.PRelu1(x2)

        x3 = self.conv1(x2)
        x3 = x3 - self.PRelu2(self.conv1_1_1(x1))

        x4 = self.deconv2(x3)
        x4 = self.PRelu3(x4)

        x4 = x4 + self.PRelu4(self.conv1_1_2(x2))

        return x4


class EDPU(nn.Module):
    """
        Enhanced down-projection units
    """
    def __init__(self, channels, kernel_size, stride, padding, use_pixel_shuffle=False):
        super(EDPU, self).__init__()
        if use_pixel_shuffle:
            assert stride == 4, "now support 4 only"
            self.deconv1 = nn.Sequential(
                nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            self.deconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PRelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.PRelu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PRelu3 = nn.PReLU(num_parameters=1, init=0.25)
        self.PRelu4 = nn.PReLU(num_parameters=1, init=0.25)

        self.conv1_1_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv1_1_2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.PRelu1(x2)

        x3 = self.deconv1(x2)

        x3 = x3 - self.PRelu2(self.conv1_1_1(x1))

        x4 = self.conv2(x3)
        x4 = self.PRelu3(x4)

        x4 = x4 + self.PRelu4(self.conv1_1_2(x2))

        return x4


class SISR_Block(nn.Module):
    def __init__(self, cl, ch):
        super(SISR_Block, self).__init__()
        self.num_stages = 3
        self.pre_deal = nn.Conv2d(cl, ch, kernel_size=1, stride=1, padding=0)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.UPU1 = EUPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.UPU2 = EUPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.UPU3 = EUPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.DPU1 = EDPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.DPU2 = EDPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.reconstruction = nn.Conv2d(self.num_stages * ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.prelu(self.pre_deal(x))

        h1 = self.UPU1(x)
        h2 = self.UPU2(self.DPU1(h1))
        h3 = self.UPU3(self.DPU2(h2))

        x = self.reconstruction(torch.cat((h3, h2, h1), 1))
        return x


class Residual_Blocks(nn.Module):
    def __init__(self, ch):
        super(Residual_Blocks, self).__init__()
        self.model = nn.Sequential(
            ResBlocks(channel_num=ch, resblock_num=5, kernel_size=3),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class MISR_Block(nn.Module):
    def __init__(self, cm, ch):
        super(MISR_Block, self).__init__()
        self.model = nn.Sequential(
            ResBlocks(channel_num=cm, resblock_num=5, kernel_size=3),
            nn.ConvTranspose2d(cm, ch, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, ch, cl):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            ResBlocks(channel_num=ch, resblock_num=5, kernel_size=3),
            nn.Conv2d(ch, cl, kernel_size=8, stride=4, padding=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Projection_Module(nn.Module):
    def __init__(self, args):
        super(Projection_Module, self).__init__()
        self.misr = MISR_Block(args.cm, args.ch)
        self.sisr = SISR_Block(args.cl, args.ch)
        self.res = Residual_Blocks(args.ch)
        self.decoder = Decoder(args.ch, args.cl)

    def forward(self, M, L):
        """

        :param M: M_{t-2}
        :param L: L_{t-1}
        :return: H_{t-2} L_{t-2}
        """
        hm = self.misr(M)
        hl = self.sisr(L)
        hm = hl - hm  # error
        hm = self.res(hm)
        hl = hl + hm
        next_l = self.decoder(hl)
        return hl, next_l


class TANETGenerator(nn.Module):
    def __init__(self, args):
        super(TANETGenerator, self).__init__()
        cl = args.cl
        cm = args.cm
        ch = args.ch

        self.nframes = args.nframes

        # Initial Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.input_nc, cl, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.feature_encoder_carb = nn.Sequential(
            nn.Conv2d(args.input_nc, cm, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.make_carb_layer(block=CARBBlock, ch_num=cm, num_blocks=5),
        )
        self.fea_L2_conv1 = nn.Conv2d(cm, cm, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(cm, cm, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(cm, cm, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(cm, cm, 3, 1, 1, bias=True)
        self.PCD_conv = PCD_Align(nf=cm, groups=cm // 64 * 8)

        # projection module
        self.Projection = Projection_Module(args)

        # reconstruction module
        self.reconstruction = nn.Conv2d((self.nframes ) * ch, args.output_nc, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def make_carb_layer(self, block, ch_num, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x: e.g. [4, 7, 3, 64, 64]
        :return:
        """
        B, N, C, H, W = x.shape  # N video frames
        mid = self.nframes // 2
        L = self.conv1(x[:, mid, ...])

        # first do feature encoder
        L1_fea = self.feature_encoder_carb(x.view(-1, C, H, W))
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        ref_fea_l = [
            L1_fea[:, mid, ...].clone(), L2_fea[:, mid, ...].clone(),
            L3_fea[:, mid, ...].clone()
        ]
        Hlist = []
        for id in range(self.nframes):
            nbr_fea_l = [
                L1_fea[:, id, ...].clone(), L2_fea[:, id, ...].clone(),
                L3_fea[:, id, ...].clone()
            ]
            M = self.PCD_conv(nbr_fea_l, ref_fea_l)
            # M = self.conv2(torch.cat((x[:, mid, ...], x[:, id, ...]), dim=1))
            # M = self.sep(L1_fea[:, mid, ...], L1_fea[:, id, ...])
            H, L = self.Projection(M, L)
            Hlist.append(H)
        del ref_fea_l
        return self.reconstruction(torch.cat(Hlist, dim=1))


def define_G(opt):
    net = TANETGenerator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
