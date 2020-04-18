import torch
import torch.nn as nn
from torch.nn import functional as F
from . import base_networks
from modules.modulated_deform_conv import ModulatedDeformConv, ModulatedDeformConvFunction
import logging


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
            nn.PReLU(),
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
    def __init__(self, channel_num, frames):
        super(Separate_non_local, self).__init__()
        self.A1 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.B1 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.D1 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.A2 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.B2 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.D2 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.A3 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.B3 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.D3 = nn.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        """
        :param x: [B, N, C', H, W]
        :return: [B, N, C', H, W]
        """
        B, N, C, H, W = x.shape
        x = x.view(B, N*C, H, W)
        A1 = self.A1(x).view(B, N*C, H*W).permute(0, 2, 1)  # [B, H*W, N*C]
        B1 = self.B1(x).view(B, N*C, H*W)
        A2 = self.A2(x).view(B, N, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, N*H*W)
        B2 = self.B2(x).view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, N*H*W, C)
        A3 = self.A3(x).view(B, N, C, H, W).view(B, N, C*H*W)
        B3 = self.B3(x).view(B, N, C, H, W).view(B, N, C*H*W).permute(0, 2, 1)
        D1 = self.D1(x).view(B, N*C, H*W).permute(0, 2, 1)
        D2 = self.D2(x).view(B, N, C, H, W).permute(0, 2, 1, 3, 4).contiguous().view(B, C, N*H*W)
        D3 = self.D3(x).view(B, N, C, H, W).view(B, N, C*H*W)
        attention1 = F.softmax(torch.matmul(A1, B1), dim=-1)
        attention2 = F.softmax(torch.matmul(A2, B2), dim=-1)
        attention3 = F.softmax(torch.matmul(A3, B3), dim=-1)
        E1 = torch.matmul(attention1, D1).permute(0, 2, 1).contiguous().view(B, N, C, H, W)
        E2 = torch.matmul(attention2, D2).view(B, C, N, H, W).permute(0, 2, 1, 3, 4)
        E3 = torch.matmul(attention3, D3).view(B, N, C, H, W)
        return x.view(B, N, C, H, W) + E1 + E2 + E3


class VESRGenerator(nn.Module):
    def __init__(self, args):
        super(VESRGenerator, self).__init__()
        self.mid = args.nframes // 2
        self.channel_size = args.channel_size
        self.conv1 = nn.Conv2d(args.input_nc, self.channel_size, kernel_size=3, stride=1, padding=1)
        self.feature_encoder_carb = nn.Sequential(
            self.make_carb_layer(block=CARBBlock, ch_num=self.channel_size, num_blocks=args.CARB_num1),
        )
        self.fea_L2_conv1 = nn.Conv2d(self.channel_size, self.channel_size, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(self.channel_size, self.channel_size, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(self.channel_size, self.channel_size, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(self.channel_size, self.channel_size, 3, 1, 1, bias=True)
        self.PCD_conv7 = PCD_Align(nf=self.channel_size, groups=self.channel_size//64 * 8)
        self.Seperate_NL8 = Separate_non_local(channel_num=self.channel_size, frames=args.nframes)
        self.conv9 = nn.Conv2d(self.channel_size * args.nframes, self.channel_size, kernel_size=3, stride=1, padding=1)

        self.reconstruct_carb = nn.Sequential(
            self.make_carb_layer(block=CARBBlock, ch_num=self.channel_size, num_blocks=args.CARB_num2),
        )

        self.conv31 = nn.Conv2d(self.channel_size, self.channel_size*4, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(self.channel_size, self.channel_size*2, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)

        self.conv35 = nn.Conv2d(self.channel_size//2, self.channel_size//2, kernel_size=3, stride=1, padding=1)
        self.conv36 = nn.Conv2d(self.channel_size//2, args.output_nc, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def init_DCN_offsets(self):
        self.PCD_conv7.L3_dcnpack.init_offset()
        self.PCD_conv7.L2_dcnpack.init_offset()
        self.PCD_conv7.L1_dcnpack.init_offset()
        self.PCD_conv7.cas_dcnpack.init_offset()

    def make_carb_layer(self, block, ch_num, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:  e.g. [32, 7, 3, 64, 64]
        """
        B, N, C, H, W = x.shape  # N video frames
        # feature encoder
        L1_fea = self.feature_encoder_carb(self.lrelu(self.conv1(x.view(-1, C, H, W))))  # [B*N, 128, 64, 64]
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
            L1_fea[:, self.mid, ...].clone(), L2_fea[:, self.mid, ...].clone(),
            L3_fea[:, self.mid, ...].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.PCD_conv7(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C', H, W]
        del ref_fea_l
        del nbr_fea_l
        del L1_fea
        del L2_fea
        del L3_fea
        aligned_fea = self.Seperate_NL8(aligned_fea)
        aligned_fea = self.lrelu(self.conv9(aligned_fea.view(B, -1, H, W)))  # [B, C', H, W]
        aligned_fea = self.reconstruct_carb(aligned_fea)
        aligned_fea = self.lrelu(self.pixelshuffle(self.conv31(aligned_fea)))  # [B, C', 2*H, 2*W]
        aligned_fea = self.lrelu(self.pixelshuffle(self.conv33(aligned_fea)))  # [B, C'/2, 4*H, 4*W]
        aligned_fea = self.lrelu(self.conv35(aligned_fea))  # [B, C'/2, 4*H, 4*W]
        aligned_fea = self.conv36(aligned_fea)

        x_center = x[:, self.mid, ...].contiguous()
        x_center = F.interpolate(x_center, scale_factor=4, mode='bicubic', align_corners=False)

        return x_center + aligned_fea

def define_G(opt):
    net = VESRGenerator(opt)
    net = base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
    return net
