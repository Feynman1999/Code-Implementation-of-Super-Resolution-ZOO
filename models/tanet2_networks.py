import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm2d
from torch.nn.modules.utils import _pair
from . import base_networks
from .tanet_networks import PCD_Align, EDPU, EUPU


"""
change all resblock to resneSt!
"""


class SplAtConv2d(nn.Module):
    """
        Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0), groups=2, bias=True,
                 radix=2, reduction_factor=4, norm_layer=None):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = nn.Conv2d(in_channels, channels*radix, kernel_size, stride, padding, groups=groups*radix, bias=bias)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, kernel_size=1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)

    def forward(self, x):
        x = self.conv(x)
        # if self.use_bn:
        #     x = self.bn0(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel//self.radix, dim=1)  # torch.chunk(self.radix)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, output_size=1)  # [B, channels, 1, 1]
        gap = self.fc1(gap)  # [B, inter_channels, 1, 1]

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))  # [B, channels*radix, 1, 1]
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


class ResneStBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super(ResneStBlock, self).__init__()

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.act = nn.PReLU()
        else:
            raise NotImplementedError("not implemented activation")

        m = []
        m.append(SplAtConv2d(in_channels=in_channels, channels=out_channels, kernel_size=kernel_size, padding=(kernel_size//2, kernel_size//2), norm_layer=BatchNorm2d))
        m.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.act(res)


class ResBlocks(nn.Module):
    def __init__(self, channel_num, resblock_num, kernel_size=3, activation='prelu'):
        super(ResBlocks, self).__init__()
        self.model = nn.Sequential(
            self.make_layer(ResneStBlock, channel_num, resblock_num, kernel_size, activation),
        )

    def make_layer(self, block, ch_out, num_blocks, kernel_size, activation):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_out, ch_out, kernel_size, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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


class TANET2Generator(nn.Module):
    def __init__(self, args):
        super(TANET2Generator, self).__init__()
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
            ResBlocks(channel_num=cm, resblock_num=5, kernel_size=3),
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
        del nbr_fea_l
        del L1_fea
        del L2_fea
        del L3_fea
        return self.reconstruction(torch.cat(Hlist, dim=1))


def define_G(opt):
    net = TANET2Generator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
