from . import base_networks
import torch
import torch.nn as nn

"""
rbpn + carb block
"""

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


class CARB_Blocks(nn.Module):
    def __init__(self, channel_num, block_num):
        super(CARB_Blocks, self).__init__()
        self.model = nn.Sequential(
            self.make_layer(CARBBlock, channel_num, block_num),
        )

    def make_layer(self, block, ch_num, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_num))
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
    def __init__(self, ch1, ch2):
        super(SISR_Block, self).__init__()
        self.num_stages = 3
        self.pre_deal = nn.Conv2d(ch1, ch2, kernel_size=1, stride=1, padding=0)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.UPU1 = EUPU(channels=ch2, kernel_size=8, stride=4, padding=2)
        self.UPU2 = EUPU(channels=ch2, kernel_size=8, stride=4, padding=2)
        self.UPU3 = EUPU(channels=ch2, kernel_size=8, stride=4, padding=2)
        self.DPU1 = EDPU(channels=ch2, kernel_size=8, stride=4, padding=2)
        self.DPU2 = EDPU(channels=ch2, kernel_size=8, stride=4, padding=2)
        self.reconstruction = nn.Conv2d(self.num_stages * ch2, ch2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.prelu(self.pre_deal(x))

        h1 = self.UPU1(x)
        h2 = self.UPU2(self.DPU1(h1))
        h3 = self.UPU3(self.DPU2(h2))

        x = self.reconstruction(torch.cat((h3, h2, h1), 1))
        return x


class Residual_Blocks(nn.Module):
    def __init__(self, ch2):
        super(Residual_Blocks, self).__init__()
        self.model = nn.Sequential(
            CARB_Blocks(channel_num=ch2, block_num=5),
            nn.Conv2d(ch2, ch2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class MISR_Block(nn.Module):
    def __init__(self, ch1, ch2):
        super(MISR_Block, self).__init__()
        self.model = nn.Sequential(
            CARB_Blocks(channel_num=ch1, block_num=5),
            nn.ConvTranspose2d(ch1, ch2, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, ch2, ch1):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            CARB_Blocks(channel_num=ch2, block_num=5),
            nn.Conv2d(ch2, ch1, kernel_size=8, stride=4, padding=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Projection_Module(nn.Module):
    def __init__(self, args):
        super(Projection_Module, self).__init__()
        self.misr = MISR_Block(args.ch1, args.ch2)
        self.sisr = SISR_Block(args.ch1, args.ch2)
        self.res = Residual_Blocks(args.ch2)
        self.decoder = Decoder(args.ch2, args.ch1)

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


class MGTV3Generator(nn.Module):
    def __init__(self, args):
        super(MGTV3Generator, self).__init__()
        self.nframes = args.nframes

        ch1 = args.ch1  # 256
        ch2 = args.ch2  # 64

        # Initial Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.input_nc, ch1, kernel_size=3, stride=4, padding=1),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.input_nc*2+0, ch1, kernel_size=3, stride=4, padding=1),
            nn.PReLU(),
        )

        # projection module
        self.Projection = Projection_Module(args)

        # reconstruction module
        self.reconstruction = nn.Conv2d((self.nframes ) * ch2, args.output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """

        :param x: e.g. [4, 5, 3, 256, 256]
        :return:
        """
        mid = self.nframes // 2
        L = self.conv1(x[:, mid, ...])

        Hlist = []
        for id in range(self.nframes):
            M = self.conv2(torch.cat((x[:, mid, ...], x[:, id, ...]), dim=1))
            H, L = self.Projection(M, L)
            Hlist.append(H)
        return self.reconstruction(torch.cat(Hlist, dim=1)) + x[:, mid, ...]


def define_G(opt):
    net = MGTV3Generator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
