from . import base_networks
import torch
import torch.nn as nn


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):  # x is input
        return x.view(x.size(0), -1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        m = []
        m.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), stride=1))
        m.append(nn.ReLU(inplace=True))
        m.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), stride=1))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class TeCoGan_G(nn.Module):
    def __init__(self, opt):
        super(TeCoGan_G, self).__init__()
        self.opt = opt
        self.channel_num = 64
        self.resblock_num = opt.resblock_num
        self.input_nc = opt.input_nc
        self.SR_factor = opt.SR_factor

        self.model = nn.Sequential(
            nn.Conv2d(self.input_nc + self.input_nc*opt.SR_factor*opt.SR_factor, self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.make_layer(ResBlock, self.channel_num, self.resblock_num),
            nn.ConvTranspose2d(self.channel_num, self.channel_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),  # x2
            nn.ConvTranspose2d(self.channel_num, self.channel_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),  # x4
            nn.Conv2d(self.channel_num, self.input_nc, kernel_size=3, stride=1, padding=1)
        )

    def make_layer(self, block, ch_out, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(ch_out, ch_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        hx = torch.nn.functional.interpolate(x[:, 0:3, ...], scale_factor=self.SR_factor, mode='bilinear')
        x = self.model(x)
        x = x + hx
        return x


class TecoGan_F(nn.Module):
    def __init__(self, opt):
        super(TecoGan_F, self).__init__()
        self.channel_num = 32
        self.maxvel = opt.maxvel
        self.model_1 = nn.Sequential(
            nn.Conv2d(2 * self.input_nc, self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.channel_num, 2 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * self.channel_num, 2 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(2 * self.channel_num, 4 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(4 * self.channel_num, 4 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(4 * self.channel_num, 8 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(8 * self.channel_num, 8 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.model_2 = nn.Sequential(
            nn.Conv2d(8 * self.channel_num, 4 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(4 * self.channel_num, 4 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.model_3 = nn.Sequential(
            nn.Conv2d(4 * self.channel_num, 2 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * self.channel_num, 2 * self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.model_4 = nn.Sequential(
            nn.Conv2d(2 * self.channel_num, self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.channel_num, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model_1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.model_2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.model_3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.model_4(x)
        x = x * self.maxvel
        return x


class TecoGan_D(nn.Module):
    def __init__(self, opt):
        super(TecoGan_D, self).__init__()
        self.input_nc = opt.input_nc
        self.channel_num = 64

        self.model = nn.Sequential(
            nn.Conv2d(9 * self.input_nc, self.channel_num, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_num),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.channel_num, self.channel_num, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_num),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(self.channel_num, 2 * self.channel_num, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * self.channel_num),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * self.channel_num, 4 * self.channel_num, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * self.channel_num),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # batchsize * 256 * 8 * 8
            Flatten(),
            nn.Linear(16384, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def define_G(opt):
    net = TeCoGan_G(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_F(opt):
    net = TecoGan_F(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_D(opt):
    net = TecoGan_D(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
