from . import base_networks
import torch
import torch.nn as nn


class CONV(nn.Module):
    def __init__(self, input_ch, output_ch, activate='relu', use_bn=False, stride=1):
        super(CONV, self).__init__()
        if activate == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activate == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError("wrong")

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(output_ch)

        self.conv = nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.model = nn.Sequential(
            CONV(3, 16),  # [B, 16, 96, 96]
            nn.MaxPool2d(2, stride=2),  # [B, 16, 48, 48]
            CONV(16, 8),  # [B, 8, 48, 48]
            nn.MaxPool2d(2, stride=2),  # [B, 8, 24, 24]
            CONV(8, 8),  # [B, 8, 24, 24]
            nn.MaxPool2d(2, stride=2),  # [B, 8, 12, 12]
        )

    def forward(self, x):  # [B, 3, 96, 96]
        return self.model(x)


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(num_parameters=1, init=0.25),  # [B, 8, 24, 24]
            CONV(8, 8),  # [B, 8, 24, 24]
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(num_parameters=1, init=0.25),  # [B, 8, 48, 48]
            CONV(8, 8),  # [B, 8, 48, 48]
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(num_parameters=1, init=0.25),  # [B, 8, 96, 96]
            CONV(8, 16),
            CONV(16, 3),
        )

    def forward(self, x):  # [B, 8, 12, 12]
        return self.model(x)


class AEGenerator(nn.Module):
    def __init__(self):
        super(AEGenerator, self).__init__()
        self.e = encoder()
        self.d = decoder()

    def forward(self, x):
        mid_value = self.e(x)
        restore = self.d(mid_value)
        return restore


def define_ae_net(opt):
    net = AEGenerator()
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)