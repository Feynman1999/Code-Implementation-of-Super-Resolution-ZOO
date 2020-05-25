from . import base_networks
import torch
import torch.nn as nn


class CONV(nn.Module):
    def __init__(self, input_ch, output_ch, activate='relu', use_bn = False, stride = 1):
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

class G1(nn.Module):
    def __init__(self, ch=48):
        super(G1, self).__init__()
        self.ch = ch

        self.conv0 = CONV(3, ch, 'relu', False, 1)
        self.conv1 = nn.Sequential(
            CONV(ch, ch, 'relu', False, 1),
            CONV(ch, ch, 'relu', False, 2)
        )
        self.conv2 = nn.Sequential(
            CONV(ch, ch, 'relu', False, 1),
            CONV(ch, ch, 'relu', False, 2)
        )
        self.conv3 = nn.Sequential(
            CONV(ch, ch, 'relu', False, 1),
            CONV(ch, ch, 'relu', False, 2)
        )
        self.conv4 = nn.Sequential(
            CONV(ch, ch, 'relu', False, 1),
            CONV(ch, ch, 'relu', False, 2)
        )
        self.conv5 = nn.Sequential(
            CONV(ch, ch, 'relu', False, 1),
            CONV(ch, ch, 'relu', False, 2)
        )
        self.deconv5 = nn.Sequential(
            CONV(ch, ch, 'relu', False, 1),
            nn.ConvTranspose2d(ch, ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            CONV(2 * ch, 2 * ch, 'relu', False, 1),
            CONV(2 * ch, 2 * ch, 'relu', False, 1),
            nn.ConvTranspose2d(2*ch, 2*ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            CONV(3 * ch, 2 * ch, 'relu', False, 1),
            CONV(2 * ch, 2 * ch, 'relu', False, 1),
            nn.ConvTranspose2d(2 * ch, 2 * ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            CONV(3 * ch, 2 * ch, 'relu', False, 1),
            CONV(2 * ch, 2 * ch, 'relu', False, 1),
            nn.ConvTranspose2d(2 * ch, 2 * ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            CONV(3 * ch, 2 * ch, 'relu', False, 1),
            CONV(2 * ch, 2 * ch, 'relu', False, 1),
            nn.ConvTranspose2d(2 * ch, 2 * ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv0 = nn.Sequential(
            CONV(3 * ch, 64, 'relu', False, 1),
            CONV(64, 32, 'relu', False, 1),
            CONV(32, 3, 'lrelu', False, 1),
        )

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        tmp = self.conv5(x4)
        tmp = self.deconv5(tmp)
        tmp = torch.cat((x4, tmp), dim=1)
        tmp = self.deconv4(tmp)
        tmp = torch.cat((x3, tmp), dim=1)
        tmp = self.deconv3(tmp)
        tmp = torch.cat((x2, tmp), dim=1)
        tmp = self.deconv2(tmp)
        tmp = torch.cat((x1, tmp), dim=1)
        tmp = self.deconv1(tmp)
        tmp = torch.cat((x0, tmp), dim=1)
        tmp = self.deconv0(tmp)
        return tmp.view(B, F, C, H, W)


class G2(nn.Module):
    def __init__(self, ch=32):
        super(G2, self).__init__()
        self.conv1 = nn.Sequential(
            CONV(9, 90, 'relu', True, 1),
            CONV(90, ch, 'relu', True, 1)
        )
        self.conv2 = nn.Sequential(
            CONV(ch, 2*ch, 'relu', True, 2),
            CONV(2*ch, 2*ch, 'relu', True, 1),
            CONV(2*ch, 2*ch, 'relu', True, 1)
        )
        self.conv3 = nn.Sequential(
            CONV(2*ch, 4*ch, 'relu', True, 2),
            CONV(4*ch, 4*ch, 'relu', True, 1),
            CONV(4*ch, 4*ch, 'relu', True, 1),
            CONV(4*ch, 4*ch, 'relu', True, 1),
            CONV(4*ch, 4*ch, 'relu', True, 1),
            CONV(4*ch, 8*ch, 'relu', True, 1),
            nn.PixelShuffle(2)
        )
        self.conv4 = nn.Sequential(
            CONV(2*ch, 2*ch, 'relu', True, 1),
            CONV(2*ch, 2*ch, 'relu', True, 1),
            CONV(2*ch, 4*ch, 'relu', True, 1),
            nn.PixelShuffle(2)
        )
        self.conv5 = nn.Sequential(
            CONV(ch, ch, 'relu', True, 1),
            CONV(ch, 3, 'relu', True, 1),
        )

    def forward(self, x):
        """

        :param x: [B,9,256,256]
        :return:
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = x2 + x3
        x4 = self.conv4(x3)
        x4 = x1 + x4
        x5 = self.conv5(x4)
        x5 = x[:, 3:6, ...] + x5
        return x5


class MGTV1Generator(nn.Module):
    def __init__(self, args):
        super(MGTV1Generator, self).__init__()
        self.nframes = args.nframes

        self.g1 = G1(args.ch1)
        self.g2_1 = G2(args.ch2)
        self.g2_2 = G2(args.ch2)

    def forward(self, x):
        """
        :param x: e.g. [4, 5, 3, 256, 256]
        :return:
        """
        self.HR_Gs = self.g1(x)  # [4, 5, 3, 256, 256]
        l = []
        for i in range(self.nframes-2):
            l.append(self.g2_1(torch.cat((self.HR_Gs[:, i, ...], self.HR_Gs[:, i+1, ...], self.HR_Gs[:, i+2, ...]), dim=1)))
        return self.HR_Gs, self.g2_2(torch.cat(l, dim=1))


def define_G(opt):
    net = MGTV1Generator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
