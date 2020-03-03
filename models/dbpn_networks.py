from . import base_networks
import torch
import torch.nn as nn



class UPU(nn.Module):
    """
        up-projection units
    """
    def __init__(self, channels, kernel_size, stride, padding):
        super(UPU, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)
        self.PRelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PRelu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.deconv2 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)
        self.PRelu3 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x1):
        x2 = self.deconv1(x1)
        x2 = self.PRelu1(x2)

        x3 = self.conv1(x2)
        x3 = self.PRelu2(x3)

        x3 = x3 - x1

        x4 = self.deconv2(x3)
        x4 = self.PRelu3(x4)

        x4 = x4 + x2

        return x4


class DPU(nn.Module):
    """
        down-projection units
    """
    def __init__(self, channels, kernel_size, stride, padding):
        super(DPU, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PRelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.deconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0)
        self.PRelu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PRelu3 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.PRelu1(x2)

        x3 = self.deconv1(x2)
        x3 = self.PRelu2(x3)

        x3 = x3 - x1

        x4 = self.conv2(x3)
        x4 = self.PRelu3(x4)

        x4 = x4 + x2

        return x4


class DbpnGenerator(nn.Module):
    def __init__(self, args):
        super(DbpnGenerator, self).__init__()

        n_0 = args.n_0
        n_R = args.n_R
        SR_factor = args.SR_factor
        kernel_size = SR_factor+4
        stride = SR_factor
        padding = 2
        self.iterations_num = args.iterations_num


        self.conv1 = nn.Conv2d(args.input_nc, n_0, kernel_size=3, padding=1, stride=1)
        self.PRelu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.conv2 = nn.Conv2d(n_0, n_R, kernel_size=1, padding=0, stride=1)
        self.PRelu2 = nn.PReLU(num_parameters=1, init=0.25)
        self.UPU = UPU(n_R, kernel_size, stride, padding)
        self.DPU = DPU(n_R, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(n_R*self.iterations_num, args.output_nc, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        H_list = []
        x = self.conv1(x)
        x = self.PRelu1(x)
        x = self.conv2(x)
        x = self.PRelu2(x)

        for i in range(self.iterations_num-1):
            H = self.UPU(x)
            H_list.append(H)
            x = self.DPU(H)

        H_list.append(self.UPU(x))
        x = torch.cat(H_list, dim=1)
        x = self.conv3(x)
        return x


def define_dbpn_net(opt):
    net = DbpnGenerator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)