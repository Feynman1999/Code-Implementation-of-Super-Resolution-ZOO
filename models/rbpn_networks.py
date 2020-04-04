from . import base_networks
import torch
import torch.nn as nn


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


class SISR_Block(nn.Module):
    def __init__(self, cl, ch):
        super(SISR_Block, self).__init__()
        self.num_stages = 3
        self.pre_deal = nn.Conv2d(cl, ch, kernel_size=1, stride=1, padding=0)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.UPU1 = UPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.UPU2 = UPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.UPU3 = UPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.DPU1 = DPU(channels = ch, kernel_size=8, stride=4, padding=2)
        self.DPU2 = DPU(channels = ch, kernel_size=8, stride=4, padding=2)
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


class RBPNGenerator(nn.Module):
    def __init__(self, args):
        super(RBPNGenerator, self).__init__()
        cl = args.cl
        cm = args.cm
        ch = args.ch

        self.nframes = args.nframes

        #Initial Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.input_nc, cl, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.input_nc*2+0, cm, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        # projection module
        self.Projection = Projection_Module(args)

        # reconstruction module
        self.reconstruction = nn.Conv2d((self.nframes-1)*ch, args.output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """

        :param x: e.g. [4, 5, 3, 64, 64]
        :return:
        """
        mid = self.nframes // 2
        # [0, mid)  [mid+1, nframes)
        L = self.conv1(x[:, mid, ...])
        Hlist = []
        for id in range(self.nframes):
            if id == mid:
                continue
            M = self.conv2(torch.cat((x[:, mid, ...], x[:, id, ...]), dim=1))
            H, L = self.Projection(M, L)
            Hlist.append(H)

        return self.reconstruction(torch.cat(Hlist, dim=1))


def define_G(opt):
    net = RBPNGenerator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
