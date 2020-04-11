from . import base_networks
import torch
import torch.nn as nn


class TANETGenerator(nn.Module):
    def __init__(self, args):
        super(TANETGenerator, self).__init__()
        pass

    def forward(self, x):
        pass


def define_G(opt):
    net = TANETGenerator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
