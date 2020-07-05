from . import base_networks
from .siren import SirenNet
import torch.nn as nn

class sirenGenerator(nn.Module):
    def __init__(self, args):
        super(sirenGenerator, self).__init__()
        self.sirennet = SirenNet(dim_in=2, dim_hidden=256, dim_out=3, num_layers=5)

    def forward(self, x):
        return self.sirennet(x)


def define_siren_net(opt):
    net = sirenGenerator(opt)
    return base_networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)