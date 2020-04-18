"""
    do test on windows or linux and save test.sh
    you need to run test.sh by yourself, and after that, you can run analysis.py to analysis the test result

    e.g.
    python generate_test_sh.py --name vimeo_tanet_04_16_18_10  --model tanet
"""
import os
import time
import argparse
from options import str2bool
import platform

# add parser
parser = argparse.ArgumentParser(description="generate test sh for analysis")
parser.add_argument('--datasetnames', type=str, default="Vid4, SPMCS, vimeo_septuplet")
parser.add_argument('--name', type=str, default="vimeo_rbpn_04_05_13_46", help="checkpoints dir name")
parser.add_argument('--model', type=str, default="rbpn", help="model name")
parser.add_argument('--auto_load', type=str2bool, default=True, help="auto find xxx.pth in the checkpoints dir")
parser.add_argument('--video_flag', type=str2bool, default=True)
parser.add_argument('--load_epoch', type=str, default="50, 100, 5", help="if do not auto load, use this, range(50,100,5)")
opt = parser.parse_args()

model_EXTENSIONS = [
    '.pth',
]


def is_model_file(filename):
    return any(filename.endswith(extension) for extension in model_EXTENSIONS)


def make_models_dataset(dir):
    models = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_model_file(fname):
                models.append(fname)
    return models


def generate_test_sh_for_one_algorithm(model, name, load_auto_flag):
    """

    :param model:
    :param name:
    :param load_auto_flag:
    :return:
    """
    if platform.system().lower() == 'windows':
        template = "python test.py --dataroot {}  --name {} --model {} --load_epoch {} --ensemble {}"
    elif platform.system().lower() == 'linux':
        # logfile = "/opt/data/private/test_{}_{}.log 2>&1 &".format(opt.model, opt.name)
        template = "python3 test.py --dataroot {}  --name {} --model {} --load_epoch {} --ensemble {}"
    else:
        raise NotImplementedError("unknow platform: {}!".format(platform.system().lower()))

    command = ""

    datasetnames = list(map(lambda s: s.strip(), opt.datasetnames.split(",")))

    if load_auto_flag:
        # find with checkpoints/name/epoch_xxxx_xxxxxxxxx.pth
        model_names = sorted(make_models_dataset(os.path.join("./checkpoints", opt.name)))
        load_epochs = list(map(lambda s: s[:s.find("_net")], model_names))
    else:
        start, end, gap = list(map(lambda s: int(s.strip()), opt.load_epoch.split(",")))
        load_epochs = list(map(lambda s: "epoch_{}".format(s), range(start, end, gap)))
        load_epochs.append("latest")

    for datasetname in datasetnames:
        if datasetname == 'vimeo_septuplet':  # due to the vimeo testset is too large, we only run test for latest model
            epochs = [load_epochs[-1], ]
        else:
            epochs = load_epochs
        for load_epoch in epochs:
            for emsemble_flag in ("False", ) if opt.video_flag else ("True", "False"):
                if platform.system().lower() == 'linux':
                    str = template.format(os.path.join('/opt/data/private/datasets', datasetname).replace('\\', '/'), name, model, load_epoch, emsemble_flag)
                elif platform.system().lower() == 'windows':
                    str = template.format(os.path.join('./datasets', datasetname).replace('\\', '/'), name, model, load_epoch, emsemble_flag)
                print(str)
                command += str
                command += "\n"

    # print(command)
    with open("./test.sh", "w+") as f:
        now = time.strftime("%c")
        content = '# ================ test shell ( {} ) ================\n'.format(now)
        content += command
        f.write(content)


if __name__ == '__main__':
    generate_test_sh_for_one_algorithm(opt.model, opt.name, load_auto_flag=opt.auto_load)
