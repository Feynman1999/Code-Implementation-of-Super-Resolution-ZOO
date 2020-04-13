"""

"""
import os
import time
import argparse
from options import str2bool
from collections import OrderedDict
import matplotlib.pyplot as plt

# add parser
parser = argparse.ArgumentParser(description="analysis test result")
parser.add_argument('--result_dir', type=str, default="./results", help="result dir")
parser.add_argument('--name', type=str, default="vimeo_rbpn_04_05_13_46", help="checkpoints and result dir name")
opt = parser.parse_args()


def get_iqa_from_txt(path):

    def get_key_value(s):
        l = s.split(":")
        assert len(l) == 2
        return l[0].strip(), float(l[1])

    Dict = OrderedDict()
    with open(path, "r") as f:
        iqalist = f.readlines()[2].strip().split("  ")
        iqalist = list(map(lambda s: s.strip(), iqalist))
        for item in iqalist:
            key, value = get_key_value(item)
            Dict[key] = value
        return Dict


if __name__ == '__main__':
    x = []
    y = []
    result_dir_path = os.path.join(opt.result_dir, opt.name)
    for cur_file in os.listdir(result_dir_path):
        path = os.path.join(result_dir_path, cur_file)  # get absolute path
        if os.path.isdir(path):
            pass
        else:  # is file
            if path.endswith("results.txt"):
                l = cur_file.split("-")[:-1]
                if l[1] == 'SPMCS' and l[-1].endswith("True"):
                    x.append(int(l[2].split("_")[-1]))
                    y.append(get_iqa_from_txt(path)['average psnr'])
                # print(get_iqa_from_txt(path))
            else:
                pass

    x.pop(8)
    y.pop(8)
    x.insert(0, 5)
    y.insert(0, 25.5155)
    print(x)
    print(y)

    plt.plot(x, y, linestyle='-.')

    plt.title('RBPN (only Luminance channel) on Vid4')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')

    plt.show()