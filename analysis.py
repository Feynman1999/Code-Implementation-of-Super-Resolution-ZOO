"""
to analysis result after run test.sh
only deal with Y channel!

e.g.
test-Vid4-epoch_5-ensemble_False-results.txt
test-Vid4-epoch_10-ensemble_False-results.txt
test-SPMCS-epoch_10-ensemble_False-results.txt
test-SPMCS-epoch_10-ensemble_False-results.txt

in xxx.txt:
================ Result with psnr / ssim (Y / rgb) (Mon Apr 13 22:08:25 2020) ================

               average psnr: 26.7214 / 25.2044   average ssim: 1.0000 / 1.0000

           calendar           :   psnr: 23.4931 / 21.7502    ssim: 1.0000 / 1.0000
             city             :   psnr: 27.3135 / 25.7842    ssim: 1.0000 / 1.0000
           foliage            :   psnr: 25.9219 / 24.5025    ssim: 1.0000 / 1.0000
             walk             :   psnr: 30.1570 / 28.7807    ssim: 1.0000 / 1.0000
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
parser.add_argument('--analysis_result_dir_name', type=str, default="analysis")
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
