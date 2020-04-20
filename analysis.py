"""
to analysis result after run test.sh
only deal with Y channel
for load_epoch option, only consider epoch_xxx style, thus don't include latest etc.
should run  `python -m visdom.server` first

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



use:

e.g.
python analysis.py --name vimeo_rbpn_04_05_13_46
"""
import os
import time
import argparse
from options import str2bool
from collections import OrderedDict
import matplotlib.pyplot as plt
from visdom import Visdom

# add parser
parser = argparse.ArgumentParser(description="analysis test result")
parser.add_argument('--result_dir', type=str, default="./results", help="result dir")
parser.add_argument('--name', type=str, default="vimeo_rbpn_04_05_13_46", help="checkpoints and result dir name")
parser.add_argument('--analysis_result_dir_name', type=str, default="analysis")
parser.add_argument('--iteration_in_one_epoch', type=int, default=55020)

opt = parser.parse_args()

def get_iqa_from_txt(path):

    def get_key_value(s):
        l = s.split(":")
        assert len(l) == 2
        l[1] = l[1].split("/")[0]
        return l[0].strip(), float(l[1])

    Dict = OrderedDict()
    with open(path, "r") as f:
        iqalist = f.readlines()[2].strip().split("  ")
        iqalist = list(map(lambda s: s.strip(), iqalist))
        for item in iqalist:
            key, value = get_key_value(item)
            Dict[key] = value
        return Dict


def add_header(header_list, l):
    for i in range(len(header_list), len(l)):
        splited = l[i].split("_")
        header_list.append(splited[0])


viz = Visdom(env='vimeo_rbpn_04_05_13_46')


def plot(dataset, iqaname, epochs, results):
    # print(dataset, iqaname)
    # print(epochs)
    # print(results)
    for i in range(len(epochs)):
        epochs[i] = epochs[i] * opt.iteration_in_one_epoch

    viz.line(
        Y=results,
        X=epochs,
        win="{}_{}".format(iqaname, dataset),
        opts=dict(title="{} on dataset {}".format(iqaname.upper(), dataset),
                  markers=True,
                  legend=[iqaname.upper(), ],
                  xlabel='iterations',
                  ylabel=iqaname.upper()),
    )


if __name__ == '__main__':
    consider_iqa = ('psnr', 'ssim')
    header_list = []
    header_list.append("phase")
    header_list.append("dataset")
    origin_length = len(header_list)
    header_ok_flag = False
    List = []
    result_dir_path = os.path.join(opt.result_dir, opt.name)
    for cur_file in sorted(os.listdir(result_dir_path)):
        path = os.path.join(result_dir_path, cur_file)  # get absolute path
        if os.path.isdir(path):
            pass
        else:  # is file
            if path.endswith("results.txt") and "epoch_" in path:
                l = cur_file.split("-")[:-1]

                if not header_ok_flag:
                    add_header(header_list, l)
                    header_ok_flag = True

                for i in range(origin_length, len(l)):
                    splited = l[i].split("_")
                    l[i] = "_".join(splited[1:])
                    if splited[0] == "epoch":
                        l[i] = int(l[i])

                iqa_dict = get_iqa_from_txt(path)
                for item in iqa_dict.items():
                    l.append(item[1])
                List.append(l)
            else:
                pass
    for item in iqa_dict.items():
        header_list.append(item[0])

    # print(header_list)
    # for item in List:
    #     print(item)

    dataset_header_id = header_list.index("dataset")
    epoch_header_id = header_list.index("epoch")
    consider_iqa_header_id = [0] * len(consider_iqa)
    for i, item1 in enumerate(consider_iqa):
        for j, item2 in enumerate(header_list):
            if item1 in item2:
                consider_iqa_header_id[i] = j
                break

    datasets = set()
    for item in List:
        datasets.add(item[dataset_header_id])



    for dataset in sorted(list(datasets)):
        needlist = []
        for item in List:
            if item[dataset_header_id] == dataset:
                needlist.append(item)
        needlist.sort(key=lambda x: x[epoch_header_id])
        for id, iqa in enumerate(consider_iqa):
            x = []
            y = []
            for item in needlist:
                x.append(item[epoch_header_id])
                y.append(item[consider_iqa_header_id[id]])
            plot(dataset=dataset, iqaname=iqa, epochs=x, results=y)
