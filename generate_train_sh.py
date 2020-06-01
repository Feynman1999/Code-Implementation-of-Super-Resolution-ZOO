"""
    do train on linux and save train.sh
    you need to run train.sh by yourself
"""
import os
import time
import argparse
from options import str2bool
import re
import platform

# add parser
parser = argparse.ArgumentParser(description="generate train sh for training")
parser.add_argument('--nohup', type=str2bool, default=True)
parser.add_argument('--append', type=str2bool, default=True)
opt = parser.parse_args()
content = '# ================ train shell ( {} ) ================\n'.format(time.strftime("%c"))

options = \
"""
        --dataroot          /opt/data/private/datasets/mgtv
        --name              mgtv_mgtv3
        --model             mgtv3
        --display_freq      2400
        --print_freq        240
        --save_epoch_freq   500
        --gpu_ids           0
        --batch_size        3
        --suffix            06_02_00_01
        --crop_size         256
        --imgseqlen         5
        --seed              2
        --max_consider_len  125
        --scenedetect       True
        --ch1               128
        --ch2               32
        --num_threads       11
"""

if __name__ == '__main__':
    # assert platform.system().lower() == 'linux'
    options = options.replace("\n", "  ")
    options = '  '.join(options.split())
    assert options.split("  ")[2].strip() == "--name"
    with open("./train.sh", "w+") as f:
        if opt.nohup:
            append = ">>" if opt.append else ">"
            command = "nohup python3 -u train.py " + options + "  {} /opt/data/private/{}.log 2>&1 &".format(append, options.split("  ")[3].strip())
        else:
            command = "python3 train.py " + options
        content += command
        print(content)
        f.write(content)
