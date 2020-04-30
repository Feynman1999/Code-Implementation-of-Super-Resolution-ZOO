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
        --dataroot          /opt/data/private/datasets/vimeo_septuplet
        --name              vimeo_tanet3
        --model             tanet3
        --display_freq      4800
        --print_freq        4800
        --gpu_ids           0,1,2
        --batch_size        6
        --suffix            04_21_09_24
        --crop_size         64
        --imgseqlen         7
        --save_epoch_freq   5
        --seed              4
        --continue_train    True
        --load_epoch        epoch_110
        --epoch_count       111
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
