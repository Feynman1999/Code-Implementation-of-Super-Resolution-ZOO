from util.util_dataset import *


# video_dataset_onlyHR2AB("./datasets/youku/B", "./datasets/youku/train")

# video_dataset_HRLR2AB(HRpath="./datasets/youku/B", LRpath="./datasets/youku/A", ABpath="./datasets/youku/train")

# vimeo90K_dataset_onlyHR2AB(dataset_path="/opt/data/private/datasets/vimeo_septuplet/vimeo_septuplet",
#                            ABpath="/opt/data/private/datasets/vimeo_septuplet",
#                            phase="train",
#                            factor=4)

# for path in os.listdir("./A/"):
#     allpath = os.path.join("./A/", path)
#     assert os.path.isdir(allpath)
#     if len(os.listdir(allpath)) != 7:
#         print(allpath)

# from DCN import *

# SPMCS_dataset_HRLR2AB()

# SPMCS_dataset_HRLR2AB(dataset_path="/opt/data/private/datasets/SPMCS/test_set",
#                       ABpath="/opt/data/private/datasets/SPMCS")

# SPMCS_dataset_onlyHR2AB(dataset_path="/opt/data/private/datasets/SPMCS/test_set",
#                         ABpath="/opt/data/private/datasets/SPMCS")


#!/usr/bin/env python

import VSR

#video_dataset_onlyHR2AB("/opt/data/private/datasets/demo/HR", "/opt/data/private/datasets/demo", phase="test")

import cv2
import numpy as np
fp_img = cv2.imread('./1.jpg', 1)
fp_img = cv2.cvtColor(fp_img, cv2.COLOR_BGR2GRAY)
print(fp_img.shape)
# 快速傅里叶变换算法得到频率分布
f_img = np.fft.fft2(fp_img)
# 默认结果中心点位置是在左上角,
# 调用fftshift()函数转移到中间位置
fshift_img = np.fft.fftshift(f_img)
# fft结果是复数, 其绝对值结果是振幅
fshift_img2 = np.log(np.abs(fshift_img))
print(fshift_img2.shape)
