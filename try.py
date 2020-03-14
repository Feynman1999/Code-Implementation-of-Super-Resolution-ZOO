import matplotlib.pyplot as plt
import numpy as np
from util import util


def bgr2ycbcr(img, only_y=True):
    """
    bgr version of rgb2ycbcr
    :param img: uint8, [0, 255]  or float, [0, 1],  [h,w,c] for image and [b,h,w,c] for video,  ndarray
    :param only_y: only return Y channel
    :return:
    """
    assert img.shape[-1] == 3
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

a = np.zeros((10, 100, 100, 3), dtype=np.uint8)

print(bgr2ycbcr(a, only_y=True).shape)