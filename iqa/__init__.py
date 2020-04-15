"""
Image Quality Assessment
"""
import importlib
import numpy as np


def find_function_using_name(iqa_name):
    """Import the module "iqa/iqa_name.py".

    In the file, the function called iqa_name() will be return. case-insensitive. remove _ in iqa_name
    """
    iqa_filename = "iqa." + iqa_name
    iqalib = importlib.import_module(iqa_filename)
    iqa = None
    target_func_name = iqa_name.replace('_', '')
    for name, func in iqalib.__dict__.items():
        if name.lower() == target_func_name.lower() \
           and hasattr(func, '__call__'):
            iqa = func

    if iqa is None:
        print("In %s.py, there should be a function name that matches %s in lowercase." % (iqa_filename, target_func_name))
        exit(0)

    return iqa


def rgb2ycbcr(img, only_y=True):
    """
    bgr version of rgb2ycbcr
    :param img: uint8, [0, 255]  or float, [0, 1],  [h,w,c] for image and [b,h,w,c] for video,  ndarray
    :param only_y: only return Y channel
    :return: [h,w] or [b,h,w] for only Y ; [h,w,c] or [b,h,w,c] for ycbcr
    """
    assert img.shape[-1] == 3
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)