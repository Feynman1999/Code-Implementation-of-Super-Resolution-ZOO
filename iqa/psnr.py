import numpy as np
import math
from PIL import Image


def psnr(HR_G, HR_GroundTruth, only_Luminance=True, crop=0):
    """
    :param HR_G: [h,w,c] for image and [b,h,w,c] for video
    :param HR_GroundTruth:
    :param only_Luminance: Whether to use only Luminance channel.use Y = 0.2126 R + 0.7152 G + 0.0722 B  https://www.itu.int/rec/R-REC-BT.709
    :param crop: For SR by factor s, we crop s pixels near image boundary before evaluation
    :return: psnr value (frame average for video)
    """
    assert HR_GroundTruth.shape == HR_G.shape
    video_flag = False
    if len(HR_G.shape) == 4:
        video_flag = True
    # crop
    h, w = HR_G.shape[-3], HR_G.shape[-2]
    assert h > 2*crop and w > 2*crop, 'crop boundary size too large'
    HR_G = HR_G[..., crop:h-crop, crop:w-crop, :]
    HR_GroundTruth = HR_GroundTruth[..., crop:h-crop, crop:w-crop, :]

    # if true, [h,w,c] -> [h,w]  or  [b,h,w,c] -> [b,h,w]
    if only_Luminance:
        ratio = (0.2126, 0.7152, 0.0722)
        HR_G = ratio[0]*HR_G[..., 0] + ratio[1]*HR_G[..., 1] + ratio[2]*HR_G[..., 2]
        HR_GroundTruth = ratio[0]*HR_GroundTruth[..., 0] + ratio[1]*HR_GroundTruth[..., 1] + ratio[2]*HR_GroundTruth[..., 2]

    def one_frame(f1, f2, fakeval = 100):
        mse = np.mean((f1 * 1.0 - f2 * 1.0) ** 2)
        if mse < 1.0e-10:
            return fakeval
        return 10 * math.log10(255.0 ** 2 / mse)

    if video_flag:
        # for [b,h,w,c] or [b,h,w]
        frame_list = []
        for i in range(HR_G.shape[0]):
            frame_list.append(one_frame(HR_G[i], HR_GroundTruth[i]))
        return sum(frame_list)/len(frame_list)

    else:
        # for [h,w] or [h,w,c]
        return one_frame(HR_G, HR_GroundTruth)