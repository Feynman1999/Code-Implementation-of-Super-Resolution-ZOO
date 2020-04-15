import numpy as np
import math
import cv2
from . import rgb2ycbcr
import scipy.ndimage
from skimage.metrics import structural_similarity


def ssim(HR_G, HR_GroundTruth, only_Luminance=True, crop=0):
    """
    :param HR_G: [h,w,c] for image and [b,h,w,c] for video,  ndarray
    :param HR_GroundTruth:
    :param only_Luminance: Whether to use only Luminance channel.
        use rgb2ycbcr function.  same as matlab rgb2ycbcr.
    :param crop: For SR by factor s, we crop s pixels near image boundary before evaluation
    :return: ssim value (frame average for video)
    """
    assert HR_GroundTruth.shape == HR_G.shape
    assert isinstance(HR_G, np.ndarray) and isinstance(HR_GroundTruth, np.ndarray)
    assert HR_GroundTruth.dtype == HR_G.dtype
    assert HR_G.dtype == np.uint8

    video_flag = False
    if len(HR_G.shape) == 4:
        video_flag = True

    # crop
    h, w = HR_G.shape[-3], HR_G.shape[-2]
    assert h > 2 * crop and w > 2 * crop, 'crop boundary size too large'
    HR_G = HR_G[..., crop:h - crop, crop:w - crop, :]
    HR_GroundTruth = HR_GroundTruth[..., crop:h - crop, crop:w - crop, :]

    HR_G = HR_G.astype(np.float32)
    HR_GroundTruth = HR_GroundTruth.astype(np.float32)

    # if true, [h,w,c] -> [h,w]  or  [b,h,w,c] -> [b,h,w]
    if only_Luminance:
        HR_G = rgb2ycbcr(HR_G)
        HR_GroundTruth = rgb2ycbcr(HR_GroundTruth)

    def one_frame(f1, f2):
        '''calculate SSIM
        '''
        # func = cal_ssim1
        # func = cal_ssim2
        if not f1.shape == f2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if f1.ndim == 2:
            return structural_similarity(f1, f2, gaussian_weights=True, data_range=255, use_sample_covariance=False, sigma=1.5)
        elif f1.ndim == 3:
            if f1.shape[2] == 3:
                return structural_similarity(f1, f2, multichannel=True, gaussian_weights=True, data_range=255, use_sample_covariance=False, sigma=1.5)
            elif f1.shape[2] == 1:
                return structural_similarity(np.squeeze(f1), np.squeeze(f2), gaussian_weights=True, data_range=255, use_sample_covariance=False, sigma=1.5)
            else:
                raise NotImplementedError("unknown channel nums: {}".format(f1.shape[2]))
        else:
            raise ValueError('Wrong input image dimensions.')

    if video_flag:
        # for [b,h,w,c] or [b,h,w]
        frame_result_list = []
        for i in range(HR_G.shape[0]):
            frame_result_list.append(one_frame(HR_G[i], HR_GroundTruth[i]))
        return sum(frame_result_list) / len(frame_result_list)

    else:
        # for [h,w] or [h,w,c]
        return one_frame(HR_G, HR_GroundTruth)


# from EDVR repo
def cal_ssim1(img1, img2, l=255):
    C1 = (0.01 * l)**2
    C2 = (0.03 * l)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


# from PFNL repo
def cal_ssim2(im1, im2, l=255):
    # k1,k2 & c1,c2 depend on L (width of color map)
    k_1 = 0.01
    c_1 = (k_1 * l)**2
    k_2 = 0.03
    c_2 = (k_2 * l)**2

    # window = np.ones((8, 8))

    window = gauss_2d((11, 11), 1.5)
    # Normalization
    # window /= np.sum(window)

    # Convert image matrices to double precision (like in the Matlab version)
    im1 = im1.astype(np.float)
    im2 = im2.astype(np.float)

    # Means obtained by Gaussian filtering of inputs
    mu_1 = scipy.ndimage.filters.convolve(im1, window)
    mu_2 = scipy.ndimage.filters.convolve(im2, window)

    # Squares of means
    mu_1_sq = mu_1**2
    mu_2_sq = mu_2**2
    mu_1_mu_2 = mu_1 * mu_2

    # Squares of input matrices
    im1_sq = im1**2
    im2_sq = im2**2
    im12 = im1 * im2

    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = scipy.ndimage.filters.convolve(im1_sq, window)
    sigma_2_sq = scipy.ndimage.filters.convolve(im2_sq, window)

    # Covariance
    sigma_12 = scipy.ndimage.filters.convolve(im12, window)

    # Centered squares of variances
    sigma_1_sq -= mu_1_sq
    sigma_2_sq -= mu_2_sq
    sigma_12 -= mu_1_mu_2

    if (c_1 > 0) & (c_2 > 0):
        ssim_map = ((2 * mu_1_mu_2 + c_1) * (2 * sigma_12 + c_2)) / \
            ((mu_1_sq + mu_2_sq + c_1) * (sigma_1_sq + sigma_2_sq + c_2))
    else:
        numerator1 = 2 * mu_1_mu_2 + c_1
        numerator2 = 2 * sigma_12 + c_2

        denominator1 = mu_1_sq + mu_2_sq + c_1
        denominator2 = sigma_1_sq + sigma_2_sq + c_2

        ssim_map = np.ones(mu_1.size)

        index = (denominator1 * denominator2 > 0)

        ssim_map[index] = (numerator1[index] * numerator2[index]) / \
            (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    # return MSSIM
    index = np.mean(ssim_map)

    return float(index)


def gauss_2d(shape=(3, 3), sigma=0.5):
    """
    Code from Stack Overflow's thread
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
