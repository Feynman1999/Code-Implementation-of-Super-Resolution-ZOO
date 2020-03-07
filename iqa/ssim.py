import numpy as np
import math

def ssim(HR_G, HR_GroundTruth, only_Luminance=True, crop=0):
    return 1