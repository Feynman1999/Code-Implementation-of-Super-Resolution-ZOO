from iqa.psnr import psnr
from util.util import tensor2im
from data.image_folder import make_images_dataset
import PIL
from PIL import Image
import numpy as np
import torch

rl1 = []
rl2 = []
rl3 = []
rl4 = []
img_list = make_images_dataset(r"E:\git_repo\Code-Implementation-of-Super-Resolution-ZOO\datasets\Set5\test\B")
for path in img_list:

    img = PIL.Image.open(path).convert("RGB")

    nd = np.array(img)

    w, h = img.size
    factor = 4

    img_l = img.resize((w//factor, h//factor), resample=Image.BICUBIC)


    # USE PIL
    img_bicubic1 = img_l.resize((w, h), resample=Image.BICUBIC)

    nd1 = np.array(img_bicubic1)

    rl1.append(psnr(nd, nd1, only_Luminance=True, crop=4))

    # use torch1
    img_ll = torch.from_numpy(np.array(img_l, dtype=np.float32)/255).permute(2,0, 1).unsqueeze(0) # totenser 0~1
    img_bicubic2 = torch.nn.functional.interpolate(img_ll, scale_factor=4, mode='bicubic', align_corners=False)
    nd2 = tensor2im(img_bicubic2[0])
    rl2.append(psnr(nd, nd2, only_Luminance=True, crop=4))

    # use torch2
    img_lll = torch.from_numpy(np.array(img_l, dtype=np.float32)).permute(2,0, 1).unsqueeze(0) # totenser 0~255
    img_bicubic3 = torch.nn.functional.interpolate(img_lll, scale_factor=4, mode='bicubic', align_corners=False)
    nd3 = tensor2im(img_bicubic3[0]/255)
    rl3.append(psnr(nd, nd3, only_Luminance=True, crop=4))

    # use torch3
    img_llll = torch.from_numpy(np.array(img_l, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)  # totenser 0~255
    img_bicubic4 = torch.nn.functional.interpolate(img_llll, scale_factor=4, mode='bicubic', align_corners=False)
    nd4 = img_bicubic4[0].add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    rl4.append(psnr(nd, nd4, only_Luminance=True, crop=4))

print(sum(rl1)/ len(rl1))
print(sum(rl2)/ len(rl2))
print(sum(rl3)/ len(rl3))
print(sum(rl4)/ len(rl4))