# 指定一个左上角的位置和size进行截取 LR 的加框 其余的截取并保存
import cv2
import os
from data.image_folder import make_images_dataset
from util.util_dataset import get_file_name


def compare(dataroot, size=32, x=0, y=0, scale=4):  # x: w  y: h
    images = make_images_dataset(dataroot)
    for path in images:
        image = cv2.imread(path)
        if "lr" in path.lower():
            image = cv2.rectangle(image, (x, y), (x+size, y+size), (0, 0, 255), 2)
        else:
            image = image[y*scale:y*scale+size*scale, x*scale:x*scale+size*scale, :]
        file_name = get_file_name(path)
        file_name = file_name+"_deal.png"
        cv2.imwrite(os.path.join(dataroot, file_name), image)
