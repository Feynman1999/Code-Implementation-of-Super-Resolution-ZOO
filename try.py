from util.util_dataset import *


# video_dataset_onlyHR2AB("./datasets/youku/B", "./datasets/youku/train")

# video_dataset_HRLR2AB(HRpath="./datasets/youku/B", LRpath="./datasets/youku/A", ABpath="./datasets/youku/train")

vimeo90K_dataset_onlyHR2AB(dataset_path="/opt/data/private/datasets/vimeo_septuplet/vimeo_septuplet",
                           ABpath="/opt/data/private/datasets/vimeo_septuplet",
                           phase="train",
                           factor=4)