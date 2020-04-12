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

SPMCS_dataset_HRLR2AB(dataset_path="/opt/data/private/datasets/SPMCS/test_set",
                      ABpath="/opt/data/private/datasets/SPMCS")