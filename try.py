from util.util_dataset import *
import ffmpeg
import sys
import pickle

# dataset_images2video(datasetpath = "./results/mgtv_mgtv1_add_scene_05_28_12_51/apply-A-epoch_1000-block_size_2_3", fps=25, suffix=".y4m")
# dataset_images2video(datasetpath = "./results/mgtv_mgtv1_add_scene_05_28_12_51/apply-A-epoch_1000-block_size_2_3", fps=25, suffix=".y4m")
images2video(dirpath="./checkpoints/trump_siren_07_05_19_13/images", fps=60, must_have="restore")
# if __name__ == '__main__':
# #     # videodataset_pre_crop(path2AB="/opt/data/private/datasets/mgtv/train", crop_size=256)
#      videodataset_pre_crop("./datasets/mgtv/train/A", crop_size=256, process_num=2)
#      videodataset_pre_crop("./datasets/mgtv/train/B", crop_size=256, process_num=2)

# path = "/opt/data/private/datasets/mgtv/GTvideos/"
# videodataset_scenedetect(path)
#
# with open(os.path.join(path, 'scene.pickle'), 'rb') as f:
#     b = pickle.load(f)
#     print(b)





# (
#     ffmpeg
#     .input("./datasets/mgtv/apply/A/mg_test_0800_damage/*.png", pattern_type='glob', framerate=25)
#     .output('mg_test_0800_damage.y4m')
#     .run()
# )

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

# import VSR

# video_dataset_onlyHR2AB("./datasets/demo/HR", "./datasets/demo", phase="test")

# video_dataset_onlyHR2AB("/opt/data/private/datasets/demo/HR", "/opt/data/private/datasets/demo", phase="test")

# videos_to_images(videos_path="./datasets/mgtv/test_damage_A", path2place="./datasets/mgtv")

# videos_to_images(videos_path="/opt/data/private/datasets/mgtv/Noisevideos", path2place="/opt/data/private/datasets/mgtv", phase="train", AorB="A")
# videos_to_images(videos_path="/opt/data/private/datasets/mgtv/GTvideos", path2place="/opt/data/private/datasets/mgtv", phase="train", AorB="B")
# videos_to_images(videos_path="./datasets/mgtv/test_damage_A", path2place="./datasets/mgtv", phase="apply")
# videos_to_images(videos_path="/opt/data/private/datasets/mgtv/test_damage_A", path2place="/opt/data/private/datasets/mgtv", phase="apply")

from util.compare import compare


# compare(dataroot="C:/Users/76397/Desktop/compare/different_size1", x=85, y=105)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_size2", x=15, y=85)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_size3", x=35, y=45)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_method_vid4_1", x=42, y=47)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_method_vid4_2", x=63, y=19)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_method_vid4_3", x=94, y=60)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_method_spmc_1", x=110, y=70)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_method_spmc_2", x=110, y=25)
# compare(dataroot="C:/Users/76397/Desktop/compare/different_method_spmc_3", x=135, y=5)
