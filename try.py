from util.util_dataset import *
import ffmpeg
import sys

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


def get_video_info(in_file):
    """
    获取视频基本信息
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            print('No video stream found', file=sys.stderr)
            sys.exit(1)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))
        sys.exit(1)

video_path = "./datasets/mgtv/test_damage_A/mg_test_0802_damage.y4m"

video_info = get_video_info(video_path)
# print (video_info)
total_frames = int(video_info['duration_ts'])
print('总帧数：' + str(total_frames))

#创建一个video_manager指向视频文件
video_manager = VideoManager([video_path])
stats_manager = StatsManager()
scene_manager = SceneManager(stats_manager)
#＃添加ContentDetector算法（构造函数采用阈值等检测器选项）。
scene_manager.add_detector(ContentDetector())
base_timecode = video_manager.get_base_timecode()

try:
    frames_num = total_frames
    # 设置缩减系数以提高处理速度。
    video_manager.set_downscale_factor()
    # 启动 video_manager.
    video_manager.start()
    # 在video_manager上执行场景检测。
    scene_manager.detect_scenes(frame_source=video_manager)

    # 获取检测到的场景列表。
    scene_list = scene_manager.get_scene_list(base_timecode)
    #与FrameTimecodes一样，如果是，则可以对scene_list中的每个场景进行排序
    #场景列表变为未排序。
    print('List of scenes obtained:')
    # print(scene_list)
    #如果scene_list不为空，整理结果列表，否则，视频为单场景
    re_scene_list = []
    if scene_list:
        for i, scene in enumerate(scene_list):
            # print('%d , %d' % (scene[0].get_frames(), scene[1].get_frames()))
            re_scene = (scene[0].get_frames(), scene[1].get_frames()-1)
            re_scene_list.append(re_scene)
    else:
        re_scene = (0, frames_num-1)
        re_scene_list.append(re_scene)
    #输出切分场景的列表结果
    print(re_scene_list)
finally:
    video_manager.release()


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
