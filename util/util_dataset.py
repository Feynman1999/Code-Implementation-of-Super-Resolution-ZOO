import ntpath

from tqdm import tqdm

from data.image_folder import make_images_dataset
from data.video_folder import make_videos_dataset, read_video, get_video_info
import shutil
import os
import cv2
import pickle
import threading

import ffmpeg
from PIL import Image
from util.util import save_image, save_video
from . import mkdir


class myThread(threading.Thread):
    def __init__(self, list, size, path):
        threading.Thread.__init__(self)
        self.list = list
        self.size = size
        self.path = path

    def run(self):
        for path in self.list:
            imgname = get_file_name(path)
            image_pre_crop(path2img=path, crop_size=self.size, path2placeblocks=os.path.join(self.path, imgname))


def image_pre_crop(path2img, crop_size, path2placeblocks):
    img = Image.open(path2img).convert('RGB')
    w, h = img.size
    w_list = list(range(0, w+1, crop_size))
    h_list = list(range(0, h+1, crop_size))
    id = 0
    for i in range(len(w_list)-1):
        for j in range(len(h_list)-1):
            box = (w_list[i], h_list[j], w_list[i+1], h_list[j+1])
            croped_img = img.crop(box)
            save_image(croped_img, os.path.join(path2placeblocks, "{}.png".format(id)))
            id += 1


def video_pre_crop(path2video, crop_size):
    """

    :param path2video:  e.g. ./datasets/mgtv/train/A/damage_01
    :param crop_size: e.g. 256
    :return:
    """
    imgpathlist = make_images_dataset(path2video)
    videoname = os.path.split(path2video)[-1]
    path2place_result = os.path.dirname(path2video)
    domain = os.path.split(path2place_result)[-1]
    path2place_result = os.path.dirname(path2place_result)
    path2place_result = os.path.join(path2place_result, domain + "_cropsize_" + str(crop_size), videoname)
    # for path in imgpathlist:
    #     imgname = get_file_name(path)
    #     image_pre_crop(path2img=path, crop_size=crop_size, path2placeblocks=os.path.join(path2place_result, imgname))

    # multi thread
    thread_nums = 8
    thread_range_len = len(imgpathlist) // thread_nums
    thread_range_list = list(range(0, len(imgpathlist)+1, thread_range_len))
    if thread_range_list[-1] != len(imgpathlist):
        thread_range_list[-1] = len(imgpathlist)
    thread_list = []
    for i in range(thread_nums):
        thread_list.append(myThread(imgpathlist[thread_range_list[i]:thread_range_list[i+1]], crop_size, path2place_result))
        thread_list[-1].start()
    for item in thread_list:
        item.join()


def videodataset_pre_crop(path2AB, crop_size=256):
    """

    :param path2AB: e.g. ./datasets/mgtv/train
    :param crop_size:
    :return:
    """
    Apath = os.path.join(path2AB, "A")
    Bpath = os.path.join(path2AB, "B")
    for videoname in os.listdir(Apath):
        print("now dealing A: {}".format(videoname))
        video_pre_crop(os.path.join(Apath, videoname), crop_size=crop_size)
    for videoname in os.listdir(Bpath):
        print("now dealing B: {}".format(videoname))
        video_pre_crop(os.path.join(Bpath, videoname), crop_size=crop_size)


def videodataset_scenedetect(dirpath):
    from scenedetect.video_manager import VideoManager
    from scenedetect.scene_manager import SceneManager
    from scenedetect.stats_manager import StatsManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.detectors import ThresholdDetector
    result_list = []
    for video_name in sorted(os.listdir(dirpath)):
        video_path = os.path.join(dirpath, video_name)
        print(video_path)
        video_info = get_video_info(video_path)
        # print (video_info)
        total_frames = int(video_info['duration_ts'])
        print('总帧数：' + str(total_frames))
        # 创建一个video_manager指向视频文件
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        # ＃添加ContentDetector算法（构造函数采用阈值等检测器选项）
        scene_manager.add_detector(ContentDetector(threshold=20))
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
            # 与FrameTimecodes一样，如果是，则可以对scene_list中的每个场景进行排序
            # 场景列表变为未排序。
            # print('List of scenes obtained:')
            # print(scene_list)
            # 如果scene_list不为空，整理结果列表，否则，视频为单场景
            re_scene_list = []
            if scene_list:
                for i, scene in enumerate(scene_list):
                    # print('%d , %d' % (scene[0].get_frames(), scene[1].get_frames()))
                    re_scene = (scene[0].get_frames(), scene[1].get_frames() - 1)
                    re_scene_list.append(re_scene)
            else:
                re_scene = (0, frames_num - 1)
                re_scene_list.append(re_scene)
            # 输出切分场景的列表结果
            print("\n")
            print(re_scene_list)
            print("\n")
            # for item in re_scene_list:
            #     assert item[1] - item[0] + 1 >= 3
            result_list.append(re_scene_list)
        finally:
            video_manager.release()
    with open(os.path.join(dirpath, 'scene.pickle'), 'wb') as f:
        pickle.dump(result_list, f)


def images2video(dirpath, fps=2, suffix='.avi', must_have="HR_G"):
    """
    give a dir which contains images , trans to video
    :param dirpath: the path to images dir
    :param fps: fps for video
    :param suffix:
    :return:
    """
    imagepathlist = sorted(make_images_dataset(dirpath))
    dn = os.path.dirname(dirpath)
    name = os.path.split(dirpath)[-1] + "_" + must_have + suffix

    # framelist = []
    # id = 0
    # for imgpath in imagepathlist:
    #     if must_have in get_file_name(imgpath):
    #         print("{} frame: {}".format(name, id))
    #         id += 1
    #         framelist.append(cv2.imread(imgpath)[..., ::-1])
    # save_video(framelist, os.path.join(dn, name), fps=fps)

    flag = False
    for item in imagepathlist:
        if must_have in get_file_name(item):
            flag = True
            break
    if not flag:
        return

    (
        ffmpeg
        .input(os.path.join(dirpath, "*{}*.png".format(must_have)), pattern_type='glob', framerate=fps)
        .output(os.path.join(dn, name), pix_fmt="yuv420p")
        .run()
    )


def dataset_images2video(datasetpath, fps=2, must_have=("HR_G", "HR_Bicubic", "LR"), suffix=".avi"):
    for home, dirs, files in sorted(os.walk(datasetpath)):
        for dir_ in dirs:
            dir_ = os.path.join(home, dir_)
            for item in must_have:
                print("now dealing {} in {}".format(item, dir_))
                images2video(dir_, fps=fps, suffix=suffix, must_have=item)

        # print("#######file list#######")
        # for filename in files:
        #     print(filename)
        #     fullname = os.path.join(home, filename)
        #     print(fullname)
        # print("#######file list#######")


def image_dataset_onlyHR2AB(HRpath, ABpath, factor=4, phase="train"):
    """
    give only HR images , transfer to domain A(LR) and domain B(HR), write to Apath and Bpath
    :param HRpath: the path to HR images
    :param ABpath: the path to place AB
    :param factor: factor
    :return:
    """
    Apath = os.path.join(ABpath, phase, "A")
    Bpath = os.path.join(ABpath, phase, "B")
    assert (not os.path.exists(Apath)) and (not os.path.exists(Bpath)), "{} or {} already exist, if you want to " \
                                                                        "generate new AB, please delete them " \
                                                                        "first".format(Apath, Bpath)
    assert check_whether_last_dir(HRpath), 'when only HR for images, HRpath should be dir and contains only image files'
    imagepath_list = make_images_dataset(HRpath)
    mkdir(Apath)
    mkdir(Bpath)
    for i in tqdm(range(len(imagepath_list))):
        img = Image.open(imagepath_list[i])
        imgname = os.path.basename(imagepath_list[i])
        img.save(os.path.join(Bpath, imgname))
        save_image(img, os.path.join(Apath, imgname), factor=factor, inverse=True)


def video_dataset_onlyHR2AB(HRpath, ABpath, factor=4, phase="train"):
    """
    give only HR videos , transfer to domain A(LR) and domain B(HR), write to Apath and Bpath (second kind style)
    :param HRpath: the path to HR videos can be single video e.g. HRpath / 0001.mp4  (first kind style)
    or  multi images  e.g.  HRpath / 0001 / 1.png 2.png ... x.png  (second kind style)
    :param ABpath: the path to place AB
    :param factor: factor
    :return:
    """
    Apath = os.path.join(ABpath, phase, "A")
    Bpath = os.path.join(ABpath, phase, "B")
    assert (not os.path.exists(Apath)) and (not os.path.exists(Bpath)), "{} or {} already exist, if you want to " \
                                                                        "generate new AB, please delete them " \
                                                                        "first".format(Apath, Bpath)
    mkdir(Apath)
    mkdir(Bpath)
    videopath_list = make_videos_dataset(HRpath)  # get all the video files path
    if len(videopath_list) > 0:  # first kind style
        assert check_whether_last_dir(HRpath), 'when only HR for video files, HRpath should be dir and contains only video files'
        for i in tqdm(range(len(videopath_list))):
            vid = read_video(videopath_list[i], PIL_Image_flag=False)  # ndarray list
            vidname = get_file_name(videopath_list[i])
            mkdir(os.path.join(Bpath, vidname))
            mkdir(os.path.join(Apath, vidname))
            for ith, img in enumerate(vid):
                imgpathB = os.path.join(Bpath, vidname, "frame_{:05d}".format(ith) + ".png")
                save_image(img, imgpathB)
                imgpathA = os.path.join(Apath, vidname, "frame_{:05d}".format(ith) + ".png")
                save_image(img, imgpathA, factor=factor, inverse=True)

    else:  # second kind style
        dir_list = sorted(os.listdir(HRpath))
        for cur_file in tqdm(dir_list):
            path = os.path.join(HRpath, cur_file)  # get absolute path
            if os.path.isdir(path):  # if is dir, think it as a set of images to form a video
                imagepath_list = make_images_dataset(path)
                mkdir(os.path.join(Bpath, cur_file))
                mkdir(os.path.join(Apath, cur_file))
                for ith, img_path in enumerate(imagepath_list):
                    img = Image.open(img_path)
                    imgpathB = os.path.join(Bpath, cur_file, "frame_{:05d}".format(ith) + ".png")
                    save_image(img, imgpathB)
                    imgpathA = os.path.join(Apath, cur_file, "frame_{:05d}".format(ith) + ".png")
                    save_image(img, imgpathA, factor=factor, inverse=True)
            else:
                print("please do not put any file in the {}, e.g. {}".format(HRpath, cur_file))
                raise NotImplementedError


def video_dataset_HRLR2AB(HRpath, LRpath, ABpath, phase = "train"):
    """
    give HR & LR videos, transfer to domain A(LR) and domain B(HR), write to Apath and Bpath (second kind style)
    :param HRpath: the path to HR videos
    :param LRpath: the path to LR videos
    :param ABpath: the path to place AB
    :return:
    """
    Apath = os.path.join(ABpath, phase, "A")
    Bpath = os.path.join(ABpath, phase, "B")
    assert (not os.path.exists(Apath)) and (not os.path.exists(Bpath)), "{} or {} already exist, if you want to " \
                                                                        "generate new AB, please delete them " \
                                                                        "first".format(Apath, Bpath)
    # first deal HR
    print("dealing with HR videos!")
    assert check_whether_last_dir(HRpath), 'HRpath should be dir and contains only video files'
    videopath_list = make_videos_dataset(HRpath)
    for i in tqdm(range(len(videopath_list))):
        vid = read_video(videopath_list[i], PIL_Image_flag=False)
        vidname = get_file_name(videopath_list[i])
        mkdir(os.path.join(Bpath, vidname))
        for ith, img in enumerate(vid):
            imgpathB = os.path.join(Bpath, vidname, "frame_{:05d}".format(ith) + ".png")
            save_image(img, imgpathB)

    # second deal LR
    print("dealing with LR videos!")
    assert check_whether_last_dir(LRpath), 'HRpath should be dir and contains only video files'
    videopath_list = make_videos_dataset(LRpath)
    for i in tqdm(range(len(videopath_list))):
        vid = read_video(videopath_list[i], PIL_Image_flag=False)
        vidname = get_file_name(videopath_list[i])
        mkdir(os.path.join(Apath, vidname))
        for ith, img in enumerate(vid):
            imgpathA = os.path.join(Apath, vidname, "frame_{:05d}".format(ith) + ".png")
            save_image(img, imgpathA)


def videos_to_images(videos_path, path2place, phase="train", AorB = "A", max_frames=25):
    """
    give videos, transfer to domain A or B (images style)
    :param videos_path:
    :param ABpath:
    :param phase:
    :return:
    """
    AorBpath = os.path.join(path2place, phase, AorB)
    # assert (not os.path.exists(AorBpath)), "{}already exist, if you want to " \
    #                                                                     "generate new AorB, please delete it " \
    #                                                                     "first".format(AorBpath)

    assert check_whether_last_dir(videos_path), 'HRpath should be dir and contains only video files'
    videopath_list = make_videos_dataset(videos_path)
    for i in tqdm(range(len(videopath_list))):
        # vid = read_video(videopath_list[i], max_frames=max_frames, PIL_Image_flag=False)
        # vidname = get_file_name(videopath_list[i])
        # mkdir(os.path.join(AorBpath, vidname))
        # for ith, img in enumerate(vid):
        #     imgpath = os.path.join(AorBpath, vidname, "frame_{:05d}".format(ith) + ".png")
        #     save_image(img, imgpath)
        #     if ith + 1 == max_frames:
        #         break
        vidname = get_file_name(videopath_list[i])
        mkdir(os.path.join(AorBpath, vidname))
        stream = ffmpeg.input(videopath_list[i])
        stream = ffmpeg.output(stream, os.path.join(AorBpath, vidname, "frame_" + '%05d.png'))
        ffmpeg.run(stream)


def vimeo90K_dataset_onlyHR2AB(dataset_path, ABpath, phase="train", factor=4, can_continue=False):
    """
    pre-deal make it suitable for this project for specific dataset: vimeo90K
    link:  http://toflow.csail.mit.edu/           notice!  the dir 00055/0896 only have one frame......   /(ㄒoㄒ)/~~!

    usage example:
        vimeo90K_dataset_onlyHR2AB(dataset_path="/opt/data/private/datasets/vimeo_septuplet/vimeo_septuplet",
                                   ABpath="/opt/data/private/datasets/vimeo_septuplet",
                                   phase="train",
                                   factor=4)

    :param datasetpath: the path to dataset dir, should have sep_testlist.txt and sep_trainlist.txt and sequences dir
    :param ABpath: the path to place AB
    :param phase: train or test
    :return: none
    """
    Apath = os.path.join(ABpath, phase, "A")
    Bpath = os.path.join(ABpath, phase, "B")
    # assert (not os.path.exists(Apath)) and (not os.path.exists(Bpath)), "{} or {} already exist, if you want to " \
    #                                                                     "generate new AB, please delete them " \
    #                                                                     "first".format(Apath, Bpath)
    video_dir = os.path.join(dataset_path, "sequences")
    txt_path = os.path.join(dataset_path, "sep_{}list.txt".format(phase))
    assert os.path.isdir(video_dir)
    assert os.path.isfile(txt_path)
    with open(txt_path) as f:
        for two_level_path in tqdm(f.readlines()):
            HR_dir_path = os.path.join(video_dir, two_level_path.strip())  # e.g. dataset_path/sequences/00010/0558
            if not os.path.isdir(HR_dir_path):
                print("illegal path: {} is not dir, continue!".format(HR_dir_path))
                continue
            new_path = two_level_path.strip()[0:10].replace('/', '_')
            if can_continue and os.path.exists(os.path.join(Apath, new_path)) and os.path.exists(os.path.join(Bpath, new_path)):
                print("{} already dealed, continue!".format(new_path))
                continue
            mkdir(os.path.join(Bpath, new_path))  # e.g.  Bpath/00010_0558
            mkdir(os.path.join(Apath, new_path))
            imagepath_list = make_images_dataset(HR_dir_path)
            for ith, img_path in enumerate(imagepath_list):
                img = Image.open(img_path)
                imgpathB = os.path.join(Bpath, new_path, "frame_{:05d}".format(ith) + ".png")
                save_image(img, imgpathB)
                imgpathA = os.path.join(Apath, new_path, "frame_{:05d}".format(ith) + ".png")
                save_image(img, imgpathA, factor=factor, inverse=True)


def youku_dataset_HRLR2AB():
    pass


def SPMCS_dataset_HRLR2AB(dataset_path="./datasets/SPMCS/test_set", ABpath="./datasets/SPMCS"):
    """
    you will find that the original input4 is not corresponding to the truth, lead to low psnr.
    so you should use the SPMCS_dataset_onlyHR2AB.
    :param dataset_path: e.g.                           ./datasets/SPMCS/test_set
    :param ABpath:  the path to place A,B     e.g.      ./datasets/SPMCS
    :return:
    """
    factor = 4
    phase = "test"
    Apath = os.path.join(ABpath, phase, "A")
    Bpath = os.path.join(ABpath, phase, "B")
    for video_name in tqdm(os.listdir(dataset_path)):
        Bdir = os.path.join(dataset_path, video_name, "truth")
        Adir = os.path.join(dataset_path, video_name, "input{}".format(factor))
        shutil.copytree(src=Bdir, dst=os.path.join(Bpath, video_name), symlinks=False)
        shutil.copytree(src=Adir, dst=os.path.join(Apath, video_name), symlinks=False)


def SPMCS_dataset_onlyHR2AB(dataset_path, ABpath):
    """

    :param dataset_path: e.g.                           ./datasets/SPMCS/test_set
    :param ABpath:  the path to place A,B     e.g.      ./datasets/SPMCS
    :return:
    """
    factor = 4
    phase = "test"
    Apath = os.path.join(ABpath, phase, "A")
    Bpath = os.path.join(ABpath, phase, "B")
    assert (not os.path.exists(Apath)) and (not os.path.exists(Bpath)), "{} or {} already exist, if you want to " \
                                                                        "generate new AB, please delete them " \
                                                                        "first".format(Apath, Bpath)
    for video_name in tqdm(os.listdir(dataset_path)):
        Bdir = os.path.join(dataset_path, video_name, "truth")
        HRdir = os.path.join(Bpath, video_name)
        shutil.copytree(src=Bdir, dst=HRdir, symlinks=False)
        # iteration high resolution to get low resolution
        imagepath_list = sorted(make_images_dataset(HRdir))
        mkdir(os.path.join(Apath, video_name))
        for ith, img_path in enumerate(imagepath_list):
            img = Image.open(img_path)
            imgname = get_file_name(img_path)
            imgpathA = os.path.join(Apath, video_name, imgname + ".png")
            save_image(img, imgpathA, factor=factor, inverse=True)


def get_dataset_name(dataroot):
    dataset_name = os.path.basename(dataroot)
    if dataset_name == "":  # dataroot is ./xxxxx/xxxxx/
        dataset_name = os.path.basename(dataroot[:-1])
    return dataset_name


def get_file_name(path):
    '''
        datasets/div2k/train/A/0001.jpg  ->  0001
    :param path: the path
    :return: the name
    '''
    short_path = ntpath.basename(path)  # get file name, it is name from domain A
    name = os.path.splitext(short_path)[0]  # Separating file name from extensions
    return name


def check_whether_last_dir(path):
    """
    check whether the path is the last dir(thus don't include another dir)
    :param path: path to dir
    :return:
    """
    if not os.path.exists(path):
        return False

    if not os.path.isdir(path):
        return False

    for root, dirs, files in os.walk(path):
        if len(dirs) > 0:
            return False

    return True


# def get_dataset_dir(dataroot):
#     """
#     :param dataroot:
#     :return:
#     """
#     dataset_dir = os.path.dirname(dataroot)
#     if dataroot.endswith("/") or dataroot.endswith("\\"):  # dataroot is ./xxxx/xxxxx/
#         dataset_dir = os.path.dirname(dataset_dir)  # again
#         dataroot = dataroot[:-1]
#
#     if os.path.basename(dataset_dir).lower() in ('datasets', 'dataset'):  # dataroot is ./xxx/datasets/Set5 for only HR
#         return dataroot
#     return dataset_dir  # dataroot is ./xxxxx/datasets/Set5/HR for only HR