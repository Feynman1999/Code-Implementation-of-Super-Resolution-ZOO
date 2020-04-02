import ntpath

from tqdm import tqdm

from data.image_folder import make_images_dataset
from data.video_folder import make_videos_dataset, read_video
from .util import *


def images2video(filepath, fps=2, suffix='.avi'):
    """
    give a dir which contains images , trans to video
    :param filepath: the path to images dir
    :param fps: fps for video
    :param suffix:
    :return:
    """
    imagepathlist = make_images_dataset(filepath)
    framelist = []
    for imgpath in imagepathlist:
        framelist.append(cv2.imread(imgpath)[..., ::-1])
    dn = os.path.dirname(filepath)
    name = os.path.split(filepath)[-1] + suffix
    save_video(framelist, os.path.join(dn, name), fps=fps)


def dataset_images2video(filepath, fps=2):
    for home, dirs, files in sorted(os.walk(filepath)):
        for dir_ in dirs:
            dir_ = os.path.join(home, dir_)
            images2video(dir_, fps=fps)

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


def vimeo90K_dataset_onlyHR2AB(dataset_path, ABpath, phase="train", factor=4):
    """
    pre-deal make it suitable for this project for specific dataset: vimeo90K
    link:  http://toflow.csail.mit.edu/

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
            new_path = two_level_path.replace('/', '_')
            if os.path.exists(os.path.join(Apath, new_path)) and os.path.exists(os.path.join(Bpath, new_path)):
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


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


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