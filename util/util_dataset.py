import ntpath

from tqdm import tqdm

from data.image_folder import make_images_dataset
from .util import *


def images2video(filepath, fps=12, Suffix = '.avi'):
    imagepathlist = make_images_dataset(filepath)
    framelist = []
    for imgpath in imagepathlist:
        framelist.append(cv2.imread(imgpath)[..., ::-1])
    dn = os.path.dirname(filepath)
    name = os.path.split(filepath)[-1] + Suffix
    save_video(framelist, os.path.join(dn, name), fps=fps)


def dataset_images2video(filepath, fps=12):
    for home, dirs, files in sorted(os.walk(filepath)):
        for dir_ in dirs:
            dir_ = os.path.join(home, dir_)
            print(dir_)
            images2video(dir_, fps=fps)

        # print("#######file list#######")
        # for filename in files:
        #     print(filename)
        #     fullname = os.path.join(home, filename)
        #     print(fullname)
        # print("#######file list#######")


def dataset_HR2AB(HRpath, path2datasets, datasetname, phase="train", factor = 4):
    """
    for bitahub:
        HRpath = /data/bitahub/DIV2K/DIV2K_train_HR
        path2datasets = /data/bitahub
        datasetname = DIV2K


    :param HRpath: the path to HR images
    :param datasetname: datasetname, will create dir in datasets
    :param SR_factor: down-sample factor
    :param phase:  train or test
    :return:
    """
    imagepath_list = make_images_dataset(HRpath)
    Apath = os.path.join(path2datasets, datasetname, phase, "A")
    mkdir(Apath)
    Bpath = os.path.join(path2datasets, datasetname, phase, "B")
    mkdir(Bpath)
    for i in tqdm(range(len(imagepath_list))):
        img = Image.open(imagepath_list[i])
        imgname = os.path.basename(imagepath_list[i])
        img.save(os.path.join(Bpath, imgname))
        save_image(np.array(img), os.path.join(Apath, imgname), factor=factor, inverse=True)


def get_file_name(path):
    '''
        datasets/div2k/train/A/0001.jpg  ->  0001
    :param path: the path
    :return: the name
    '''
    short_path = ntpath.basename(path)  # get file name, it is name from domain A
    name = os.path.splitext(short_path)[0]  # Separating file name from extensions
    return name