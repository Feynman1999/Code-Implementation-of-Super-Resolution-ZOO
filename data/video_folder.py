"""A modified video folder class

this class can load videos from both current directory and its subdirectories.
"""

import os
import cv2
from PIL import Image

VIDEO_EXTENSIONS = [
    '.mp4', '.MP4', '.avi', '.AVI',
    '.rmvb', '.RMVB', '.flv', '.FLV',
    '.mkv', '.MKV',
]


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def make_videos_dataset(dir, max_dataset_size=float("inf")):
    videos = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_video_file(fname):
                path = os.path.join(root, fname)
                videos.append(path)
    return videos[:min(max_dataset_size, len(videos))]


def read_video(videopath):
    """

    :param videopath: the path to the video
    :return: PIL.image frames list
    """
    cap = cv2.VideoCapture(videopath)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(25)
        else:
            break
    cap.release()  # close vapture

    for i, img in enumerate(frames):
        frames[i] = Image.fromarray(img)

    return frames
    # cv2.destroyAllWindows()
