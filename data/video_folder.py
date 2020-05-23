"""A modified video folder class

this class can load videos from both current directory and its subdirectories.
"""

import os
import cv2
from PIL import Image
import y4m
import numpy as np

VIDEO_EXTENSIONS = [
    '.mp4', '.MP4', '.avi', '.AVI',
    '.rmvb', '.RMVB', '.flv', '.FLV',
    '.mkv', '.MKV', '.y4m', '.mpeg'
]


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def make_videos_dataset(dir, max_dataset_size=float("inf")):
    """
        only support for xx/xxx.mp4 style , return a list of path to videos
    """
    videos = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_video_file(fname):
                path = os.path.join(root, fname)
                videos.append(path)
    return videos[:min(max_dataset_size, len(videos))]


y4m_frames = []


def process_y4m_frame(frame):
    global y4m_frames
    assert frame.headers['C'] == '420mpeg2' and frame.headers['I'] == 'p', 'Encoding method not supported temporarily'
    rows = frame.headers['H']
    cols = frame.headers['W']
    rc = rows * cols
    assert rows % 2 == 0 and cols % 2 == 0, 'otherwise maybe wrong?'
    img_y = np.reshape(np.fromstring(frame.buffer[:rc], np.uint8), (rows, cols))
    img_u = np.reshape(np.fromstring(frame.buffer[rc:rc//4 + rc], np.uint8), (rows // 2, cols // 2))
    img_v = np.reshape(np.fromstring(frame.buffer[rc//4+rc:], np.uint8), (rows // 2, cols // 2))
    enlarge_u = cv2.resize(img_u, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
    enlarge_v = cv2.resize(img_v, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
    dst = cv2.merge([img_y, enlarge_u, enlarge_v])
    bgr = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)

    y4m_frames.append(bgr[..., ::-1])


def read_video(videopath, max_frames=100000, PIL_Image_flag=True):
    """

    :param videopath: the path to the video
    :param PIL_Image_flag: whether return PIL.Image.Image list
    :return: PIL.image frames list or numpy.ndarray list
    """
    if videopath.endswith(".y4m"):
        global y4m_frames
        y4m_frames = []
        parser = y4m.Reader(process_y4m_frame, verbose=False)
        infd = open(videopath, "rb")

        with infd as f:
            while True:
                data = f.read(1024 * 1024)
                if not data:
                    break

                parser.decode(data)

                if len(y4m_frames) == max_frames:
                    break

        if PIL_Image_flag:
            return [Image.fromarray(f) for f in y4m_frames]
        else:
            return y4m_frames

    else:
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

            if len(frames) == max_frames:
                break
        cap.release()  # close vapture

        for i, bgr in enumerate(frames):
            frames[i] = bgr[..., ::-1]

        if PIL_Image_flag:
            return [Image.fromarray(f) for f in frames]
        else:
            return frames
        # cv2.destroyAllWindows()
