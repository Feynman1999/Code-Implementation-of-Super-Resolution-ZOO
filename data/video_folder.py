"""A modified video folder class

this class can load videos from both current directory and its subdirectories.
"""

import os

VIDEO_EXTENSIONS = [
    '.mp4', '.MP4', '.avi', '.AVI',
    '.rmvb', '.RMVB', '.flv', '.FLV',
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
