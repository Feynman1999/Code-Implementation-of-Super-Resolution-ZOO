"""This package includes a miscellaneous collection of useful helper functions."""
import os

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


def remove_pad_for_tensor(tensor, HR_GT_h_w, factor, LR_flag=True):
    assert len(tensor.shape) == 4
    _, _, now_h, now_w = tensor.shape
    des_h, des_w = HR_GT_h_w
    assert des_h % factor == 0 and des_w % factor == 0
    if LR_flag:
        des_h = des_h // factor
        des_w = des_w // factor
    assert now_h >= des_h and now_w >= des_w

    delta_h = now_h - des_h
    delta_w = now_w - des_w

    if LR_flag:
        start_h = delta_h // 2
        start_w = delta_w // 2
        return tensor[..., start_h: start_h + des_h, start_w: start_w + des_w]
    else:
        assert delta_w % factor == 0 and delta_h % factor == 0
        delta_h = delta_h // factor
        delta_w = delta_w // factor
        start_h = delta_h // 2
        start_w = delta_w // 2
        return tensor[..., start_h*factor: start_h*factor + des_h, start_w*factor: start_w*factor + des_w]
