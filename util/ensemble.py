

def ensemble(LR):
    """

    :param LR:
    :return: ensembled_LR_list
    """

    LR_90 = LR.transpose(-2, -1).flip(-2)  # rotate 90
    LR_180 = LR.flip((-2, -1))  # rotate 180
    LR_270 = LR.transpose(-2, -1).flip(-1)  # rotate 270
    LR_f = LR.flip(-1)
    LR_90f = LR_90.flip(-1)
    LR_180f = LR_180.flip(-1)
    LR_270f = LR_270.flip(-1)
    LR_list = [LR, LR_90, LR_180, LR_270, LR_f, LR_90f, LR_180f, LR_270f]

    return LR_list


def ensemble_inverse(HR_list):
    """

    :param HR_list:
    :return: inversed HR_list
    """
    HR_list[1] = HR_list[1].transpose(-2, -1).flip(-1)  # rotate 270
    HR_list[2] = HR_list[2].flip((-2, -1))  # rotate 180
    HR_list[3] = HR_list[3].transpose(-2, -1).flip(-2)  # rotate 90
    HR_list[4] = HR_list[4].flip(-1)
    HR_list[5] = HR_list[5].flip(-1).transpose(-2, -1).flip(-1)  # flip and 270
    HR_list[6] = HR_list[6].flip(-2)
    HR_list[7] = HR_list[7].flip(-1).transpose(-2, -1).flip(-2)  # flip and 90
    return HR_list


