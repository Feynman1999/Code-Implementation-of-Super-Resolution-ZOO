import os
import time

# add parser

def generate_test_sh_for_one_algorithm(model, name, load_auto_flag=False):
    """

    :param model:
    :param name:
    :param load_auto_flag:
    :return:
    """
    datasetnames = ["Set5", ]
    template = "python test.py --dataroot {}  --name {} --model {} --load_epoch {} --ensemble {} --only_Y {}"
    load_epochs = [3750, 5000, 6000, 8000, 10000, 12000, 14000, 16000]

    if load_auto_flag:
        # with find with checkpoints/name/epoch_xxxx_xxxxxxxxx.pth
        load_epochs = []
        pass
    command = ""
    for datasetname in datasetnames:
        for load_epoch in load_epochs:
            for emsemble_flag in ("True", "False"):
                for only_Y_flag in ("True", "False"):
                    command += template.format(os.path.join('./datasets', datasetname), name, model, load_epoch, emsemble_flag, only_Y_flag)
                    command += "\n"

    # print(command)
    with open("./test.sh", "w+") as f:
        now = time.strftime("%c")
        content = '# ================ test shell ( {} ) ================\n'.format(now)
        content += command
        f.write(content)


if __name__ == '__main__':
    dct={}
    dct["dbpn"] = []
    dct["dbpn"].append("DIV2K_dbpn")  # key -> value:      model -> [checkpoints_name_list]
    for model in sorted(dct.keys()):
        for checkpoints_name in dct[model]:
            generate_test_sh_for_one_algorithm(model, checkpoints_name)