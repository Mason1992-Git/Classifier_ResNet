import os
from shutil import copyfile

path =r"E:\YL_Project\Projects\XiAn\diaoxian_zailiuhuan_queshi\classifier_model\results_1\cls1"

path1 = r"E:\YL_Project\Projects\XiAn\diaoxian_zailiuhuan_queshi\classifier_model\halcon_test"

save_path=r"E:\YL_Project\Projects\XiAn\diaoxian_zailiuhuan_queshi\classifier_model\results_1\src"


if __name__ == '__main__':
    # 删除
    list_1 = []
    for imgname1 in os.listdir(path):
        list_1.append(imgname1)
    print(len(list_1))
    for imgname in os.listdir(path1):
        if imgname in list_1:
            os.remove(os.path.join(path1,imgname))
    #

    # 选择
    # list_1 = []
    # for imgname1 in os.listdir(path):
    #     imgname1 = imgname1[5:]
    #     list_1.append(imgname1)
    # print(len(list_1))
    # for imgname in os.listdir(path1):
    #     if imgname in list_1:
    #         copyfile(os.path.join(path1,imgname),os.path.join(save_path,imgname))