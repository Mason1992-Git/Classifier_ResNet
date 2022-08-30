import os

import cv2
import torch
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import torch.onnx
from torch import jit
from shutil import copyfile
# 导入网络
from res18_net_softmax import ResNet18_Block
# 导入dataset类
from face_dataset_2 import CotterDataset

# path_images = r"F:\00-TYE_FILE\4C_Project\XIAN\dengdianweixian_queshi\classifier_model\all"
path_images = r"D:\XIAN_DAIOXIAN_FL\exp3\crops\1"
save_path_0 = r"G:\Nest_Classifier\classifier\eval_results\0"
save_path_1 = r"G:\Nest_Classifier\classifier\eval_results\1"
save_path_2 = r"G:\Nest_Classifier\classifier\eval_results\5"
save_path_3 = r"G:\Nest_Classifier\classifier\eval_results\3"
save_path_4 = r"E:\YL_Project\Projects\2C_Project\Nest_Classifier\classifier\eval_results\4"
save_path_5 = r"E:\YL_Project\Projects\2C_Project\CK_2C\Pillar_Number\classifier\test_results\5"
save_path_6 = r"E:\YL_Project\Projects\2C_Project\CK_2C\Pillar_Number\classifier\test_results\6"
save_path_7 = r"E:\YL_Project\Projects\2C_Project\CK_2C\Pillar_Number\classifier\test_results\7"
save_path_8 = r"E:\YL_Project\Projects\2C_Project\CK_2C\Pillar_Number\classifier\test_results\8"
save_path_9 = r"E:\YL_Project\Projects\2C_Project\CK_2C\Pillar_Number\classifier\test_results\9"
if __name__ == '__main__':

    weight_path = r"classifier_nut_30_GPU.pt"
    # weight_path = r"weight_train_pre\classifier_kaikouxiao_pre_16.pt"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    if os.path.exists(weight_path):
        model = torch.jit.load(weight_path)
        print("load weight successfully!")
    batch_size = 1
    inputs = torch.zeros(1,1,224,224)
    all_num = 0
    acc_num = 0
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    num_5 = 0
    num_6 = 0
    num_7 = 0
    num_8 = 0
    num_9 = 0

    model.eval()
    for file in os.listdir(path_images):
        all_num += 1
        img_path = os.path.join(path_images,file)
        print(img_path)
        img = cv2.imread(img_path,0)
        black_img = np.zeros((224, 224), dtype=np.uint8)
        w_b, h_b = black_img.shape
        w, h = img.shape
        # print(w)
        # print(h)
        max_num = max(w, h)
        # print(w_b)
        ratio = w_b / max_num
        # print(ratio)
        img_data = cv2.resize(img, (int(h * ratio), int(w * ratio)))
        # print(data.shape)
        black = Image.fromarray(cv2.cvtColor(black_img, cv2.COLOR_GRAY2RGB))
        img_pil = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB))
        black.paste(img_pil, (0, 0))
        data = cv2.cvtColor(np.asarray(black), cv2.COLOR_RGB2GRAY)
        # img.show()
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img_use = transform(data)
        img_use = torch.unsqueeze(img_use,dim=0).to(device)
        # print(img_data.shape)
        # torch_model = jit.trace(net,inputs.to(device))
        # torch_model.save("classifier_model_cpu.pt")
        rst = model.forward(img_use)
        # print("原始结果：",rst)
        # eval_predict = torch.squeeze(torch.squeeze(torch.squeeze(rst, dim=2), dim=2),dim=0)
        eval_predict = rst
        # print(eval_predict)
        out = F.softmax(eval_predict).to("cpu")
        print("softmax后：",out)
        pre = torch.argmax(out,dim=1).to("cpu")
        pre_confidence = torch.squeeze(out,dim=0)[pre].detach().numpy().tolist()[0]
        print("confidence = ",pre_confidence)
        print("finally_res:",pre)
        if pre == 0:
            # print("test")
            num_0 += 1
            copyfile(img_path,os.path.join(save_path_0,str(pre_confidence)[:4]+"_"+file))
        elif pre == 1:
            # print("test")
            # acc_num += 1
            num_1 += 1
            copyfile(img_path,os.path.join(save_path_1,str(pre_confidence)[:4]+"_"+file))

        elif pre == 2:
            # print("test")
            # acc_num += 1
            num_2 += 1
            copyfile(img_path,os.path.join(save_path_2,str(pre_confidence)[:4]+"_"+file))
        elif pre == 3:
            # print("test")
            # acc_num += 1
            num_3 += 1
            copyfile(img_path,os.path.join(save_path_3,str(pre_confidence)[:4]+"_"+file))
        elif pre == 4:
            # print("test")
            # acc_num += 1
            num_4 += 1
            copyfile(img_path,os.path.join(save_path_4,str(pre_confidence)[:4]+"_"+file))
        elif pre == 5:
            # print("test")
            # acc_num += 1
            num_5 += 1
            copyfile(img_path,os.path.join(save_path_5,str(pre_confidence)[:4]+"_"+file))
        elif pre == 6:
            # print("test")
            # acc_num += 1
            num_6 += 1
            copyfile(img_path,os.path.join(save_path_6,str(pre_confidence)[:4]+"_"+file))
        elif pre == 7:
            # print("test")
            # acc_num += 1
            num_7 += 1
            copyfile(img_path,os.path.join(save_path_7,str(pre_confidence)[:4]+"_"+file))
        elif pre == 8:
            # print("test")
            # acc_num += 1
            num_8 += 1
            copyfile(img_path,os.path.join(save_path_8,str(pre_confidence)[:4]+"_"+file))
        elif pre == 9:
            # print("test")
            # acc_num += 1
            num_9 += 1
            copyfile(img_path,os.path.join(save_path_9,str(pre_confidence)[:4]+"_"+file))
    print("class-0 = :",num_0)
    print("class-1 = :", num_1)
    print("class-2 = :", num_2)
    print("class-3 = :", num_3)
    print("class-4 = :", num_4)
    print("class-5 = :", num_5)
    print("class-6 = :", num_6)
    print("class-7 = :", num_7)
    print("class-8 = :", num_8)
    print("class-9 = :", num_9)







