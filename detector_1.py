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
# 导入网络
from res18_net_softmax import ResNet18_Block
# 导入dataset类
from face_dataset_2 import CotterDataset


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = ResNet18_Block().to(device)
    net.eval()
    weight_path = r"weight_train\300.pt"
    # img = Image.open(r"images\99.jpg")
    net.load_state_dict(torch.load(weight_path))
    print("下载权重完成！")
    img = cv2.imread(r"F:\00-TYE_FILE\4C_Project\GeKu\shuangeranzi\classifier_model\test_data\1\defect (16).jpg",0)
    # img = cv2.resize(img,(64,64))
    black_img = np.zeros((64, 64), dtype=np.uint8)
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
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_use = transform(data)
    img_use = torch.unsqueeze(img_use,dim=0)
    # print(img_data.shape)

    batch_size = 1
    # print(inputs.shape)
    # torch_model = jit.trace(net,inputs.to(device))
    # torch_model.save("classifier_model_cpu.pt")
    rst = net(img_use.to(device))
    print(rst)
    eval_predict = torch.squeeze(torch.squeeze(torch.squeeze(rst, dim=2), dim=2),dim=0)
    print(eval_predict)
    out = F.softmax(eval_predict)
    print(out)

