import os
import torch
from PIL import ImageFont,Image,ImageDraw
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import torch.onnx
from torch import jit
# 导入网络
from res18_net_softmax import ResNet18_Block,ResNet18_src
# 导入dataset类
from face_dataset_2 import CotterDataset

# print(torch.__version__)
# exit()
if __name__ == '__main__':

    weight_path = r"weights_train\30.pt"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    # net = ResNet18_Block().to(device)
    net = ResNet18_src().to(device)
    net.eval()
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("load weight successfully!")
    batch_size = 1
    inputs = torch.randn(1,1,224,224)
    torch_model = jit.trace(net,inputs.to(device))
    torch_model.save("classifier_nut_30_GPU.pt")