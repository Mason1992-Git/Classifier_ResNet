import torch
from res18_net_softmax import ResNet18_Block,ResNet18_src

device = "cuda:0" if torch.cuda.is_available() else "cpu"

src_net = ResNet18_src().to(device)
mod_net = ResNet18_Block().to(device)

print(src_net)

print("====================")

print(mod_net)