import os
import time
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# 导入网络
from res18_net_softmax import ResNet18_Block,ResNet18_src
# 导入dataset类
from face_dataset_2 import CotterDataset
import torch.nn.functional as F
from focal_loss import CE_Focal_loss

#不使用cudnn加速
# torch.backends.cudnn.enabled = False
# 训练数据路径和验证数据路径
train_data_path = r"G:\Nest_Classifier\classifier\train_data"
test_data_path = r"G:\Nest_Classifier\classifier\test_data"
# eval_data_path = r"M:\DRIVE\test"
weights_save_path = r"weights_train/00.pt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
def train_eval_process(batch_size=16, epochs=100,accumulation_steps = 1):
    # 实例化网络
    # net = ResNet18_Block().to(device)
    net = ResNet18_src().to(device)

    if os.path.exists(weights_save_path):
        net.load_state_dict(torch.load(weights_save_path))
        print("下载权重完成！")

    # 实例化trian_data
    train_dataset = CotterDataset(train_data_path)
    # print("train_dataset = ",type(train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # 实例化eval_data
    eval_dataset = CotterDataset(test_data_path, is_train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, drop_last=True)

    # 实例化优化器和损失函数
    #Adam优化器
    optimizer = optim.Adam(net.parameters())
    #二值交叉熵损失函数
    # loss_func = nn.CrossEntropyLoss()
    loss_func = CE_Focal_loss(class_num=3)

    summary = SummaryWriter("logs")

    for epoch in range(epochs):
        # 训练过程
        train_start_time = time.time()
        train_total_loss = 0
        train_total_accuracy = 0
        eval_total_accuracy = 0
        net.train()
        for index, (image_data, label_data) in enumerate(train_dataloader):
            image_data, label_data = image_data.to(device), torch.squeeze(label_data.to(device))
            # print(image_data.shape)
            # print(label_data.shape)

            # 网络预测
            train_predict = net(image_data)
            # print("label_data>>:", label_data)
            # print("label_data_shape>>>:",label_data.shape)
            # print("train_predict_shape>>>:", train_predict.shape)
            # print("label_data>>>:",label_data)
            # train_predict = torch.squeeze(torch.squeeze(train_predict,dim=2),dim=2)
            # print("train_predict_shape1>>>:", train_predict.shape)

            #计算损失
            loss = loss_func(train_predict, label_data.long()) / accumulation_steps

            # 三部曲
            optimizer.zero_grad()
            if (epoch +1) % accumulation_steps == 0:
                loss.backward()
                optimizer.step()

            # print("train_predict>>>:", train_predict)
            predict_ID = torch.argmax(F.softmax(train_predict),dim=1).cpu()
            # print("predict>>>:",predict_ID)
            # b_predict = (F.sigmoid(train_predict.cpu()) > 0.6).int()
            train_total_accuracy += (torch.sum(torch.eq(predict_ID, label_data.cpu()).int()) / batch_size)

            # print("b_predict>>>:",b_predict)
            # train_total_accuracy += (torch.sum(torch.eq(b_predict, label_data.cpu()).int()) / batch_size)
            # print("train_total_acc>>>:",train_total_accuracy)
            # train_total_accuracy += torch.sum((torch.eq(b_predict, label_data.cpu())).int()) / (batch_size * 5)
            train_total_loss += loss.detach().item()
        train_avg_accuracy = train_total_accuracy / len(train_dataloader)
        train_avg_loss = train_total_loss / len(train_dataloader)
        net.eval()
        for index_test, (image_data_test, label_data_test) in enumerate(eval_dataloader):
            image_data_test, label_data_test = image_data_test.to(device), torch.squeeze(label_data_test.to(device))

            # 网络预测
            eval_predict = net(image_data_test)
            # eval_predict = torch.squeeze(torch.squeeze(eval_predict,dim=2),dim=2)
            predict_ID_test = torch.argmax(F.softmax(eval_predict), dim=1).cpu()
            eval_total_accuracy += (torch.sum(torch.eq(predict_ID_test, label_data_test.cpu()).int()) / batch_size)
        eval_avg_accuracy = eval_total_accuracy / len(eval_dataloader)
        print(f"epoch:{epoch} train_loss:{train_avg_loss} train_accuracy:{train_avg_accuracy} eval_accuracy:{eval_avg_accuracy}"
              f"time:{time.time() - train_start_time}")
        summary.add_scalars("loss_and_accuracy", {"avg_loss": train_avg_loss,"avg_acc": train_avg_accuracy,"eval_acc": eval_avg_accuracy}, epoch)
        if epoch % 1 ==0:
            torch.save(net.state_dict(), fr"weights_train\{epoch}.pt", _use_new_zipfile_serialization=False)
if __name__ == '__main__':
    train_eval_process(batch_size=32, epochs=200,accumulation_steps = 1)
