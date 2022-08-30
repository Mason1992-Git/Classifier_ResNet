import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------Mish activation function------------------------
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


# -----------------------RESBLOCK(helper block)--------------------------
class ResBlock(nn.Module):
    '''
    ResBlock component in ResNet.
    '''

    def __init__(self, in_channels=64, mid_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1),
                 is_downsampling=False):
        super().__init__()
        self.is_downsampling = is_downsampling

        # the downsampling layer to halve the the feature map size
        self.down_sample_x = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2))
        self.down_sample = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=(2, 2), padding=padding)

        # conv layer when downsampling is not needed
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # the residual layer to have the out_channel size
        self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        layers = [nn.BatchNorm2d(mid_channels),
                  Mish(),
                  nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                  nn.BatchNorm2d(out_channels)]

        # sequential layers: with and without downsampling
        self.layers = nn.Sequential(self.conv, *layers)
        self.layers_dsample = nn.Sequential(self.down_sample, *layers)

        self.mish = Mish()

    def forward(self, x):

        if self.is_downsampling:
            residule = self.down_sample_x(x)
            x = self.layers_dsample(x)
            out = self.mish(x + residule)

        else:
            residule = self.residual_layer(x)
            x = self.layers(x)
            out = self.mish(x + residule)

        return out


# --------------------------------CONV 1-----------------------------------------
class Conv_Pool(nn.Module):
    '''
    Input layer of ResNet18. (everything before the ResBlock starts)
    (conv1 in the paper.)
    '''

    def __init__(self, in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        )

    def forward(self, x):
        return self.layer(x)


class ResNet18_Block(nn.Module):
    '''
    ResNet18
    '''

    def __init__(self):
        super().__init__()

        self.conv1 = Conv_Pool(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # print("")

        self.conv2_x1 = ResBlock(64, 64, 64, (3, 3))
        self.conv2_x2 = ResBlock(64, 64, 64, (3, 3))

        self.conv3_x1 = ResBlock(64, 128, 128, (3, 3), is_downsampling=True)
        self.conv3_x2 = ResBlock(128, 128, 128, (3, 3))

        self.conv4_x1 = ResBlock(128, 64, 64, (3, 3), is_downsampling=True)
        self.conv4_x2 = ResBlock(64, 64, 64, (3, 3))

        self.conv5_x1 = ResBlock(64, 64, 32, (3, 3), is_downsampling=True)
        self.conv5_x3 = ResBlock(32, 2, 2, (3, 3))
        self.adp_pool = nn.AdaptiveAvgPool2d((1, 1))


        self.liner_layer = nn.Linear(18,2)

    def forward(self, x):
        # input layer: conv1
        x = self.conv1(x)
        # print("CONV1_SHAPE>>>:",x.shape)
        # conv2_x
        x = self.conv2_x1(x)
        # print("CONV2_SHAPE>>>:", x.shape)
        x = self.conv2_x2(x)
        # print("CONV2_SHAPE>>>:", x.shape)
        # conv3_x (with downsampling)
        x = self.conv3_x1(x)
        # print("CONV3_SHAPE>>>:", x.shape)
        x = self.conv3_x2(x)
        # print("CONV3_SHAPE>>>:", x.shape)
        # conv4_x (with downsampling)
        x = self.conv4_x1(x)
        # print("CONV4_SHAPE>>>:", x.shape)
        x = self.conv4_x2(x)
        # print("CONV4_SHAPE>>>:", x.shape)
        # conv5_x (with downsampling)
        x = self.conv5_x1(x)
        # print("CONV5_SHAPE>>>:", x.shape)
        x = self.conv5_x3(x)
        # print("CONV5_SHAPE>>>:", x.shape)
        x = self.adp_pool(x)
        # print("CONV6_SHAPE>>>:", x.shape)
        # x = x.reshape(x.size(0),-1)
        # x = self.liner_layer(x)


        return x

def ResNet18_src():
    net = models.resnet18()
    net.conv1 = torch.nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    net.fc = torch.nn.Linear(in_features=512,out_features=2,bias=True)
    return net
if __name__ == '__main__':
    res_block = ResNet18_Block()
    inputs = torch.randn(size=(1, 1, 416, 416))
    # out = res_block(inputs)
    net = ResNet18_src()
    print(net)
    out = net(inputs)
    print(out)
    print(out.shape)
