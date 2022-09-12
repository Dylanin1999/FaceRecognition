import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import summary
import math

class InvertedResidual(nn.Module):
    def __init__(self,inputChannel,outChannel,strides,expand_ratio):
        super(InvertedResidual, self).__init__()
        self.inputChannel = inputChannel
        self.outChannel = outChannel
        self.strides = strides
        self.hiddenlayerChannel = self.inputChannel*expand_ratio
        self.Conv2dRelu6 = nn.Sequential(
                nn.Conv2d(in_channels=self.inputChannel, out_channels=self.hiddenlayerChannel, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.hiddenlayerChannel),
                nn.ReLU6(True)
            )

        self.Conv2dLinear = nn.Sequential(
                nn.Conv2d(in_channels=self.hiddenlayerChannel, out_channels=self.outChannel, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.outChannel)

            )
        self.Dwise = nn.Sequential(
                nn.Conv2d(in_channels=self.hiddenlayerChannel,out_channels=self.hiddenlayerChannel,kernel_size=3,
                          stride=self.strides,groups=self.hiddenlayerChannel,padding=1),
                nn.BatchNorm2d(self.hiddenlayerChannel),
                nn.ReLU6(True)
            )

    def forward(self, input):
        x = self.Conv2dRelu6(input)
        x = self.Dwise(x)
        out = self.Conv2dLinear(x)
        if self.strides == 1 and self.inputChannel == self.outChannel:
            out = out+input
        return out


class MobileNetV2(nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()

        #拓展因子 输出channel 重复次数 stride
        self.BottleneckCfg = [
            [2, 64, 5, 2],
            [2, 128, 1, 2],
            [4, 128, 6, 1],
            [2, 16, 1, 1]
        ]
        #根据配置信息去先生产对应的层，然后放在list里
        self.bottleneckList = nn.ModuleList()

        for layer, i in enumerate(self.BottleneckCfg):
            for time in range(1, i[2] + 1):
                if time == 1:
                    if layer == 0:
                        self.bottleneck = InvertedResidual(i[1], i[1], i[3], i[0])
                    else:
                        self.bottleneck = InvertedResidual(self.BottleneckCfg[layer - 1][1], i[1], i[3], i[0])
                else:
                    self.bottleneck = InvertedResidual(i[1], i[1], 1, i[0])
                bottleName = '''bottneck_%d_%d'''%(layer,time)

                self.bottleneckList.append(self.bottleneck)

        self.Conv2d_1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=self.BottleneckCfg[0][1],kernel_size=3,stride=2,padding=1),
                                      nn.BatchNorm2d(self.BottleneckCfg[0][1]))


        self.Conv2d_2 = nn.Sequential(nn.Conv2d(in_channels=self.BottleneckCfg[0][1], padding=1,out_channels=self.BottleneckCfg[0][1], kernel_size=3, stride=2,groups=self.BottleneckCfg[0][1]),
                                      nn.BatchNorm2d(self.BottleneckCfg[0][1]))

    def forward(self,input):
        x = self.Conv2d_1(input)
        x = self.Conv2d_2(x)

        for layer,bottleNeck in enumerate(self.bottleneckList):
            x = bottleNeck(x)
            if layer == 0:
                AuxInput = x

        return x, AuxInput

class  AuxiliaryNet(nn.Module):
    def baseConv(self,inChannel, kernelSize, outChannel, stride, padding):
        out = nn.Conv2d(in_channels=inChannel, kernel_size=kernelSize, out_channels=outChannel, stride=stride,
                        padding=padding)
        return out
    def __init__(self):
        super(AuxiliaryNet, self).__init__()

        # inchannel ,kernel_size,outchannel,stride,padding``
        self.auxiliaryCfg = [
            [64, 3, 128, 2, 1],
            [128,3, 128, 1, 1],
            [128,3, 32, 2, 1],
            [32, 7, 128, 1, 0],
        ]

        self.AuxNetList = nn.ModuleList()
        for layer, i in enumerate(self.auxiliaryCfg):
            kernelSize = i[1]
            stride = i[3]
            padding = i[4]
            outChannels = i[2]
            inchannels = i[0]


            self.AuxLayer = self.baseConv(inChannel=inchannels, kernelSize=kernelSize, outChannel=outChannels, stride=stride, padding=padding)
            self.AuxNetList.append(self.AuxLayer)

        self.Flatten = torch.nn.Flatten()
        self.AuxNetList.append(self.Flatten)
        self.Linear = torch.nn.Linear(self.auxiliaryCfg[-1][2],32)
        self.AuxNetList.append(self.Linear)
        self.Linear = torch.nn.Linear(32, 3)
        self.AuxNetList.append(self.Linear)
    def forward(self,input):
        x = input
        for layer, AuxLayer in enumerate(self.AuxNetList):
            x = AuxLayer(x)
        return x


class WingLoss(nn.Module):

    def __init__(self, wing_w=10.0, wing_epsilon=2.0):
        super(WingLoss, self).__init__()
        self.wing_w = wing_w
        self.wing_epsilon = wing_epsilon
        self.wing_c = self.wing_w * (1.0 - math.log(1.0 + self.wing_w / self.wing_epsilon))

    def forward(self, targets, predictions, euler_angle_weights=None):
        abs_error = torch.abs(targets - predictions)
        loss = torch.where(torch.le(abs_error, self.wing_w),
                           self.wing_w * torch.log(1.0 + abs_error / self.wing_epsilon), abs_error - self.wing_c)
        loss_sum = torch.sum(loss, 1)
        if euler_angle_weights is not None:
            loss_sum *= euler_angle_weights
        return torch.mean(loss_sum)


x = torch.rand(4,3,224,224)
x = x.cuda()
MobaNet = MobileNetV2().cuda()
AuxNet = AuxiliaryNet().cuda()
out,AuxInput = MobaNet(x)
AuxInput = AuxInput.cuda()
Auxout = AuxNet(AuxInput)

print("out shape: ",out.size())
print("Auxint shape: ",Auxout.size())
#
# print("Auxout shape: ",Auxout.size())






