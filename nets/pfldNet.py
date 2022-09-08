import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import summary
import math

class InvertedResidual(nn.Module):
    def __init__(self,inputChannel,outChannel,strides,expand_ratio):
        super(InvertedResidual, self).__init__()
        #self.paddings = kernelSize-1
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

    # def endConv(self, input):
    #     s1 = input
    #     s2 = nn.Conv2d(in_channels=self.BottleneckCfg[-1][1], out_channels=32, kernel_size=3, stride=2, padding=1)(s1)
    #     s3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=7, stride=1)(s2)
    #     s1 = nn.AvgPool2d(kernel_size=s1.size()[-1])(s1)
    #     s2 = nn.AvgPool2d(kernel_size=s2.size()[-1])(s2)
    #     out = torch.cat([s1, s2, s3], 1)
    #     out = torch.flatten(out, 1)
    #     return out

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

        # print( self.bottleneckList)
        # self.endConvS  = self.endConv()


        self.Conv2d_1 = nn.Conv2d(in_channels=3,out_channels=self.BottleneckCfg[0][1],kernel_size=3,stride=2,padding=1)
        self.Conv2d_2 = nn.Conv2d(in_channels=self.BottleneckCfg[0][1], padding=1,out_channels=self.BottleneckCfg[0][1], kernel_size=3, stride=2,groups=self.BottleneckCfg[0][1])

    def forward(self,input):
        x = self.Conv2d_1(input)
        x = self.Conv2d_2(x)

        for bottleNeck in self.bottleneckList:
            x = bottleNeck(x)



            # if layer==0:
            #     Auxint = x
        # out = self.endConv(x)

        # out = nn.Linear(out.size()[1],136)(out)
        return x#,Auxint

class  AuxiliaryNet(nn.Module):

    def __init__(self):
        super(AuxiliaryNet, self).__init__()

        # input kernel size,outchannel,stride,padding``
        self.auxiliaryCfg = [
            [3, 128, 2, 1],
            [3, 128, 1, 1],
            [3, 32, 2, 1],
            [7, 128, 1, 0],
        ]

        self.baseConv = nn.Sequential(
            nn.Conv2d(in_channels=input.size()[1], kernel_size=kernelSize, out_channels=outChannel, stride=stride,
                        padding=padding)(input)
        )

    def baseConv(self, input, kernelSize, outChannel, stride, padding):
        out = nn.Conv2d(in_channels=input.size()[1], kernel_size=kernelSize, out_channels=outChannel, stride=stride,
                        padding=padding)(input)
        return out

    def forward(self,input):
        for layer, i in enumerate(self.auxiliaryCfg):
            kernelSize = i[0]
            stride = i[2]
            padding = i[3]
            x = input
            x = self.baseConv(input=x,kernelSize=kernelSize,outChannel=i[1],stride=stride,padding=padding)
            x = torch.flatten(x,1)
            x = nn.Linear(x.size()[1], 32)(x)
            x = nn.Linear(x.size()[1], 3)(x)
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

# class TestL(nn.Module):
#     def __init__(self):
#         super(TestL, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1)
#         self.conv2 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1)
#         self.conv3 = nn.Conv2d(in_channels=3,out_channels=24,kernel_size=3,stride=1)
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x

x = torch.rand(4,3,224,224)
x = x.cuda()
net = MobileNetV2().cuda()
# aux = AuxiliaryNet().cuda()
out = net(x)
# out,Auxint = net(x)
# Auxout = aux(Auxint)
print("out shape: ",out.size())
# print("Auxint shape: ",Auxint.size())
#
# print("Auxout shape: ",Auxout.size())






