import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import summary


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
        # print("111",input.size())
        x = self.Conv2dRelu6(input)
        # print("222",x.size())
        x = self.Dwise(x)
        # print("3333",x.size())
        out = self.Conv2dLinear(x)
        # print("444",out.size())
        if self.strides == 1:
            print("!!!!RES ADD")
            out =out+input
        # print("444",out.size())
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
    def forward(self,input):
        x = nn.Conv2d(in_channels=3,out_channels=self.BottleneckCfg[0][1],kernel_size=3,stride=2,padding=1)(input)
        x = nn.Conv2d(in_channels=self.BottleneckCfg[0][1], padding=1,out_channels=self.BottleneckCfg[0][1], kernel_size=3, stride=2,groups=self.BottleneckCfg[0][1])(x)
        for layer, i in enumerate(self.BottleneckCfg):
            for time in range(1,i[2]+1):
               # if time != i[2]-1:
                if time == 1:
                    if layer == 0:
                        bottleneck = InvertedResidual(i[1],i[1],i[0],i[3])
                    else:
                        bottleneck = InvertedResidual(self.BottleneckCfg[layer-1][1], i[1], i[0], i[3])
                else:
                    bottleneck = InvertedResidual(i[1], i[1], 1, i[3])
                x = bottleneck(x)
                print("operator out size: ",x.size())
                # else:
                #     bottleneck = InvertedResidual(i[1], self.BottleneckCfg[layer+1][1], i[0], i[3])
                #     x = bottleneck(x)
                #     print(x.size())

        return x

x = torch.rand(4,3,224,224)

net = MobileNetV2()
out = net(x)





