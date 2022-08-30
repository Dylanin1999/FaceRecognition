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
        print(input.size())
        x = self.Conv2dRelu6(input)
        print(x.size())
        x = self.Dwise(x)
        print(x.size())
        out = self.Conv2dLinear(x)
        print(out.size())
        if self.strides == 1:
            print("!!!!RES ADD")
            out =out+input
        print(out.size())
        return x



s

class MobileNetV2(nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.BottleneckCfg = [
            [2, 64, 5, 2],
            [2, 128, 1, 2],
            [4, 128, 6, 1],
            [2, 16, 1, 1]
        ]
    def forward(self,input):
        for i in self.BottleneckCfg:
            bottleneck = InvertedResidual(i[0],i[1],i[2],i[3])
            outPut = bottleneck(input)

        return outPut

x = torch.rand(4,32,32,3)

net = MobileNetV2()
out = net(x)





