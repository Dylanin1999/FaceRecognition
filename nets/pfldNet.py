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

        #input kernel size,outchannel,stride,padding``
        self.auxiliaryCfg = [
            [3, 128, 2,1],
            [3, 128, 1,1],
            [3, 32, 2,1],
            [7, 128, 1,0],
        ]



    def endConv(self,input):
        s1 = input
        print("s1 Shape:", s1.size())
        s2 = nn.Conv2d(in_channels=self.BottleneckCfg[-1][1],out_channels=32,kernel_size=3,stride=2,padding=1)(s1)
        print("s2 Shape:", s2.size())
        s3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=7, stride=1)(s2)

        s1 = nn.AvgPool2d(kernel_size=s1.size()[-1])(s1)
        s2 = nn.AvgPool2d(kernel_size=s2.size()[-1])(s2)
        # s1 = torch.flatten(s1, 1)
        # s2 = torch.flatten(s2, 1)
        # s3 = torch.flatten(s3, 1)
        print("s1 Shape:", s1.size())
        print("s2 Shape:", s2.size())
        print("s3 Shape:", s3.size())
        out = torch.cat([s1, s2, s3], 1)
        out = torch.flatten(out,1)
      #      nn.Conv2d(in_channels=128,kernel_siz)
        return out

    #

    def baseConv(self,input,kernelSize,outChannel,stride,padding):
        out = nn.Conv2d(in_channels=input.size()[1],kernel_size=kernelSize,out_channels=outChannel,stride=stride,padding=padding)(input)
        return out


    def AuxiliaryNet(self,input):
        x = input
        for layer, i in enumerate(self.auxiliaryCfg):
            kernelSize = i[0]
            stride = i[2]
            padding = i[3]

            x = self.baseConv(input=x,kernelSize=kernelSize,outChannel=i[1],stride=stride,padding=padding)
            print("aux size: ",x.size())
        x = torch.flatten(x,1)
        x = nn.Linear(x.size()[1], 32)(x)
        x = nn.Linear(x.size()[1], 3)(x)
        print("aux out size: ", x.size())
        return x

    def forward(self,input):
        x = nn.Conv2d(in_channels=3,out_channels=self.BottleneckCfg[0][1],kernel_size=3,stride=2,padding=1)(input)
        x = nn.Conv2d(in_channels=self.BottleneckCfg[0][1], padding=1,out_channels=self.BottleneckCfg[0][1], kernel_size=3, stride=2,groups=self.BottleneckCfg[0][1])(x)
        for layer, i in enumerate(self.BottleneckCfg):
            for time in range(1, i[2]+1):
                if time == 1:
                    if layer == 0:
                        bottleneck = InvertedResidual(i[1],i[1],i[3],i[0])
                    else:
                        bottleneck = InvertedResidual(self.BottleneckCfg[layer-1][1], i[1], i[3], i[0])
                else:
                    bottleneck = InvertedResidual(i[1], i[1], 1,i[0])
                x = bottleneck(x)
            if layer==0:
                Auxint = x
                print("AUX input size: ",Auxint.size())
                auxout = self.AuxiliaryNet(Auxint)


        print("out Shape:", x.size())
        out = self.endConv(x)

        out = nn.Linear(out.size()[1],136)(out)
        return out,auxout

x = torch.rand(4,3,224,224)

net = MobileNetV2()

out,auxout = net(x)
print("out shape: ",out.size())





