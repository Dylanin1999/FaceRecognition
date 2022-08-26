import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import summary


class bottleNeck(nn.Sequential):
    def __init__(self,intput,inputChannel,outChannel,kernelSize,strides):
        self.paddings = kernelSize-1
        self.inputChannel = inputChannel
        self.outChannel = outChannel
        self.strides = strides

    def Conv2dRelu6(self):
        x = nn.Sequential(
            nn.Conv2d(in_channels=self.inputChannel, out_channels=self.outChannel, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.outChannel),
            nn.ReLU6(True)
        )

    def Conv2dLinear(self):
        x = nn.Sequential(
            nn.Conv2d(in_channels=self.inputChannel, out_channels=self.outChannel, kernel_size=1,
                      stride=1, padding=0, bias=False),

        )
    def Dwise(self):
        x = nn.Sequential(
            nn.Conv2d(in_channels=self.outChannel,out_channels=self.outChannel,kernel_size=3,stride=self.strides,groups=self.outChannel),
            nn.BatchNorm2d(self.outChannel),
            nn.ReLU6(True)
        )

    def forward(self, input):
        x = self.Conv2dRelu6(input)
        x = self.Dwise(x)
        x = self.Conv2dLinear(x)

