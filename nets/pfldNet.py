import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import summary


class bottleNect(nn.Sequential):
    def __init__(self,intput,inputChannel,outChannel,kernelSize,strides):
        paddings = kernelSize-1

    def Conv2d(self):
        x = nn.Sequential(
            nn.Conv2d(in_channels=self.inputChannel, out_channels=self.outChannel, kernel_size=self.kernelSize,
                      stride=self.strides, padding=self.paddings, bias=False)
        )

