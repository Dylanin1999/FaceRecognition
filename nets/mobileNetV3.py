import torch.nn as nn
from collections import OrderedDict
import torch
from torchsummary import summary

#定义基本的Conv_Bn_activate
class baseConv(nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size,stride,groups=1,active=False,bias=False):
        super(baseConv, self).__init__()

        #定义使用的激活函数
        if active=='HS':
            ac=nn.Hardswish
        elif active=='RE':
            ac=nn.ReLU6
        else:
            ac=nn.Identity

        pad=kernel_size//2
        self.base=nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=kernel_size,stride=stride,padding=pad,groups=groups,bias=bias),
            nn.BatchNorm2d(outchannel),
            ac()
        )
    def forward(self,x):
        x=self.base(x)
        return x

#定义SE模块
class SEModule(nn.Module):
    def __init__(self,inchannels):
        super(SEModule, self).__init__()
        hidden_channel=int(inchannels/4)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.linear1=nn.Sequential(
            nn.Conv2d(inchannels,hidden_channel,1),
            nn.ReLU6()
        )
        self.linear2=nn.Sequential(
            nn.Conv2d(hidden_channel,inchannels,1),
            nn.Hardswish()
        )

    def forward(self,x):
        out=self.pool(x)
        out=self.linear1(out)
        out=self.linear2(out)
        return out*x

#定义bneck模块
class bneckModule(nn.Module):
    def __init__(self,inchannels,expand_channels,outchannels,kernel_size,stride,SE,activate):
        super(bneckModule, self).__init__()
        self.module=[]     #存放module

        if inchannels!=expand_channels:         #只有不相等时候才有第一层的升维操作
            self.module.append(baseConv(inchannels,expand_channels,kernel_size=1,stride=1,active=activate))

        self.module.append(baseConv(expand_channels,expand_channels,kernel_size=kernel_size,stride=stride,active=activate,groups=expand_channels))

        #判断是否有se模块
        if SE==True:
            self.module.append(SEModule(expand_channels))

        self.module.append(baseConv(expand_channels,outchannels,1,1))
        self.module=nn.Sequential(*self.module)

        #判断是否有残差结构
        self.residual=False
        if inchannels==outchannels and stride==1:
            self.residual=True

    def forward(self,x):
        out1=self.module(x)
        if self.residual:
            return out1+x
        else:
            return out1


#定义v3结构
class mobilenet_v3(nn.Module):
    def __init__(self,num_classes,init_weight=True):
        super(mobilenet_v3, self).__init__()

        # [inchannel,expand_channels,outchannels,kernel_size,stride,SE,activate]
        net_config = [[16, 16, 16, 3, 1, False, 'HS'],
                      [16, 64, 24, 3, 2, False, 'RE'],
                      [24, 72, 24, 3, 1, False, 'RE'],
                      [24, 72, 40, 5, 2, True, 'RE'],
                      [40, 120, 40, 5, 1, True, 'RE'],
                      [40, 120, 40, 5, 1, True, 'RE'],
                      [40, 240, 80, 3, 2, False, 'HS'],
                      [80, 200, 80, 3, 1, False, 'HS'],
                      [80, 184, 80, 3, 1, False, 'HS'],
                      [80, 184, 80, 3, 1, False, 'HS'],
                      [80, 480, 112, 3, 1, True, 'HS'],
                      [112, 672, 112, 3, 1, True, 'HS'],
                      [112, 672, 160, 5, 2, True, 'HS'],
                      [160, 960, 160, 5, 1, True, 'HS'],
                      [160, 960, 160, 5, 1, True, 'HS']]

        #定义一个有序字典存放网络结构
        modules=OrderedDict()
        modules.update({'layer1':baseConv(inchannel=3,kernel_size=3,outchannel=16,stride=2,active='HS')})

        #开始配置
        for idx,layer in enumerate(net_config):
            modules.update({'bneck_{}'.format(idx):bneckModule(layer[0],layer[1],layer[2],layer[3],layer[4],layer[5],layer[6])})

        modules.update({'conv_1*1':baseConv(layer[2],960,1,stride=1,active='HS')})
        modules.update({'pool':nn.AdaptiveAvgPool2d((1,1))})

        self.module=nn.Sequential(modules)

        self.classifier=nn.Sequential(
            nn.Linear(960,1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280,num_classes)
        )

        if init_weight:
            self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)


    def forward(self,x):
        out=self.module(x)
        out=out.view(out.size(0),-1)
        out=self.classifier(out)
        return out


if __name__ == '__main__':
    net=mobilenet_v3(10).to('cuda')
    summary(net,(3,224,224))