"将DABNet模块换成混合空洞卷积[1,2,5]*3  [3,5,7]*3  [5,9,11]*3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ["DABNet"]

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size() #x:[3, 256, 56, 56]
        # print("x.size:",x.shape)
        # print("self.avg_pool(x):",self.avg_pool(x).shape)
        y = self.avg_pool(x).view(b, c) #[3, 256]
        # print("y1:",y.shape)
        # print("self.fc(y):",self.fc(y).shape)
        y = self.fc(y).view(b, c, 1, 1) #y:[3, 256, 1, 1]
        # print("y2:",y.shape)
        return x * y.expand_as(x) #[3, 256, 56, 56]

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)

        self.dconv3x1 = Conv(nIn, nIn, (dkSize, 1), 1,
                              padding=(1 * d[0], 0), dilation=(d[0], 1), groups=nIn, bn_acti=True)
        self.dconv1x3 = Conv(nIn, nIn, (1, dkSize), 1,
                              padding=(0, 1 * d[0]), dilation=(1, d[0]), groups=nIn, bn_acti=True)
        self.ddconv3x1 = Conv(nIn, nIn, (dkSize, 1), 1,
                              padding=(1 * d[1], 0), dilation=(d[1], 1), groups=nIn, bn_acti=True)
        self.ddconv1x3 = Conv(nIn, nIn, (1, dkSize), 1,
                              padding=(0, 1 * d[1]), dilation=(1, d[1]), groups=nIn, bn_acti=True)
        self.dddconv3x1 = Conv(nIn, nIn, (dkSize, 1), 1,
                              padding=(1 * d[2], 0), dilation=(d[2], 1), groups=nIn, bn_acti=True)
        self.dddconv1x3 = Conv(nIn, nIn, (1, dkSize), 1,
                              padding=(0, 1 * d[2]), dilation=(1, d[2]), groups=nIn, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn*4)
        self.conv1x1 = Conv(nIn*4, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br3 = self.dddconv3x1(output)
        br3 = self.dddconv1x3(br3)

        output = torch.cat((br1 , br2 , br3 , input ),1)  #br1 + br2 + br3 + input
        # output = br1 + br2 + br3
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output

# class DABModule(nn.Module):
#     def __init__(self, nIn, d, kSize=3, dkSize=3):
#         super().__init__()
#
#         self.bn_relu_1 = BNPReLU(nIn)
#         self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
#
#         self.dconv3x1 = Conv(nIn//2, nIn//2, (dkSize, 1), 1,
#                               padding=(1 * d[0], 0), dilation=(d[0], 1), groups=nIn//2, bn_acti=True)
#         self.dconv1x3 = Conv(nIn//2, nIn//2, (1, dkSize), 1,
#                               padding=(0, 1 * d[0]), dilation=(1, d[0]), groups=nIn//2, bn_acti=True)
#         self.ddconv3x1 = Conv(nIn//2, nIn//2, (dkSize, 1), 1,
#                               padding=(1 * d[1], 0), dilation=(d[1], 1), groups=nIn//2, bn_acti=True)
#         self.ddconv1x3 = Conv(nIn//2, nIn//2, (1, dkSize), 1,
#                               padding=(0, 1 * d[1]), dilation=(1, d[1]), groups=nIn//2, bn_acti=True)
#         self.dddconv3x1 = Conv(nIn//2, nIn//2, (dkSize, 1), 1,
#                               padding=(1 * d[2], 0), dilation=(d[2], 1), groups=nIn//2, bn_acti=True)
#         self.dddconv1x3 = Conv(nIn//2, nIn//2, (1, dkSize), 1,
#                               padding=(0, 1 * d[2]), dilation=(1, d[2]), groups=nIn//2, bn_acti=True)
#
#         self.bn_relu_2 = BNPReLU(nIn//2)
#         self.conv1x1 = Conv(nIn//2, nIn, 1, 1, padding=0, bn_acti=False)
#
#     def forward(self, input):
#         output = self.bn_relu_1(input)
#         output = self.conv3x3(output)  # 将输入通道数减半
#
#         br1 = self.dconv3x1(output)
#         br1 = self.dconv1x3(br1)
#         br2 = self.ddconv3x1(output)
#         br2 = self.ddconv1x3(br2)
#         br3 = self.dddconv3x1(output)
#         br3 = self.dddconv1x3(br3)
#
#         output = br1 + br2 + br3
#         output = self.bn_relu_2(output)
#         output = self.conv1x1(output)
#
#         return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input

class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):#2倍的上采样
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)#b,c,w,h->b,c*f^2,w,h
        self.up = nn.PixelShuffle(factor)#上采样的一种方式->b,c,h*f,w*f
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1,
                padding=1, groups=64, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 259, kernel_size=1, stride=1,
                padding=0, bias=False),
        )

        self.right = nn.Sequential(
            nn.Conv2d(
                259, 259, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(259),
            nn.Upsample(scale_factor=2)
        )
        self.Conv1 = Conv(259,259,3,1,padding=1, bn_acti=True)

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left = self.left(x_d)
        right = self.right(x_s)
        out = left + torch.sigmoid(right)
        out = self.Conv1(out)
        return out

class zyz_ANet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=[1,2,5]))
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # self.attention1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                                 Conv(64, 64, 1, 1, padding=0, bn_acti=True),
        #                                 nn.Conv2d(64, 64, 1, 1, padding=0),
        #                                 nn.BatchNorm2d(64, eps=1e-3),
        #                                 nn.Sigmoid(),
        #                                 )
        # self.adap = nn.AdaptiveAvgPool2d(1)
        # self.conv = Conv(64, 64, 1, 1, padding=0, bn_acti=True)
        # self.con = nn.Conv2d(64, 64, 1, 1, padding=0)
        # self.bn = nn.BatchNorm2d(64, eps=1e-3)
        # self.sigmoid = nn.Sigmoid()

        # DAB Block 2
        dilation_block_2 = [[3,5,7],[3,5,7],[3,5,7],[5,9,11],[5,9,11],[5,9,11]]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)

        # self.attention2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                                 Conv(128, 128, 1, 1, padding=0, bn_acti=True),
        #                                 nn.Conv2d(128, 128, 1, 1, padding=0),
        #                                 nn.BatchNorm2d(128, eps=1e-3),
        #                                 nn.Sigmoid()
        #                                 )

        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

        self.eca_layer = eca_layer()
        # self.se1 = SELayer(32 + 3, 16)
        # self.se2 = SELayer(128 + 3, 16)
        # self.se3 = SELayer(256 + 3, 16)

        self.bga = BGALayer()


        self.conv0 = nn.Sequential(Conv(35, classes, 1, 1, padding=0))
        self.conv1 = nn.Sequential(Conv(131, classes, 1, 1, padding=0))
        self.conv2 = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):
        b,c,h,w = input.size()
        output0 = self.init_conv(input)  #初始块[1, 32, 256, 512]

        down_1 = self.down_1(input) #[1, 3, 256, 512]第一次下采样
        down_2 = self.down_2(input) #[1, 3, 128, 256]第二次下采样
        down_3 = self.down_3(input) #[1, 3, 64, 128]第三次下采样

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1)) #[1, 35, 256, 512]
        output0_cat = self.eca_layer(output0_cat) #混合之后，下采样之前加入通道注意力[1, 35, 256, 512]
        # output0_cat = self.se1(output0_cat)

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat) #[1, 64, 128, 256]，我要融合的
        output1 = self.DAB_Block_1(output1_0)
        # a1 = self.attention1(output1)
        # output1 = torch.mul(a1,output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))
        output1_cat = self.eca_layer(output1_cat) #[1, 131, 128, 256]
        # output1_cat = self.se2(output1_cat)

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat) #[1, 128, 64, 128]
        output2 = self.DAB_Block_2(output2_0)
        # a2 = self.attention2(output2)
        # output2 = torch.mul(a2, output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))
        output2_cat = self.eca_layer(output2_cat) #[1, 259, 64, 128],我要融合的
        # output2_cat = self.se3(output2_cat)

        out = self.bn_prelu_3(self.bga(output1_0,output2_cat))#(1,259,128,256)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        """out2 = F.interpolate(output2_cat, (h//2,w//2), mode='bilinear', align_corners=False) #[1, 259, 256, 512]
        out2 = self.conv2(out2)
        out1 = F.interpolate(output1_cat, (h//2,w//2), mode='bilinear', align_corners=False) #[1, 131, 256, 512]
        out1 = self.conv1(out1)
        out0 = output0_cat #[1, 35, 256, 512]
        out0 = self.conv0(out0) #[1, 19, 256, 512]

        out = out2 + out1 + out0
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)"""

        return out
# input=torch.randn(1,3,512,1024)
# model=eighteen_DABNet(classes=19, block_1=3, block_2=6)
# output=model(input)
# # print(model)
# print(output.shape)
# # from torchsummary import summary
# # # summary(model, input_size=(3,512,1024),device='cpu')
# from torchstat import stat
# stat(model, (3, 512,1024)) #680,888

# def netParams(model):
#     """
#     computing total network parameters
#     args:
#        model: model
#     return: the number of parameters
#     """
#     total_paramters = 0
#     for parameter in model.parameters():
#         i = len(parameter.size())
#         p = 1
#         for j in range(i):
#             p *= parameter.size(j)
#         total_paramters += p
#
#     return total_paramters
#
# total_paramters = netParams(model)
# print(total_paramters)
