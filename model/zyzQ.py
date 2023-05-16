"在K的基础上，去掉原图的1/4，1/8,修改下采样，R=6的3x3空洞卷积与3x3卷积相加，后与MAXPOOL CONCAT"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ["DABNet"]

class ShuffleBlock(nn.Module):#通道混洗
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

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

        dividedRate=0.5
        self.rightC = nIn - round(nIn * dividedRate)#round是四舍五入
        self.leftC = round(nIn * dividedRate)

        self.dconv3x1 = Conv(self.rightC, self.rightC, (dkSize, 1), 1,
                              padding=(1 * d[0], 0), dilation=(d[0], 1), groups=1, bn_acti=True)
        self.dconv1x3 = Conv(self.rightC, self.rightC, (1, dkSize), 1,
                              padding=(0, 1 * d[0]), dilation=(1, d[0]), groups=1, bn_acti=True)
        self.ddconv3x1 = Conv(self.leftC, self.leftC, (dkSize, 1), 1,
                              padding=(1 * d[1], 0), dilation=(d[1], 1), groups=1, bn_acti=True)
        self.ddconv1x3 = Conv(self.leftC, self.leftC, (1, dkSize), 1,
                              padding=(0, 1 * d[1]), dilation=(1, d[1]), groups=1, bn_acti=True)
        self.conv3x3 = Conv(nIn, nIn, 3, 1,
                              padding=1, dilation=1, groups=1, bn_acti=True)
        self.conv1x1 = Conv(nIn*3, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle_end = ShuffleBlock(groups=nIn)


    def forward(self, input):
        output = self.bn_relu_1(input)

        xlData = output[:, :self.leftC, :, :]#左边支路处理的通道是从头到leftC
        xrData = output[:, self.leftC:, :, :]#右边支路处理的通道是从leftC到最后

        br11 = self.dconv3x1(xlData)
        br21 = self.ddconv3x1(xrData)
        b1 = br11 + br21
        br1 = self.dconv1x3(b1)
        br2 = self.ddconv1x3(b1)
        br3 = self.conv3x3(output)

        output = torch.cat((br1 , br2 , br3 , output),1)  #c + c + c
        output = self.conv1x1(output)
        output = self.shuffle_end(output)
        return output

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
        self.conv1x1 = Conv(nOut, nOut, kSize=1, stride=1, padding=0)
        self.conv3x32 = Conv(nIn, nConv, kSize=3, stride=2, padding=1, dilation=6)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        output1 = self.conv3x32(input)
        output = output + output1

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)
            output = self.conv1x1(output)

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

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace= True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class zyz_QNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.bn_prelu_1 = BNPReLU(32 + 3)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=[2,5]))
        self.bn_prelu_2 = BNPReLU(128)

        # DAB Block 2
        dilation_block_2 = [[3,7],[3,7],[3,7],[9,11],[9,11],[9,11]]
        self.downsample_2 = DownSamplingBlock(128 , 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256)

        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

        self.eca_layer = eca_layer()

        self.upsample_1 = UpsamplerBlock(256, classes)
        self.upsample_2 = UpsamplerBlock(128, classes)
        #self.upsample_3 = UpsamplerBlock(35, classes)
        self.upsample_4 = UpsamplerBlock(classes, classes)
        self.conv0 = nn.Sequential(Conv(35, classes, 1, 1, padding=0))

    def forward(self, input):
        b,c,h,w = input.size()
        output0 = self.init_conv(input)  #初始块[1, 32, 256, 512]

        down_1 = self.down_1(input) #[1, 3, 256, 512]第一次下采样

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1)) #[1, 35, 256, 512]
        output0_cat = self.eca_layer(output0_cat) #混合之后，下采样之前加入通道注意力[1, 35, 256, 512]

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat) #[1, 64, 128, 256]
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0], 1))#[1, 128, 128, 256]
        output1_cat = self.eca_layer(output1_cat) #[1, 128, 128, 256]


        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat) #[1, 128, 64, 128]
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0], 1)) #[1, 256, 64, 128]
        output2_cat = self.eca_layer(output2_cat) #[1, 256, 64, 128]

        out2 = self.upsample_1(output2_cat)  # [1, 19, 128,256]
        out2 = self.upsample_4(out2)  # [1, 19, 256, 512]
        out1 = self.upsample_2(output1_cat)  # [1, 19, 256, 512]
        out0 = self.conv0(output0_cat)  # [1, 19, 256, 512]
        out = out2 + out1 + out0
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)# [1, 19, 512,1024]

        return out
