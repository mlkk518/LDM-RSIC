
import torch
import torch.nn as nn
from models.CBAM import CBAM
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, stride=stride)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        x = x2 + x1
        return self.conv(x)


class Up_last(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class Fuse_module(nn.Module):
    def __init__(self, ch=32, reduction_ratio=4, Last=False):
        super(Fuse_module, self).__init__()

        self.last =  Last
        # k = 512
        dim1 = int(ch/3)
        dim2 = ch - 2*dim1

        self.f_vl1 = nn.Sequential(
            nn.Conv2d(ch, dim1, kernel_size=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_vl2 = nn.Sequential(
            nn.Conv2d(ch, dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True)
        )
        self.f_vl3 = nn.Sequential(
            nn.Conv2d(ch, dim1, kernel_size=5, padding=2),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_ir1 = nn.Sequential(
            nn.Conv2d(ch, dim1, kernel_size=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_ir2 = nn.Sequential(
            nn.Conv2d(ch, dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim2),
            nn.ReLU(inplace=True)
        )
        self.f_ir3 = nn.Sequential(
            nn.Conv2d(ch, dim1, kernel_size=5, padding=2),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Conv2d(2*ch, ch, kernel_size=1)


        self.cbam = CBAM(ch, reduction_ratio=reduction_ratio)
        self.down =  Down(ch, out_channels= 2*ch, stride=2)


    def forward(self, inputs, inputf, pre_input=None): ##  inputs  student,  inputf

        vl_feat1 = self.f_vl1(inputs)  # dim1
        vl_feat2 = self.f_vl2(inputs)
        vl_feat3 = self.f_vl3(inputs)

        ir_feat1 = self.f_ir1(inputf)  # dim1
        ir_feat2 = self.f_ir2(inputf)
        ir_feat3 = self.f_ir3(inputf)
        cat_feat = torch.cat([vl_feat1, vl_feat2, vl_feat3, ir_feat1, ir_feat2, ir_feat3], dim=1)
        cat_feat = self.conv1x1(cat_feat)

        if pre_input != None:
            cat_feat = cat_feat + pre_input
        out1 = self.cbam(cat_feat)

        if self.last == False:
            out = self.down(out1)
        else:
            out = out1

        return out, out1

