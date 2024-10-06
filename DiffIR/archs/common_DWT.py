import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):  #若k =3  则无论d为多少图像大小不变
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2) * dilation , bias=bias, dilation=dilation) #padding=(kernel_size // 2) + dilation - 1


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):  #group代表分组卷积
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)


# def shuffle_channel()

def channel_shuffle(x, groups): #通道洗牌
    batchsize, num_channels, height, width = x.size()  #B C H W

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)  # B G C/G H W

    x = torch.transpose(x, 1, 2).contiguous()  #B C/G G H W

    # flatten
    x = x.view(batchsize, -1, height, width) # B C H W

    return x


def pixel_down_shuffle(x, downsacale_factor):  #像素降size扩大channel并洗牌
    batchsize, num_channels, height, width = x.size()  #B C H W

    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels, out_height, downsacale_factor, out_width,  # B C H/D D W/D D
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()  #B C D D H/D W/D

    return unshuffle_out.view(batchsize, num_channels, out_height, out_width) #B C*D*D H/D H/D


class DWTForward(nn.Module):

    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y


def sp_init(x): #像素切分
    x01 = x[:, :, 0::2, :]   #双冒号   开始：结束：步长  进行切片  ||结尾省略的时候，默认到数组最后
    x02 = x[:, :, 1::2, :]
    x_LL = x01[:, :, :, 0::2]
    x_HL = x02[:, :, :, 0::2]
    x_LH = x01[:, :, :, 1::2]
    x_HH = x02[:, :, :, 1::2]

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def dwt_init(x):  #哈尔小波变换，见论文eq2
    x01 = x[:, :, 0::2, :] / 2
    # print(x01.size())
    x02 = x[:, :, 1::2, :] / 2
    # print(x02.size())
    x1 = x01[:, :, :, 0::2]
    # print(x1.size())
    x2 = x02[:, :, :, 0::2]
    # print(x2.size())
    x3 = x01[:, :, :, 1::2]
    # print(x3.size())
    x4 = x02[:, :, :, 1::2]
    # print(x4.size())
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):  #哈尔小波逆变换，见论文eq3
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()  #B C H W
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(   #B C/4 H*2 W*2
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class Channel_Shuffle(nn.Module): # 封装通道洗牌
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)


class SP(nn.Module):  #封装像素切分
    def __init__(self):
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return sp_init(x)


class Pixel_Down_Shuffle(nn.Module):  #封装降像素洗牌
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)


class DWT(nn.Module):  #封装DWT变换
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module): #封装IWT逆变换
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.create_graph = False
            self.volatile = True


class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign == -1:
            self.volatile = True


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=False, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class BBlock(nn.Module): # 基础块Conv+Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DBlock_com(nn.Module):  #Conv（d=2）+Relu+Conv（d=3）+Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2)) #空洞d=2
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3)) #空洞d=3
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv(nn.Module):  #Conv（d=3）+Relu+Conv（d=2）+Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com1(nn.Module):  #Conv（d=2）+Relu+Conv（d=1）+Relu  #怀疑写反
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv1(nn.Module): #Conv（d=2）+Relu+Conv（d=1）+Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com2(nn.Module): #Conv（d=2）+Relu+Conv（d=2）+Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv2(nn.Module):  #Conv（d=2）+Relu+Conv（d=2）+Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class ShuffleBlock(nn.Module):  #Conv + Channel_Shuffle + Relu
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, conv_groups=1):
        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):  #Conv+Relu+Conv（k=1）+Relu
    def __init__(
            self, conv, conv1, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Block(nn.Module): #4个conv
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
