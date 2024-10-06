"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import math
from models.utils import get_upsample_filter
from models.imagenet.mobilenetv2 import InvertedResidual
from models.imagenet.gate import GateModule

__all__ = ['mobilenetv2dct_inputgate_deconv']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1., doubleGate=False, dwLA=False, wo_blockGate=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 1],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual

        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                if 'gate_l' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)
                    m.bias.data[::2].fill_(0.1)
                    m.bias.data[1::2].fill_(2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def mobilenetv2(pretrained=True, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model =  MobileNetV2(**kwargs)
    if pretrained:
        state_dict = torch.load('/mnt/ssd/kai.x/work/code/iftc/pretrained/mobilenetv2_1.0-0c6065bc.pth')
        model.load_state_dict(state_dict)
    # if pretrained:
    #     state_dict = torch.load('/mnt/ssd/kai.x/work/code/iftc/pretrained/mobilenetv2_1.0-0c6065bc.pth')
    #     model_state = model.state_dict()
    #     for name, param in state_dict.items():
    #         if name in model_state: # and isinstance(param, nn.Parameter):
    #             model_state[name].data.copy_(param.data)
    #         else:
    #             print('This layer is not loaded: {}'.format(name))

    return model

class MobileNetV2DCT_Inputgate_Deconv(nn.Module):
    def __init__(self, input_gate=False, doubleGate=False, dwLA=False):
        super(MobileNetV2DCT_Inputgate_Deconv, self).__init__()

        self.input_gate, self.doubleGate, self.dwLA = input_gate, doubleGate, dwLA

        in_ch, out_ch = 64, 64
        self.in_ch= in_ch

        self.conv_y = nn.Sequential(
            # TODO
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))
        self.deconv_cb = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))
        self.deconv_cr = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))
        self.input_layer = nn.Sequential(
            # pw
            nn.Conv2d(3*in_ch, 3*in_ch, 3, 1, 1, groups=3*in_ch, bias=False),
            nn.BatchNorm2d(3*in_ch),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(3*in_ch, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24))

        if input_gate:
            self.inp_GM = GateModule(192, 28, doubleGate, dwLA)

        # only initialize layers not included in the MobileNet v2
        self._initialize_weights()

        model = mobilenetv2(pretrained=True, doubleGate=doubleGate, dwLA=dwLA)
        start_layer = 4
        self.features = nn.Sequential(*list(model.children())[0][start_layer:])
        # self.features = nn.Sequential(*list(model.children())[0][4:])
        print('start from layer: {}'.format(start_layer))
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]


    def forward(self, dct_y, dct_cb, dct_cr, temperature=1.):
        if self.input_gate:
            x_y, x_cb, x_cr, inp_atten = self.inp_GM(dct_y, dct_cb, dct_cr)
            x_y = self.conv_y(x_y)
            x_cb = self.deconv_cb(x_cb)
            x_cr = self.deconv_cr(x_cr)
        else:
            x_y = self.conv_y(dct_y)
            x_cb = self.deconv_cb(dct_cb)
            x_cr = self.deconv_cr(dct_cr)

        x = torch.cat((x_y, x_cb, x_cr), dim=1)
        x = self.input_layer(x)

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.input_gate:
            return x, inp_atten
        else:
            return x

    def _initialize_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                if 'gate_l' in str(k):
                    # Initialize last layer of gate with low variance
                    m.weight.data.normal_(0, 0.001)
                    m.bias.data[::2].fill_(0.1)
                    m.bias.data[1::2].fill_(2)
            elif isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def mobilenetv2dct_inputgate_deconv(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_Inputgate_Deconv(**kwargs)

    return model


if __name__ == '__main__':
    import numpy as np

    dct_y  = torch.from_numpy(np.random.randn(16, 64, 28, 28)).float()
    dct_cb = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()
    dct_cr = torch.from_numpy(np.random.randn(16, 64, 14, 14)).float()

    # model_mobilenetv2autosubset = mobilenetv2dct_autosubset(input_gate=True, doubleGate=False, dwLA=True)
    # x = model_mobilenetv2autosubset(dct_y, dct_cb, dct_cr)
    # print(x[0].shape)
    #
    # model_mobilenetv2autosubset = mobilenetv2dct_autosubset(input_gate=True, doubleGate=False, dwLA=False)
    # x = model_mobilenetv2autosubset(dct_y, dct_cb, dct_cr)
    # print(x[0].shape)

    # model_mobilenetv2autosubset = mobilenetv2dct_autosubset(input_gate=True, doubleGate=True, dwLA=False)
    # x = model_mobilenetv2autosubset(dct_y, dct_cb, dct_cr)
    # print(x[0].shape)

    model_mobilenetv2autosubset = mobilenetv2dct_inputgate_deconv(input_gate=True, doubleGate=False, dwLA=False)
    x = model_mobilenetv2autosubset(dct_y, dct_cb, dct_cr)
    import csv
    with open("input.csv", "w") as f:
        w = csv.writer(f)
        for key, val in list(model_mobilenetv2autosubset.named_parameters()):
            w.writerow([key, val])
    print(x[0].shape)


