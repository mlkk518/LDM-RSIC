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
import operator
from itertools import islice
from collections import OrderedDict
from models.imagenet.gumbel import GumbleSoftmax
from models.imagenet.mobilenetv2 import InvertedResidual

__all__ = ['mobilenetv2dct_autosubset_alllayers']


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


class Sequential_gate(nn.Module):
    def __init__(self, *args):
        super(Sequential_gate, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential_gate, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input, temperature=1, openings=None):
        gate_activations = []
        for i, module in enumerate(self._modules.values()):
            input, gate_activation = module(input, temperature)
            gate_activations.extend(gate_activation)
        return input, gate_activations

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
        if wo_blockGate:
            block = InvertedResidual
        else:
            block = InvertedResidualGate

        spatial_size = 56
        for t, c, n, s in self.cfgs:
            if s == 2: spatial_size //= 2
            output_channel = _make_divisible(c * width_mult, 8)
            if wo_blockGate:
                layers.append(block(input_channel, output_channel, s, t))
            else:
                layers.append(block(input_channel, output_channel, s, t, spatial_size, doubleGate, dwLA))
            input_channel = output_channel
            for i in range(1, n):
                if wo_blockGate:
                    layers.append(block(input_channel, output_channel, 1, t))
                else:
                    layers.append(block(input_channel, output_channel, 1, t, spatial_size, doubleGate, dwLA))
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
        model_state = model.state_dict()
        for name, param in state_dict.items():
            if name in model_state: # and isinstance(param, nn.Parameter):
                model_state[name].data.copy_(param.data)
            else:
                new_name = name.split('.')
                if int(new_name[3]) == 6:
                    new_name[2], new_name[3] ='conv2', '0'
                    new_name = '.'.join(new_name)
                    model_state[new_name].data.copy_(param.data)
                elif int(new_name[3]) == 7:
                    new_name[2], new_name[3] ='conv2', '1'
                    new_name = '.'.join(new_name)
                    model_state[new_name].data.copy_(param.data)
                else:
                    print('This layer is not loaded: {}'.format(name))
    return model

class InvertedResidualGate(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, spatial_size, doubleGate, dwLA):
        super(InvertedResidualGate, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        self.oup = oup
        self.doubleGate, self.dwLA = doubleGate, dwLA

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )

            # split here to insert a GM
            self.conv2 = nn.Sequential(
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.GM =  GateModule(hidden_dim, 4, spatial_size, doubleGate, dwLA)
        self.GM2 = GateModule(oup, 2, spatial_size, doubleGate, dwLA)
        self.gs = GumbleSoftmax()

    def forward(self, x, temperature=1):
        x_hatten = []
        residual = x
        x = self.conv(x)
        x, x_hatten2 = self.GM(x)
        x_hatten.append(x_hatten2)

        x = self.conv2(x)
        x, x_hatten2  = self.GM2(x)
        x_hatten.append(x_hatten2)

        if self.identity:
            return residual + x, x_hatten
        else:
            return x , x_hatten

class GateModule(nn.Module):
    def __init__(self, in_ch, reduction=1, kernel_size=28, doubleGate=False, dwLA=False):
        super(GateModule, self).__init__()

        self.doubleGate, self.dwLA = doubleGate, dwLA
        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch

        if dwLA:
            if doubleGate:
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch,
                              bias=True),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch, bias=True),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch,
                                   bias=True)
        else:
            if doubleGate:
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch // reduction),
                nn.ReLU6(inplace=True),
            )
            self.inp_gate_l = nn.Conv2d(in_ch // reduction, in_ch * 2, kernel_size=1, stride=1, padding=0,
                                        groups=in_ch // reduction, bias=True)

    def forward(self, x, temperature=1.):
        if self.doubleGate:
            if self.dwLA:
                hatten_d1 = self.inp_att(x)
                hatten_d2 = self.inp_gate(x)
                hatten_d2 = self.inp_gate_l(hatten_d2)
            else:
                hatten = self.avg_pool(x)
                hatten_d1 = self.inp_att(hatten)
                hatten_d2 = self.inp_gate(hatten)
                hatten_d2 = self.inp_gate_l(hatten_d2)

            hatten_d2 = hatten_d2.reshape(hatten_d2.size(0), self.in_ch, 2, 1)
            hatten_d2 = self.inp_gs(hatten_d2, temp=temperature, force_hard=True)
        else:
            if self.dwLA:
                hatten_d2 = self.inp_gate(x)
                hatten_d2 = self.inp_gate_l(hatten_d2)
            else:
                if isinstance(x, list):
                    hatten_y, hatten_cb, hatten_cr = self.avg_pool(x[0]), self.avg_pool(x[1]), self.avg_pool(x[2])
                    hatten_d2 = torch.cat((hatten_y, hatten_cb, hatten_cr), dim=1)
                else:
                    hatten_d2 = self.avg_pool(x)
                hatten_d2 = self.inp_gate(hatten_d2)
                hatten_d2 = self.inp_gate_l(hatten_d2)

            hatten_d2 = hatten_d2.reshape(hatten_d2.size(0), self.in_ch, 2, 1)
            hatten_d2 = self.inp_gs(hatten_d2, temp=temperature, force_hard=True)


        if self.doubleGate:
            x = x * hatten_d1 * hatten_d2[:, :, 1].unsqueeze(2)
        else:
            if isinstance(x, list):
                x[0] = x[0] * hatten_d2[:, :64, 1].unsqueeze(2)
                x[1] = x[1] * hatten_d2[:, 64:128, 1].unsqueeze(2)
                x[2] = x[2] * hatten_d2[:, 128:, 1].unsqueeze(2)
                return x[0], x[1], x[2], hatten_d2[:, :, 1]
            else:
                x = x * hatten_d2[:, :, 1].unsqueeze(2)
                return x, hatten_d2[:, :, 1]

class MobileNetV2DCT_AUTOSUBSET(nn.Module):
    def __init__(self, input_gate=False, doubleGate=False, dwLA=False):
        super(MobileNetV2DCT_AUTOSUBSET, self).__init__()

        self.input_gate, self.doubleGate, self.dwLA = input_gate, doubleGate, dwLA

        in_ch, out_ch = 64, 64
        self.in_ch= in_ch

        self.conv_y = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))
        self.deconv_cb = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))
        self.deconv_cr = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False),
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
            self.y_GM   = GateModule(64, 28, doubleGate, dwLA)
            self.cb_GM  = GateModule(64, 28, doubleGate, dwLA)
            self.cr_GM  = GateModule(64, 28, doubleGate, dwLA)

        # only initialize layers not included in the MobileNet v2
        self._initialize_weights()

        model = mobilenetv2(pretrained=True, doubleGate=doubleGate, dwLA=dwLA)
        self.features = Sequential_gate(*list(model.children())[0][3:])
        self.conv = list(model.children())[1]
        self.avgpool = list(model.children())[2]
        self.classifier = list(model.children())[3]

    def forward(self, dct_y, dct_cb, dct_cr, temperature=1):
        if self.input_gate:
            x_y, x_cb, x_cr, inp_atten = self.inp_GM([dct_y, dct_cb, dct_cr])
            x_y = self.conv_y(x_y)
            x_cb = self.deconv_cb(x_cb)
            x_cr = self.deconv_cr(x_cr)
        else:
            x_y = self.conv_y(dct_y)
            x_cb = self.deconv_cb(dct_cb)
            x_cr = self.deconv_cr(dct_cr)

        x = torch.cat((x_y, x_cb, x_cr), dim=1)
        x = self.input_layer(x)

        # x, inp_hatten = self.inp_GM(x)

        x, atten = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.input_gate:
            atten.insert(0, inp_atten)

        return x, atten

    def _initialize_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

def mobilenetv2dct_autosubset_alllayers(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2DCT_AUTOSUBSET(**kwargs)

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

    model_mobilenetv2autosubset = mobilenetv2dct_autosubset_alllayers(input_gate=True, doubleGate=False, dwLA=False)
    x = model_mobilenetv2autosubset(dct_y, dct_cb, dct_cr)
    print(x[0].shape)


