import torch
import torch.nn as nn



# 均匀分布
def weight_init(net):
    for m in net.modules():
        # print("!!!!! weight_xavier_init type(m).__module__ ", type(m).__module__)
        if type(m).__module__ != 'DiffIR.archs.VGG16':
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#  xavier  正态分布
def weight_xavier_init(net):
    for m in net.modules():
        if type(m).__module__ != 'DiffIR.archs.VGG16':
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


#  正交初始化
def weight_orthogonal_init(net):
    for m in net.modules():
        if type(m).__module__ != 'DiffIR.archs.VGG16':
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                   m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()