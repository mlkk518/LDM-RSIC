
import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class Attention_dilated(nn.Module):
    def __init__(self, dim, num_heads, bias, stride):
        super(Attention_dilated, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Sequential(
                                 nn.Conv2d(dim, dim*3, kernel_size=3, bias=bias, padding=1),
                                 )
        self.qkv_dwconv = nn.Sequential(
                                        nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=stride,  groups=dim * 3, bias=bias),
                                        )
        self.project_out = nn.Sequential(
                                         nn.Conv2d(dim, dim, kernel_size=3, bias=bias, padding=1),)

    def forward(self, x):
        b, c, h, w = x.shape
        identify = x

        conv_qkv = self.qkv(x)

        qkv = self.qkv_dwconv(conv_qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # # 沿着指定的维度进行平均池化
        # q = torch.mean(q, dim=2, keepdim=True)
        # k = torch.mean(k, dim=3, keepdim=True)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        output = self.project_out(out) + identify
        return output

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_dilated(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, stride):
        super(FeedForward_dilated, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=stride, groups=hidden_features*2, bias=bias, dilation=stride)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class DTB_block(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, stride):
        super(DTB_block, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_dilated(dim, num_heads, bias, stride=stride)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_dilated(dim, ffn_expansion_factor, bias, stride=stride)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.act1 = nn.LeakyReLU(0.2, inplace=True)
        DTB_block(dim=self.mid_channel, num_heads=4, bias=False, ffn_expansion_factor=1.0,
                  LayerNorm_type='BiasFree', stride=2)

        # self.conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)
        # # self.bn2 = nn.BatchNorm2d(out_channels * 2)
        # self.act2 = nn.LeakyReLU(0.2, inplace=True)
        # self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x = self.act1(self.bn1(self.conv1(x)))
        batch = x.shape[0]
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv2(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act2(ffted)
        ffted = self.conv3(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')

        return output + x

class SplitChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SplitChannelAttention, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer_frequency = nn.Sequential(
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.layer_spatial = nn.Sequential(
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.layer(x)
        return x * self.layer_frequency(squeeze), x * self.layer_spatial(squeeze)


##  高低频分支   mlkk refer to  STDIP
class Global_local_module(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding_mode, need_bias):
        super(Global_local_module, self).__init__()
        self.sca = SplitChannelAttention(channel=in_channel, ratio=16)

        self.fu = FourierUnit(in_channel, out_channel)
        # self.padding_mode = padding_mode
        # to_pad = int((kernel_size - 1) / 2)
        # if padding_mode == 'reflection':
        #     self.pad1 = nn.ReflectionPad2d(to_pad)
        #     self.pad2 = nn.ReflectionPad2d(to_pad)
        #     to_pad = 0

        self.spatial_bratch = DenseBlock(num_repeats=3)

        # self.spatial_branch1 = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
        #     # nn.BatchNorm2d(in_channel),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.spatial_branch2 = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=to_pad, dilation=1, bias=need_bias),
        #     # nn.BatchNorm2d(out_channel),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )


    def forward(self, x):
        x_f, x_s = self.sca(x)
        # if self.padding_mode == 'reflection':
        #     x_s = self.pad1(x_s)
        # x_s = self.spatial_branch1(x_s)
        # if self.padding_mode == 'reflection':
        #     x_s = self.pad2(x_s)
        # x_s = self.spatial_branch2(x_s)

        spatial_local = self.spatial_bratch(x_s)
        frequency_global = self.fu(x_f)

        return spatial_local,  frequency_global


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, RELU=True):
        super().__init__()

        if RELU == True:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, int(in_channels//2), kernel_size=kernel_size, padding=padding),
                                      nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(int(in_channels//2), out_channels, kernel_size=kernel_size, padding=padding),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      )
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, int(in_channels * 0.8), kernel_size=kernel_size, padding=padding),
                                      nn.Conv2d(int(in_channels * 0.8), out_channels, kernel_size=kernel_size, padding=padding),
                                      )

    def forward(self, x):

        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, int(channels * 0.8) , kernel_size=3, padding=1),
                    CNNBlock(int(channels * 0.8), channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        prev_inputs=[]
        prev_inputs.append(x)

        for layer in self.layers:
            if self.use_residual:
                x = layer(x)
                for prev_input  in prev_inputs :
                    x = x + prev_input
                prev_inputs.append(x)
            else:
                x = layer(x)

        return x

