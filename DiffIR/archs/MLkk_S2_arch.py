
import DiffIR.archs.common as common
from ldm.ddpm import DDPM
import DiffIR.archs.attention as attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x, prior=None):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out

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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        # if group == 1:
        # self.ln1 = nn.Linear(embed_dim * 4, dim)
        # self.ln2 = nn.Linear(embed_dim * 4, dim)

    def forward(self, x, prior=None):
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2
            x = torch.squeeze(x)

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class FeedForward_HIM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim):
        super(FeedForward_HIM, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        # if group == 1:
        # self.kernel_K = nn.Linear(embed_dim * 4, dim)
        # self.kernel_V = nn.Linear(embed_dim * 4, dim)

    def forward(self, x, prior=None):
        b, c, h, w = x.shape
        # if prior is not None:
        #     k_v1 = self.kernel_K(prior).view(b, -1, 1, 1)
        #     k_v2 = self.kernel_V(prior).view(b, -1, 1, 1)
        #     x = (x * k_v1) + k_v2

        x = self.project_in(x)
        x1, gate = self.dwconv(x).chunk(2, dim=1)
        x = x1 * F.gelu(gate)
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # # prior
        # if group == 1:
        #     self.ln1 = nn.Linear(embed_dim * 4, dim)
        #     self.ln2 = nn.Linear(embed_dim * 4, dim)

    def forward(self, x, prior=None):
        b, c, h, w = x.shape
        # if prior is not None:
        #     k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
        #     k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
        #     x = (x * k1) + k2
        #     x = torch.squeeze(x)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# ##########################################################################
# ## Hierarchical Integration Module
# class HIM(nn.Module):
#     def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
#         super(HIM, self).__init__()
#
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
#         self.norm2 = LayerNorm_Without_Shape(embed_dim * 4, LayerNorm_type)
#
#         self.q = nn.Linear(dim, dim, bias=bias)
#         self.kv = nn.Linear(embed_dim * 4, 2 * dim, bias=bias)
#
#         self.proj = nn.Linear(dim, dim, bias=True)
#
#     def forward(self, x, prior):
#         B, C, H, W = x.shape
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         _x = self.norm1(x)
#         prior = self.norm2(prior)
#
#         q = self.q(_x)
#         kv = self.kv(prior)
#         k, v = kv.chunk(2, dim=-1)
#
#         q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
#         k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
#         v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#         out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
#         out = self.proj(out)
#
#         # sum
#         x = x + out
#         x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
#
#         return x
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 使用平均池化和最大池化的全局信息来增强通道注意力的表现
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通过一个简单的全连接层来实现注意力机制的学习
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out  # 将平均和最大池化得到的注意力特征相加
        return x * out.view(b, c, 1, 1)

class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group =4, k_dim=128, v_dim=128):
        super(HIM, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.group = group

        self.scale = num_heads ** -0.5
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_q1 = nn.Linear(dim, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(embed_dim*4, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(embed_dim*4, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, dim)

        # if group != 1:
        #     self.pool = nn.AdaptiveAvgPool2d((group, group))
        #     self.proj_x = nn.Linear(dim, embed_dim*4)
        if group != 1:
            self.conv_cat = nn.Conv2d(embed_dim*4 + dim, dim, kernel_size=1, bias=bias)
            self.channel_attention = ChannelAttention(embed_dim * 4 + dim)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, prior):
        # b, c, h, w = x.shape
        # x = rearrange(x, 'b c h w-> b (h w) c')

        # batch_size, seq_len1, in_dim1 = x.size()


        if self.group != 1:  ##  如果不等于1 ， 将x 融合到  prior 中

            prior = rearrange(prior, 'b (h w) c->b c h w', h=self.group, w=self.group)
            # prior = self.conv(prior)

            prior = torch.nn.functional.interpolate(prior, scale_factor=4)
            # prior = rearrange(prior, 'b c h w-> b (h w) c')
            fea_cat = torch.cat((x,prior), dim=1)
            fea_cat = self.channel_attention(fea_cat)
            x = self.conv_cat(fea_cat)
            # print("prior  shape", prior.shape)
            # x = upsampled_x + x

        # seq_len2 = prior.size(1)
        # q1 = self.proj_q1(x).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        # k1 = self.proj_k2(prior).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        # v1 = self.proj_v2(prior).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        #
        # attn = torch.matmul(q1, k1) / self.k_dim ** 0.5
        # attn = F.softmax(attn, dim=-1)
        # output = torch.matmul(attn, v1).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        # x = self.proj_o(output)
        # x = rearrange(x, 'b (h w) c->b c h w',h=h, w=w)


        # print(" x cross -shape", x.shape)

        ### 融合后 refine
        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        #
        # attn = (q @ k.transpose(-2, -1)) * self.temperature2
        # attn = attn.softmax(dim=-1)
        #
        # out = (attn @ v)
        #
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        #
        # out = self.project_out(out)

        return x


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class Forward_gate(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class HIM_block(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, group, ffn_expansion_factor):
        super(HIM_block, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HIM(dim, num_heads, bias, embed_dim, group)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_HIM(dim, ffn_expansion_factor, bias, embed_dim)

    def forward(self, x, prior=None):

        x = x + self.attn(self.norm1(x), prior) ## 结合进去
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, embed_dim, group)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, prior=None):
        x = x + self.attn(self.norm1(x), prior)
        x = x + self.ffn(self.norm2(x), prior)

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, num_blocks, group):

        super().__init__()
        self.group = group
        self.blocks = nn.ModuleList(
            [ResidualBottleneckBlock(dim) for i in
             range(num_blocks)])

        # # build blocks
        self.blocks = nn.ModuleList([ResidualBottleneckBlock(dim) for i in range(num_blocks)])

        # # build blocks
        # self.blocks = nn.ModuleList(
        #     [TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
        #                       bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in
        #      range(num_blocks)])

        self.him = HIM_block(dim, num_heads, bias, embed_dim, LayerNorm_type, group, ffn_expansion_factor)

    def forward(self, x, prior=None):
        if prior is not None:
            x = self.him(x, prior)
            # prior = None
        for blk in self.blocks:
            x = blk(x, None)

        return x


##########################################################################
# The implementation builds on Restormer code https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
class Transformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=True,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 embed_dim=48,
                 group=4,
                 ):

        super(Transformer, self).__init__()


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = BasicLayer(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                         num_blocks=num_blocks[0], group=1)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1],
                                         group=1)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = BasicLayer(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2],
                                         group=1)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = BasicLayer(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
                                 num_blocks=num_blocks[3], group=group)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicLayer(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2],
                                         group=1)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1],
                                         group=1)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0],
                                         group=1)

        # self.refinement = BasicLayer(dim=int(dim * 2 ** 1), num_heads=heads[0],
        #                              ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                              LayerNorm_type=LayerNorm_type, embed_dim=embed_dim,
        #                              num_blocks=num_refinement_blocks, group=group)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, prior=None):


        # multi-scale prior
        # prior_1 = prior
        # prior_2 = self.down_1(prior_1)
        # prior_3 = self.down_2(prior_2).flatten(1)

        # print("prior_1  shape", prior.shape)
        # print("prior_2  shape", prior_2.shape)
        # print("prior_3  shape", prior_3.shape)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, None)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, None)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, None)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, prior)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, None)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, None)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        # print("inp_dec_level1 shape", inp_dec_level1.shape)
        # print("out_enc_level1 shape", out_enc_level1.shape)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, None)

        # out_dec_level1 = self.refinement(out_dec_level1, None)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class CPEN_mlkk(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6, group= 4):
        super(CPEN_mlkk, self).__init__()
        E1=[nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]

        E3 = [  ## 进行一次下采样
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((group, group)),
            ]

        E=E1+E2 + E3
        self.E = nn.Sequential(
            *E
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
    def forward(self, x,gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)
        x = self.E(x)

        S1_IPR = []
        x = rearrange(x, 'b c h w-> b (h w) c')
        fea = self.mlp1(x)
        S1_IPR.append(x)

        return fea, S1_IPR

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        
        fea = self.resmlp(c)

        return fea 

@ARCH_REGISTRY.register()
class DiffIR_mlkkS2(nn.Module):
    def __init__(self,         
        n_encoder_res=6,         
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        n_denoise_res = 1, 
        linear_start= 0.1,
        linear_end= 0.99, 
        timesteps = 4 ):
        super(DiffIR_mlkkS2, self).__init__()
        group = 8
        embed_dim = 64
        # Generator
        self.G = Transformer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,  ## Other option 'BiasFree'
            group=group,
            embed_dim=embed_dim,
        )

        self.condition = CPEN_mlkk(n_feats=64, n_encoder_res=n_encoder_res, group=group)

        self.denoise= denoise(n_feats=64, n_denoise_res=n_denoise_res,timesteps=timesteps)

        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition ,n_feats=64,linear_start= linear_start, linear_end= linear_end, timesteps = timesteps)


    def forward(self, img, IPRS1=None):
        if self.training:
            IPRS2, pred_IPR_list=self.diffusion(img,IPRS1)
            print("img  shape", img.shape)
            print("len pred_IPR_list  ", len(pred_IPR_list))
            sr = self.G(img, IPRS2)
            return sr, pred_IPR_list
        else:
            IPRS2=self.diffusion(img)
            sr = self.G(img, IPRS2)
            return sr
