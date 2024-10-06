###  使用设计的网络对压缩后的图像进行微调。
import torch
import  torch.nn as nn
from models.perception_loss import VGGPerceptual
from utils.Fusion_fun import Fuse_module, Up, Up_last
# import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
from einops.layers.torch import Rearrange
import numbers

class VGGG16_Net(nn.Module):
    def __init__(self, Requires_grad= False):
        super().__init__()

        self.VGG16_tea = VGGPerceptual(Requires_grad=Requires_grad)

    def forward(self, x):
        x = self.VGG16_tea(x)
        return x


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

# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)

##########################################################################
##refer to https://github.com/zhengchen1999/HI-Diff/blob/main/hi_diff/archs/Transformer_arch.py

## Refer to
## Hierarchical Integration Module
class MLKK_HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
        super(MLKK_HIM, self).__init__()

        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1)) # mlkk
        # self.scale =  qk_scale or head_dim ** -0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim * 4, LayerNorm_type)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(embed_dim * 4, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Linear(dim, dim, bias=bias)
        # self.kv = nn.Linear(embed_dim * 4, 2 * dim, bias=bias)
        #
        # self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)

        q = self.q(_x)
        kv = self.kv(prior)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x




#  融合第一阶段
class Fea_FusNet(nn.Module):
    def __init__(self, Requires_grad=True):
        super().__init__()

        self.VGG16_Final = VGGPerceptual(Requires_grad=Requires_grad)

        n_fea = [64, 128, 256, 512]

        self.Fusers = []

        self.fuser1 = Fuse_module(ch=n_fea[0], reduction_ratio=1)
        self.fuser2 = Fuse_module(ch=n_fea[1], reduction_ratio=2)
        self.fuser3 = Fuse_module(ch=n_fea[2], reduction_ratio=4)
        self.fuser4 = Fuse_module(ch=n_fea[3], reduction_ratio=6, Last=True)

        self.up1 = Up(n_fea[3], n_fea[2], bilinear=False)
        self.up2 = Up(n_fea[2], n_fea[1], bilinear=False)
        self.up3 = Up(n_fea[1], n_fea[0], bilinear=False)
        self.up4 = Up_last(n_fea[0], 3, bilinear=False)

        self.conv0 = nn.Conv2d(3, n_fea[0]//2, kernel_size=1)
        self.conv = nn.Conv2d(32, 3, kernel_size=1)


    def forward(self, x0, y):  ##  x  输入的原始压缩图像的解压图像； y VGGG16_student  学习到的多尺度特征

        identify = x0
        x = self.VGG16_Final(x0)

        x1, out1 = self.fuser1(x[0], y[0])
        x2, out2 = self.fuser2(x[1], y[1], x1)
        x3, out3 = self.fuser3(x[2], y[2], x2)
        x4, out4 = self.fuser4(x[3], y[3], x3)

        z1 = self.up1(x4, out3)
        z2 = self.up2(z1, out2)
        z3 = self.up3(z2, out1)
        out = self.up4(z3) + identify

        # print(" out - identify ", out)

        return out

def visualize_feature_map(feature_map, title):
    plt.imshow(feature_map, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as transforms



    inputs = torch.randn((4,3, 256,256))
    image_path = './images/23217.png'
    input_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    chechpoint_stu = "./models/checkpoint_best_loss.pth.tar"

    input_image = transform(input_image).unsqueeze(0)

    model1 = VGGG16_Net()

    model1.load_state_dict(torch.load(chechpoint_stu))
    model1.eval()


    out  = model1(inputs)


    for i  in range(len(out)):
        print("i == ", i)
        print("Out shape", out[i].shape)

    # Plot the feature maps after each block
    # for i in range(len(out)):
    #     print("Block", i + 1)
    #     visualize_feature_map(out[i][0, 0].cpu().detach().numpy(), f'Block {i + 1} Feature Map')

    model2 = Fea_FusNet()
    out = model2(inputs, out)

    print("model2", model2)

    print(" out shape", out.shape)



