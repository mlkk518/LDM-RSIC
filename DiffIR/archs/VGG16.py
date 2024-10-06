import torch
from torchvision.models import vgg16_bn,VGG16_BN_Weights
import matplotlib.pyplot as plt
class VGGPerceptual(torch.nn.Module):
    def __init__(self,  Requires_grad = False):
        super(VGGPerceptual, self).__init__()
        # blocks = []
        Weights = VGG16_BN_Weights.IMAGENET1K_V1
        # Weights = False
        # blocks.append(vgg16(weights=weights).features[:4].eval())
        # blocks.append(vgg16(weights=weights).features[4:9].eval())
        # blocks.append(vgg16(weights=weights).features[9:16].eval())
        # blocks.append(vgg16(weights=weights).features[16:23].eval())
        #
        # blocks.append(vgg16_bn(weights=Weights).features[:7].eval())
        # blocks.append(vgg16_bn(weights=Weights).features[7:14].eval())
        # blocks.append(vgg16_bn(weights=Weights).features[14:24].eval())
        # blocks.append(vgg16_bn(weights=Weights).features[24:43].eval())

        self.vgg_fea = vgg16_bn(weights=Weights).features[:43]

        for param in self.vgg_fea.parameters():
            param.requires_grad = False

        # for bl in blocks:
        #     for p in bl.parameters():
        #         p.requires_grad = Requires_grad
        # self.blocks = torch.nn.ModuleList(blocks)
        # self.transform = torch.nn.functional.interpolate
        # self.resize = resize

    def forward(self, input):

        # input = (input-self.mean) / self.std
        # x = input
        # feas = []
        # for i, block in enumerate(self.blocks):
        #     x = block(x)
        #     feas.append(x)
        feas = self.vgg_fea(input)

        return feas