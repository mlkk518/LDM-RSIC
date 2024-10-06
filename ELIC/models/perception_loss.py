import torch
from torchvision.models import vgg16_bn,VGG16_BN_Weights
import matplotlib.pyplot as plt
class VGGPerceptual(torch.nn.Module):
    def __init__(self,  Requires_grad = False):
        super(VGGPerceptual, self).__init__()
        blocks = []
        Weights = VGG16_BN_Weights.IMAGENET1K_V1
        Weights = False
        # blocks.append(vgg16(weights=weights).features[:4].eval())
        # blocks.append(vgg16(weights=weights).features[4:9].eval())
        # blocks.append(vgg16(weights=weights).features[9:16].eval())
        # blocks.append(vgg16(weights=weights).features[16:23].eval())

        blocks.append(vgg16_bn(weights=Weights).features[:7].eval())
        blocks.append(vgg16_bn(weights=Weights).features[7:14].eval())
        blocks.append(vgg16_bn(weights=Weights).features[14:24].eval())
        blocks.append(vgg16_bn(weights=Weights).features[24:43].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = Requires_grad
        self.blocks = torch.nn.ModuleList(blocks)
        # self.transform = torch.nn.functional.interpolate
        # self.resize = resize

    def forward(self, input):

        # input = (input-self.mean) / self.std
        x = input
        feas = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            feas.append(x)

        return feas

        # loss = 0.0
        # x = input
        # for i, block in enumerate(self.blocks):
        #     x = block(x)
        #     y = block(y)
        #     if i in feature_layers:
        #         loss += torch.nn.functional.l1_loss(x, y)
        #     if i in style_layers:
        #         act_x = x.reshape(x.shape[0], x.shape[1], -1)
        #         act_y = y.reshape(y.shape[0], y.shape[1], -1)
        #         gram_x = act_x @ act_x.permute(0, 2, 1)
        #         gram_y = act_y @ act_y.permute(0, 2, 1)
        #         loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        # return loss

def visualize_feature_map(feature_map, title):
    plt.imshow(feature_map, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    data  = torch.randn((2, 3, 256, 256))

    Network1  = VGGPerceptual()
    # Network2  = VGGPerceptual()


    # Initialize the network
    Network1 = VGGPerceptual()

    out = Network1(data)

    # net  = vgg16_bn()
    # print("net", net)

    for i  in range(len(out)):
        print("i == ", i)
        print("Out shape", out[i].shape)

    # Plot the feature maps after each block
    for i in range(len(out)):
        print("Block", i + 1)
        visualize_feature_map(out[i][0, 0].cpu().detach().numpy(), f'Block {i + 1} Feature Map')
