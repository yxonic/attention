import fret

import torch
import torch.nn as nn
from torchvision.models import vgg11_bn, resnet18


class ImageEncoder(nn.Module):
    def __init__(self, out_dim, factor):
        super().__init__()
        self.out_dim = out_dim
        self.factor = factor
        self.model = None

    def forward(self, x):
        """
        Inputs: img (batch, channel, H, W)
        Outputs: fea (batch, out_dim, H // factor, W // factor)
        """
        # dulicate channel for grayscale images
        size = list(x.size())
        size[1] = 3
        x = x.expand(*size)

        return self.model(x)


@fret.configurable
class VGGEncoder(ImageEncoder):
    def __init__(self, out_dim=128):
        super().__init__(out_dim, 8)
        cfg = [8, 16, 'M', 32, 32, 'M', 64, out_dim, 'M']
        self.model = make_layers(cfg, batch_norm=True)


@fret.configurable
class ResNetEncoder(ImageEncoder):
    def __init__(self):
        super().__init__(128, 8)
        res = resnet18()
        self.model = nn.Sequential(*list(res.children())[:-4])


@fret.configurable
class FullyConvEncoder(ImageEncoder):
    def __init__(self, out_dim=128):
        super().__init__(out_dim, 1)


# from torchvision/models/vgg.py
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
