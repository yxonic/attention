import fret

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


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
        x = x.expand(*size) - 0.5

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
class OnmtEncoder(ImageEncoder):
    """A simple encoder CNN -> RNN for image src borrowed from OpenNMT-py.
    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout):
        super().__init__(rnn_size, 8)
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.layer1 = nn.Conv2d(3, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        src_size = 512
        dropout = dropout[0] if type(dropout) is list else dropout
        self.rnn = nn.LSTM(src_size, int(rnn_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.pos_lut = nn.Embedding(1000, src_size)

    def forward(self, src):
        size = list(src.size())
        size[1] = 3
        src = src.expand(*size)

        # (batch_size, 64, imgH, imgW)
        # layer 1
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)

        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        src = F.relu(self.layer2(src), True)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        src = F.relu(self.batch_norm1(self.layer3(src)), True)

        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        src = F.relu(self.layer4(src), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.relu(self.batch_norm3(self.layer6(src)), True)

        # (batch_size, 512, H, W)
        all_outputs = []
        # batch_size = src.size(0)
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            # row_vec = torch.Tensor(batch_size).type_as(inp.data) \
            #     .long().fill_(row)
            # pos_emb = self.pos_lut(row_vec)
            # with_pos = torch.cat(
            #     (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(inp)
            all_outputs.append(outputs)
        # (H x W, bs, rs) -> (bs, rs, H x W)
        out = torch.cat(all_outputs, 0).permute(1, 2, 0)
        return out.view(src.size(0), -1, src.size(2), src.size(3))


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
