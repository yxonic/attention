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
    """Encoder that resembles VGG architecture."""
    def __init__(self, out_dim=128):
        super().__init__(out_dim, 8)
        cfg = [8, 16, 'M', 32, 32, 'M', 64, out_dim, 'M']
        self.model = make_layers(cfg, batch_norm=True)


@fret.configurable
class ResNetEncoder(ImageEncoder):
    """Encoder that resembles ResNet architecture."""
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
class DarknetEncoder(ImageEncoder):
    """Darknet from YOLOv3."""
    def __init__(self):
        super().__init__(256, 8)
        import os
        import sh
        if not os.path.exists('data/yolov3.cfg'):
            os.makedirs('data', exist_ok=True)
            sh.wget('-O', 'data/yolov3.cfg',
                    'https://raw.githubusercontent.com/'
                    'pjreddie/darknet/master/cfg/yolov3.cfg')
        self.blocks = parse_cfg('data/yolov3.cfg')
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x):
        size = list(x.size())
        size[1] = 3
        x = x.expand(*size)

        modules = self.blocks[1:38]
        outputs = {}
        # write = 0
        for i, module in enumerate(modules):
            module_type = (module['type'])

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers.split(',')]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                break
            #     anchors = self.module_list[i][0].anchors
            #     # Get the input dimensions
            #     inp_dim = int (self.net_info["height"])
            #     # Get the number of classes
            #     num_classes = int (module["classes"])
            #     # Transform
            #     x = x.data
            #     x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
            #     if not write:
            #         detections = x
            #         write = 1
            #     else:
            #         detections = torch.cat((detections, x), 1)
            outputs[i] = x

        return x


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


# from https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
def parse_cfg(cfgfile):
    """Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except KeyError:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size,
                             stride, pad, bias=bias)
            module.add_module(f'conv_{index}', conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)

            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{index}', activn)

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module(f'upsample_{index}', upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except IndexError:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module(f'route_{index}', route)
            if end < 0:
                filters = output_filters[index + start] + \
                    output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}', shortcut)

        elif x['type'] == 'yolo':
            # stop here
            break
            # mask = x['mask'].split(',')
            # mask = [int(x) for x in mask]

            # anchors = x['anchors'].split(',')
            # anchors = [int(a) for a in anchors]
            # anchors = [(anchors[i], anchors[i+1])
            #            for i in range(0, len(anchors), 2)]
            # anchors = [anchors[i] for i in mask]

            # detection = DetectionLayer(anchors)
            # module.add_module(f'Detection_{index}', detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()


# class DetectionLayer(nn.Module):
#     def __init__(self, anchors):
#         super().__init__()
#         self.anchors = anchors
