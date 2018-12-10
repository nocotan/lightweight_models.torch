# -*- coding: utf-8 -*-
import math
import torch.nn as nn

from models.mobilenet_v2.modules import ConvBlock
from models.mobilenet_v2.modules import Conv1x1Block
from models.mobilenet_v2.modules import InvResBlock



class MobileNet_v2(nn.Module):
    def __init__(self, n_classes=1000, input_size=224, width_mult=1.):
        super(MobileNet_v2, self).__init__()
        in_channels = int(32 * width_mult)

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0

        self.last_channel = int(1280 * width_mult) if width_mult > 1. else 1280
        self.features = [ConvBlock(3, in_channels, 2)]
        self.features += self._make_layers(in_channels,
                                           width_mult)
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(self.last_channel, n_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _make_layers(self, in_channels, width_mult):
        layers = []
        for t, c, n, s in self.interverted_residual_setting:
            out_channels = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    layers.append(InvResBlock(in_channels,
                                              out_channels,
                                              stride=s,
                                              expand_ratio=t))
                else:
                    layers.append(InvResBlock(in_channels,
                                              out_channels,
                                              stride=1,
                                              expand_ratio=t))
                in_channels = out_channels

        layers.append(Conv1x1Block(in_channels, self.last_channel))

        return layers

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
