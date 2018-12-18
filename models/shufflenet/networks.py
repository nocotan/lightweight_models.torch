# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from models.shufflenet.modules import ShuffleBlock
from models.shufflenet.modules import Bottleneck


class ShuffleNet(nn.Module):
    def __init__(self,
                 n_classes=1000,
                 out_channels=[200, 400, 800],
                 n_blocks=[4, 8, 4],
                 groups=2):
        super(ShuffleNet, self).__init__()
        out_channels = out_channels
        n_blocks = n_blocks
        groups = groups

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], n_blocks[0], groups)
        self.layer2 = self._make_layer(out_channels[1], n_blocks[1], groups)
        self.layer3 = self._make_layer(out_channels[2], n_blocks[2], groups)
        self.linear = nn.Linear(out_channels[2], n_classes)

    def _make_layer(self, out_channels, n_blocks, groups):
        layers = []
        for i in range(n_blocks):
            stride = 2 if i == 0 else 1
            cat_channels = self.in_channels if i == 0 else 0
            layers.append(Bottleneck(self.in_channels, out_channels-cat_channels, stride=stride, groups=groups))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
