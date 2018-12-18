# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from models.shufflenet_v2.modules import BasicBlock
from models.shufflenet_v2.modules import DownBlock


class ShuffleNet_v2(nn.Module):
    def __init__(self,
                 n_classes=1000,
                 out_channels=[48, 96, 192, 1024],
                 n_blocks=[3, 7, 3],
                 groups=3):
        super(ShuffleNet_v2, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], n_blocks[0], groups)
        self.layer2 = self._make_layer(out_channels[1], n_blocks[1], groups)
        self.layer3 = self._make_layer(out_channels[2], n_blocks[2], groups)
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], 10)

    def _make_layer(self, out_channels, n_blocks, groups):
        layers = [DownBlock(self.in_channels, out_channels, groups=groups)]
        for i in range(n_blocks):
            layers.append(BasicBlock(out_channels, groups=groups))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

