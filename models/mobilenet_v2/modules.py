# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class InvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvResBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_ch = round(in_channels * expand_ratio)
        self.use_re_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_ch,
                          hidden_ch,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          groups=hidden_ch,
                          bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_ch,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels,
                          hidden_ch,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_ch,
                          hidden_ch,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          groups=hidden_ch,
                          bias=False),
                nn.BatchNorm2d(hidden_ch),
                          nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_ch,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.use_re_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)), inplace=True)


class Conv1x1Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1Block, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)), inplace=True)
