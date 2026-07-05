"""Building-block layers used by YuNet's backbone, neck, and head.

Verbatim copy of libfacedetection.train/mmdet/models/utils/yunet_layer.py —
no mmcv coupling, no changes needed.
"""

import torch.nn as nn


class ConvDPUnit(nn.Module):
    """1x1 pointwise + 3x3 depthwise + (optional) BN+ReLU.

    Used everywhere in YuNet — backbone blocks, FPN laterals, head sub-convs.
    """

    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = self.relu(x)
        return x


class Conv_head(nn.Module):
    """Initial conv head with a stride-2 3x3, then a ConvDPUnit. Used as stage 0."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels, True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class Conv4layerBlock(nn.Module):
    """Two stacked ConvDPUnits — the building block for backbone stages 1+."""

    def __init__(self, in_channels, out_channels, withBNRelu=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
