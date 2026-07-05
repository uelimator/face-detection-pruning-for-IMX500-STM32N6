"""TFPN — Tiny Feature Pyramid Network used by YuNet.

Original: mmdet/models/necks/tfpn.py
Changes: removed @NECKS.register_module() decorator and registry import.
"""

import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvDPUnit


class TFPN(nn.Module):
    """Top-down feature pyramid with lateral ConvDPUnits.

    For the yunet_n config: in_channels=[64, 64, 64], out_idx=[0, 1, 2].
    """

    def __init__(self, in_channels, out_idx):
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(ConvDPUnit(in_channels[i], in_channels[i], True))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        # feats is a list of three tensors (strides 8, 16, 32)
        feats = list(feats)  # don't mutate the caller's list
        num_feats = len(feats)

        # top-down flow: upsample higher-stride features and add to lower-stride
        for i in range(num_feats - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i], scale_factor=2.0, mode="nearest"
            )

        feats[0] = self.lateral_convs[0](feats[0])

        return [feats[i] for i in self.out_idx]
