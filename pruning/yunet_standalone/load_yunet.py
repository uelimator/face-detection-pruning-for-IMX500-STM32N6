"""Writing my own code for once"""


from typing import Any


from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import sys


from .backbone import YuNetBackbone
from .head import YuNet_Head
from .neck import TFPN

pathToConfig = Path("./training/libfacedetection.train/configs")
modelName = Path('yunet_n.py')


pathToWeights = Path('./training/libfacedetection.train/weights')
weightsName = Path('yunet_n.pth')


"""Checking if models exists"""

"""
if (pathToConfig / modelName).exists():
    print("Model:", modelName, "found")
else:
    print("Model:", modelName, "cannot be found, stopping script")
    sys.exit()


if(pathToWeights / weightsName).exists():
    print("Weights:",weightsName, "found")
else:
    print("Weights:", weightsName, "Not Found")"""



#Replicate yunet_n.py from original Repo

# Architecture parameters extracted from configs/yunet_n.py --> Nano variant with same structure backbone, neck and head
YUNET_N_CFG: dict[str, Any] = {
    "backbone": {
        "stage_channels": [[3, 16, 16], [16, 64], [64, 64], [64, 64], [64, 64], [64, 64]],
        "downsample_idx": [0, 2, 3, 4],
        "out_idx": [3, 4, 5],
    },
    "neck": {
        "in_channels": [64, 64, 64],
        "out_idx": [0, 1, 2],
    },
    "head": {
        "num_classes": 1,
        "in_channels": 64,
        "shared_stacked_convs": 1,
        "stacked_convs": 0,
        "feat_channels": 64,
        "strides": (8, 16, 32),
        "use_kps": True,
        "kps_num": 5,
    },
}

class YuNet(nn.Module):
    """End-to-end YuNet detector. Wraps backbone + neck + head."""

    def __init__(self, cfg: dict[str, Any] = YUNET_N_CFG) -> None:
        super().__init__()
        self.backbone = YuNetBackbone(**cfg["backbone"])
        self.neck = TFPN(**cfg["neck"])
        self.bbox_head = YuNet_Head(**cfg["head"])

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.bbox_head(feats)

    @torch.no_grad()
    def load_pretrained(self, pth_path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load weights from a libfacedetection.train .pth checkpoint.

        The checkpoint typically wraps the state_dict under a 'state_dict' key;
        we handle both wrapped and bare forms.

        Returns (missing_keys, unexpected_keys) so the caller can decide whether
        any mismatches are acceptable. Training-only keys (loss buffers,
        prior_generator state) are expected to be unexpected since this
        standalone class drops those modules.
        """
        ckpt = torch.load(pth_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        return list(missing), list(unexpected)


