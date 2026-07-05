"""YuNet detection head — inference-only standalone version.

Adapted from libfacedetection.train/mmdet/models/dense_heads/yunet_head.py.
Stripped:
  - BaseDenseHead / BBoxTestMixin inheritance (only nn.Module needed)
  - All mmcv coupling: build_loss, build_assigner, build_sampler,
    build_prior_generator, force_fp32, batched_nms
  - Training methods: forward_train, loss, _get_target_single, get_bboxes,
    _bbox_decode, _kps_decode, _kps_encode, _bboxes_nms
  - The branching on torch.onnx.is_in_onnx_export() — we always run the
    "export-style" forward which produces the 12 named output tensors.

The forward() returns sigmoided cls/obj (matching the ONNX) so numerical
comparison is direct. For finetuning later, you'd modify the loss to expect
sigmoided inputs, OR add a `return_logits` flag and route training through
the unsigmoided path.
"""

import torch
import torch.nn as nn

from .layers import ConvDPUnit


class YuNet_Head(nn.Module):
    """Multi-scale, multi-task head: cls + obj + bbox + kps per stride.

    For yunet_n config:
        num_classes=1, in_channels=64, feat_channels=64,
        shared_stacked_convs=1, stacked_convs=0, use_kps=True, kps_num=5,
        strides=[8, 16, 32]
    """

    def __init__(self, num_classes, in_channels, feat_channels=256,
                 shared_stacked_convs=2, stacked_convs=2, strides=(8, 16, 32),
                 use_kps=False, kps_num=5):
        super().__init__()
        self.num_classes = num_classes
        self.NK = kps_num
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_kps = use_kps
        self.shared_stack_convs = shared_stacked_convs
        self.strides = tuple(strides)
        self.strides_num = len(self.strides)

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        if self.shared_stack_convs > 0:
            self.multi_level_share_convs = nn.ModuleList()
        if self.stacked_convs > 0:
            self.multi_level_cls_convs = nn.ModuleList()
            self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_bbox = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        if self.use_kps:
            self.multi_level_kps = nn.ModuleList()

        for _ in self.strides:
            if self.shared_stack_convs > 0:
                single_level_share_convs = []
                for i in range(self.shared_stack_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    single_level_share_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_share_convs.append(nn.Sequential(*single_level_share_convs))

            if self.stacked_convs > 0:
                single_level_cls_convs = []
                single_level_reg_convs = []
                for i in range(self.stacked_convs):
                    chn = self.in_channels if (i == 0 and self.shared_stack_convs == 0) else self.feat_channels
                    single_level_cls_convs.append(ConvDPUnit(chn, self.feat_channels))
                    single_level_reg_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_reg_convs.append(nn.Sequential(*single_level_reg_convs))
                self.multi_level_cls_convs.append(nn.Sequential(*single_level_cls_convs))

            chn = self.in_channels if (self.stacked_convs == 0 and self.shared_stack_convs == 0) else self.feat_channels
            self.multi_level_cls.append(ConvDPUnit(chn, self.num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(chn, 4, False))
            if self.use_kps:
                self.multi_level_kps.append(ConvDPUnit(chn, self.NK * 2, False))
            self.multi_level_obj.append(ConvDPUnit(chn, 1, False))

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
        """Always uses the 'ONNX-export' shape: per-stride flattened to (B, H*W, C).

        Returns a flat tuple of 12 tensors in the canonical order:
            cls_8, cls_16, cls_32,
            obj_8, obj_16, obj_32,
            bbox_8, bbox_16, bbox_32,
            kps_8, kps_16, kps_32
        where cls and obj are sigmoid-applied (matching the libfacedetection.train ONNX).
        """
        if self.shared_stack_convs > 0:
            feats = [convs(feat) for feat, convs in zip(feats, self.multi_level_share_convs)]

        if self.stacked_convs > 0:
            feats_cls, feats_reg = [], []
            for i in range(self.strides_num):
                feats_cls.append(self.multi_level_cls_convs[i](feats[i]))
                feats_reg.append(self.multi_level_reg_convs[i](feats[i]))
            cls_preds = [c(f) for f, c in zip(feats_cls, self.multi_level_cls)]
            bbox_preds = [c(f) for f, c in zip(feats_reg, self.multi_level_bbox)]
            obj_preds = [c(f) for f, c in zip(feats_reg, self.multi_level_obj)]
            kps_preds = [c(f) for f, c in zip(feats_reg, self.multi_level_kps)] if self.use_kps else None
        else:
            cls_preds = [c(f) for f, c in zip(feats, self.multi_level_cls)]
            bbox_preds = [c(f) for f, c in zip(feats, self.multi_level_bbox)]
            obj_preds = [c(f) for f, c in zip(feats, self.multi_level_obj)]
            kps_preds = [c(f) for f, c in zip(feats, self.multi_level_kps)] if self.use_kps else None

        # Reshape to (B, H*W, C) and sigmoid cls / obj — matches the ONNX export branch.
        cls_out = [
            f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, self.num_classes).sigmoid()
            for f in cls_preds
        ]
        obj_out = [
            f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, 1).sigmoid()
            for f in obj_preds
        ]
        bbox_out = [
            f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, 4)
            for f in bbox_preds
        ]
        kps_out = [
            f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, self.NK * 2)
            for f in (kps_preds if kps_preds is not None else [])
        ]

        # Return as a flat tuple, ordered to match the ONNX output names.
        if self.use_kps:
            return (*cls_out, *obj_out, *bbox_out, *kps_out)
        return (*cls_out, *obj_out, *bbox_out)
