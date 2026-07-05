"""Distillation loss for finetuning a pruned YuNet against the unpruned teacher.

Reorganized 2026-05-17: switched from the simplified YuNetLoss
(center-prior + BCE + GIoU) to knowledge distillation. Reason: the
simplified loss was incompatible with how the pretrained YuNet was trained
(SimOTA dynamic matching), so finetuning destroyed the model. Distillation
sidesteps the matching problem entirely — the loss just tells the pruned
student to reproduce the unpruned teacher's outputs.

The old YuNetLoss code is preserved at the bottom of this file inside a
triple-quoted block, in case you want to revive and properly implement
SimOTA + focal loss later for from-scratch / aggressive-pruning training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DistillationLoss:
    """MSE distillation between a frozen teacher's outputs and the student's.

    The student and teacher must have identical output structure (12 tensors
    in the YuNet order: cls_8, cls_16, cls_32, obj_8, obj_16, obj_32,
    bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32). When the student is
    a structurally-pruned variant of the teacher, that's automatic — pruning
    doesn't change the output channel counts (we ignored those in the prune
    config), only the intermediate channels.

    Per-tensor weights compensate for differing magnitudes:
      - cls/obj are sigmoid-applied, in [0, 1] — small MSE values
      - bbox/kps are raw offsets, often in [-2, 3] — naturally larger MSE
    Without weighting, bbox/kps would dominate the loss.
    """

    DEFAULT_WEIGHTS = (
        1.0, 1.0, 1.0,   # cls_8,  cls_16, cls_32   (sigmoided)
        1.0, 1.0, 1.0,   # obj_8,  obj_16, obj_32   (sigmoided)
        0.1, 0.1, 0.1,   # bbox_8, bbox_16, bbox_32 (raw)
        0.1, 0.1, 0.1,   # kps_8,  kps_16, kps_32   (raw)
    )

    OUTPUT_NAMES = (
        "cls_8", "cls_16", "cls_32",
        "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32",
        "kps_8", "kps_16", "kps_32",
    )

    def __init__(self, weights: tuple[float, ...] | None = None) -> None:
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS

    def __call__(self, student_outputs: tuple[torch.Tensor, ...],
                 teacher_outputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        """Returns dict with 'total' (scalar tensor with grad) and 'per_output'
        (list of detached scalars for logging).
        """
        per_output: list[torch.Tensor] = []
        for s, t, w in zip(student_outputs, teacher_outputs, self.weights):
            per_output.append(w * F.mse_loss(s, t))
        total = torch.stack(per_output).sum()
        return {
            "total": total,
            "per_output": [p.detach() for p in per_output],
            "names": list(self.OUTPUT_NAMES),
        }


# ============================================================================
# OLD CODE BELOW (commented out via triple-quoted string) — YuNetLoss with
# center-prior matching + BCE + GIoU. Kept for reference. See module docstring
# above for why this was replaced.
# ============================================================================

"""
from dataclasses import dataclass

# Stride at which to assign a GT face based on its sqrt(w*h). Tuned so each
# stride handles roughly an octave of face sizes around its receptive field.
SIZE_BUCKETS = (
    (8,  0.0,    32.0),
    (16, 32.0,   96.0),
    (32, 96.0,   float("inf")),
)


@dataclass
class LossWeights:
    cls: float = 1.0
    obj: float = 1.0
    bbox: float = 5.0


def assign_stride(face_size: float) -> int:
    for stride, lo, hi in SIZE_BUCKETS:
        if lo <= face_size < hi:
            return stride
    return SIZE_BUCKETS[-1][0]


def decode_bbox(bbox_pred, grid_cx, grid_cy, stride):
    cx = grid_cx + bbox_pred[..., 0] * stride
    cy = grid_cy + bbox_pred[..., 1] * stride
    w = torch.exp(bbox_pred[..., 2]) * stride
    h = torch.exp(bbox_pred[..., 3]) * stride
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def giou(pred_xyxy, gt_xyxy, eps=1e-7):
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)
    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    gt_area = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = pred_area + gt_area - inter
    iou = inter / (union + eps)
    enc_x1 = torch.minimum(px1, gx1)
    enc_y1 = torch.minimum(py1, gy1)
    enc_x2 = torch.maximum(px2, gx2)
    enc_y2 = torch.maximum(py2, gy2)
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
    return iou - (enc_area - union) / (enc_area + eps)


class YuNetLoss:
    def __init__(self, input_size=320, weights=None):
        self.input_size = input_size
        self.weights = weights or LossWeights()
        self.strides = tuple(s for s, _, _ in SIZE_BUCKETS)

    def _grid(self, stride, device, dtype):
        n = self.input_size // stride
        ys, xs = torch.meshgrid(torch.arange(n, device=device, dtype=dtype),
                                torch.arange(n, device=device, dtype=dtype),
                                indexing="ij")
        cx = xs.flatten() * stride
        cy = ys.flatten() * stride
        return cx, cy

    def __call__(self, outputs, gt_boxes_per_image):
        device = outputs[0].device
        dtype = outputs[0].dtype
        cls_outs = outputs[0:3]
        obj_outs = outputs[3:6]
        bbox_outs = outputs[6:9]
        batch_size = cls_outs[0].shape[0]
        eps = 1e-7
        per_stride_loss_cls = []
        per_stride_loss_obj = []
        per_stride_loss_bbox = []
        for level, stride in enumerate(self.strides):
            cls_pred = cls_outs[level]
            obj_pred = obj_outs[level]
            bbox_pred = bbox_outs[level]
            num_cells = cls_pred.shape[1]
            grid_cx, grid_cy = self._grid(stride, device, dtype)
            cls_target = torch.zeros_like(cls_pred)
            obj_target = torch.zeros_like(obj_pred)
            pos_mask = torch.zeros(batch_size, num_cells, dtype=torch.bool, device=device)
            pos_pred_boxes = []
            pos_gt_boxes = []
            pos_grid_cx = []
            pos_grid_cy = []
            grid_per_side = self.input_size // stride
            for b in range(batch_size):
                gts = gt_boxes_per_image[b]
                if gts.numel() == 0:
                    continue
                for box in gts:
                    x, y, w, h = box.tolist()
                    face_size = (w * h) ** 0.5
                    if assign_stride(face_size) != stride:
                        continue
                    cx = x + w / 2
                    cy = y + h / 2
                    grid_x = int(cx // stride)
                    grid_y = int(cy // stride)
                    if not (0 <= grid_x < grid_per_side and 0 <= grid_y < grid_per_side):
                        continue
                    cell = grid_y * grid_per_side + grid_x
                    pos_mask[b, cell] = True
                    cls_target[b, cell, 0] = 1.0
                    obj_target[b, cell, 0] = 1.0
                    pos_pred_boxes.append(bbox_pred[b, cell])
                    pos_gt_boxes.append(torch.tensor([x, y, x + w, y + h], device=device, dtype=dtype))
                    pos_grid_cx.append(grid_cx[cell])
                    pos_grid_cy.append(grid_cy[cell])
            loss_cls = F.binary_cross_entropy(cls_pred.clamp(eps, 1 - eps), cls_target, reduction="sum")
            loss_obj = F.binary_cross_entropy(obj_pred.clamp(eps, 1 - eps), obj_target, reduction="sum")
            n_pos = pos_mask.sum().clamp(min=1).float()
            loss_cls = loss_cls / n_pos
            loss_obj = loss_obj / n_pos
            if pos_pred_boxes:
                pp = torch.stack(pos_pred_boxes, dim=0)
                gg = torch.stack(pos_gt_boxes, dim=0)
                gcx = torch.stack(pos_grid_cx, dim=0)
                gcy = torch.stack(pos_grid_cy, dim=0)
                pp_xyxy = decode_bbox(pp, gcx, gcy, stride)
                g = giou(pp_xyxy, gg)
                loss_bbox = (1.0 - g).mean()
            else:
                loss_bbox = torch.zeros((), device=device, dtype=dtype)
            per_stride_loss_cls.append(loss_cls)
            per_stride_loss_obj.append(loss_obj)
            per_stride_loss_bbox.append(loss_bbox)
        loss_cls = torch.stack(per_stride_loss_cls).mean()
        loss_obj = torch.stack(per_stride_loss_obj).mean()
        loss_bbox = torch.stack(per_stride_loss_bbox).mean()
        total = (self.weights.cls * loss_cls
                 + self.weights.obj * loss_obj
                 + self.weights.bbox * loss_bbox)
        return {
            "total": total,
            "cls": loss_cls.detach(),
            "obj": loss_obj.detach(),
            "bbox": loss_bbox.detach(),
        }
"""
