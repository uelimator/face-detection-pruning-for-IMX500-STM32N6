"""WIDER FACE train-set dataloader for YuNet finetuning.

Reads the *transformed* annotations file (boxes already in 320x320-cropped
coordinate space) plus the matching original images, applies Path B
preprocessing (center-crop to square + resize), returns (image, boxes) pairs.

Augmentation: horizontal flip is the only one applied — for finetuning after
pruning we want gentle augmentation, not aggressive distortion that
recapitulates from-scratch training. Skipping color jitter / mosaic / etc.

Annotations format (from transform_annotations.py):
    <event>/<image>.jpg
    <num_faces>
    x y w h <6 attribute flags>
    ...
The 4th attribute flag is "invalid" — we skip those faces entirely (they
don't contribute to training).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WiderSample:
    image: torch.Tensor       # (3, H, W) float32 BGR 0-255
    boxes: torch.Tensor       # (N, 4) float32 — x, y, w, h in 320x320 space
    image_path: str           # for debugging


def parse_blocks(path: Path):
    """Yield (image_relpath, list_of_boxes) from a WIDER-format .txt file.

    Boxes that are flagged 'invalid' (attr index 3) are dropped.
    """
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        rel = lines[i].strip()
        if not rel:
            i += 1
            continue
        i += 1
        n = int(lines[i].strip())
        i += 1
        if n == 0:
            i += 1  # placeholder "0 0 0 0 ..."
            yield rel, []
            continue
        boxes = []
        for k in range(n):
            parts = lines[i + k].split()
            x, y, w, h = (float(v) for v in parts[:4])
            invalid = (len(parts) >= 8 and parts[7] == "1")
            if invalid or w <= 0 or h <= 0:
                continue
            boxes.append([x, y, w, h])
        i += n
        yield rel, boxes


def center_crop_to_square_and_resize(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


class WiderFaceTrain(Dataset):
    """WIDER train dataset returning Path-B-preprocessed images + boxes in 320x320 space."""

    def __init__(self, wider_root: Path, annotations: Path, input_size: int = 320,
                 horizontal_flip: bool = True, skip_empty: bool = True) -> None:
        self.wider_root = Path(wider_root)
        self.input_size = input_size
        self.horizontal_flip = horizontal_flip

        self.samples: list[tuple[str, list[list[float]]]] = []
        for rel, boxes in parse_blocks(Path(annotations)):
            if skip_empty and not boxes:
                continue
            self.samples.append((rel, boxes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> WiderSample:
        rel, raw_boxes = self.samples[idx]
        img_path = self.wider_root / "images" / rel
        image = cv2.imread(str(img_path))
        if image is None:
            # Return an empty sample rather than crash — the collator handles N=0
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            boxes_arr = np.empty((0, 4), dtype=np.float32)
        else:
            image = center_crop_to_square_and_resize(image, self.input_size)
            boxes_arr = np.array(raw_boxes, dtype=np.float32) if raw_boxes else np.empty((0, 4), dtype=np.float32)

        if self.horizontal_flip and np.random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            if len(boxes_arr) > 0:
                # flip x: new_x = W - x - w  (w/h unchanged)
                boxes_arr[:, 0] = self.input_size - boxes_arr[:, 0] - boxes_arr[:, 2]

        # cv2.dnn.blobFromImage produces (1, 3, H, W). We strip the batch dim
        # because the DataLoader will re-batch.
        blob = cv2.dnn.blobFromImage(image)[0]  # (3, H, W) float32 BGR 0-255

        return WiderSample(
            image=torch.from_numpy(blob),
            boxes=torch.from_numpy(boxes_arr),
            image_path=str(img_path),
        )


def collate_fn(samples: list[WiderSample]) -> dict:
    """Stack images into (B, 3, H, W); keep boxes as a list since N varies per image."""
    return {
        "images": torch.stack([s.image for s in samples], dim=0),
        "boxes": [s.boxes for s in samples],
        "paths": [s.image_path for s in samples],
    }
