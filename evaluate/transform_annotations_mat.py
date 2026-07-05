"""Generate 320x320-space .mat ground-truth files for the WIDER eval toolkit.

Reads the original four .mat files (wider_face_val.mat + the three difficulty
splits) and writes transformed copies into --output-dir. Path B preprocessing
is applied to every bounding box: center-crop to min(H, W), resize to
--input-size, drop faces entirely outside the crop, clip partial ones, drop
faces below --min-side after resize. The difficulty (easy/medium/hard) indices
are remapped to point into the surviving faces.

The output files are drop-in replacements for the originals — the official
WiderFace-Evaluation-master toolkit can be pointed at --gt <output-dir> with
no other changes.

Original locations (the script reads from --val-mat and --difficulty-dir):
    /Users/.../wider_face_split/wider_face_val.mat
    /Users/.../ground_truth/wider_{easy,medium,hard}_val.mat

Example:
    python transform_annotations_mat.py \\
        --wider-root ~/Documents/Projects/Datasets/Wider_Faces/WIDER_val \\
        --val-mat    ~/Documents/Projects/Datasets/Wider_Faces/wider_face_split/wider_face_val.mat \\
        --difficulty-dir ~/Documents/Projects/Datasets/Wider_Faces/ground_truth \\
        --output-dir ~/Documents/Projects/Datasets/Wider_Faces/wider_face_split_320x320
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat, savemat


def build_object_array(items: list) -> np.ndarray:
    """Pack a Python list into a (N, 1) numpy object array (MATLAB cell-array shape)."""
    arr = np.empty((len(items), 1), dtype=object)
    for i, item in enumerate(items):
        arr[i, 0] = item
    return arr


def transform_face_row(box: np.ndarray, left: int, top: int, side: int,
                       scale: float, min_side: float) -> np.ndarray | None:
    """Return transformed [x, y, w, h] or None if dropped."""
    x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x -= left; y -= top
    x2, y2 = x + w, y + h
    cx1 = max(0.0, x); cy1 = max(0.0, y)
    cx2 = min(float(side), x2); cy2 = min(float(side), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    nx, ny = cx1 * scale, cy1 * scale
    nw, nh = (cx2 - cx1) * scale, (cy2 - cy1) * scale
    if nw < min_side or nh < min_side:
        return None
    return np.array([nx, ny, nw, nh], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wider-root", type=Path, required=True,
                        help="Path to WIDER_val (containing images/)")
    parser.add_argument("--val-mat", type=Path, required=True,
                        help="Original wider_face_val.mat")
    parser.add_argument("--difficulty-dir", type=Path, required=True,
                        help="Directory containing wider_{easy,medium,hard}_val.mat")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the four transformed .mat files")
    parser.add_argument("--input-size", type=int, default=320,
                        help="Target square side (must match run_yunet_wider.py preprocessing)")
    parser.add_argument("--min-side", type=float, default=2.0,
                        help="Drop boxes with width or height < this after resize")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.val_mat.name} ...")
    val = loadmat(str(args.val_mat))
    event_list = val["event_list"]
    file_list = val["file_list"]
    face_bbx_list = val["face_bbx_list"]
    attr_keys = ["blur_label_list", "expression_label_list", "illumination_label_list",
                 "invalid_label_list", "occlusion_label_list", "pose_label_list"]
    attr_lists = {k: val[k] for k in attr_keys}

    n_events = event_list.shape[0]

    # Per-event, per-image remapping table: old_face_idx (0-based) -> new_face_idx (0-based)
    # or -1 if dropped. Used later when rewriting difficulty index lists.
    remap_per_event: list[list[np.ndarray]] = []

    new_face_bbx_per_event: list[np.ndarray] = []
    new_attrs_per_event: dict[str, list[np.ndarray]] = {k: [] for k in attr_keys}

    total_in = 0
    total_kept = 0
    total_outside = 0
    total_small = 0
    total_unreadable = 0

    for i in range(n_events):
        event_name = str(event_list[i, 0][0])
        img_inner = file_list[i, 0]
        bbx_inner = face_bbx_list[i, 0]
        n_imgs = img_inner.shape[0]

        new_bbx_inner = np.empty((n_imgs, 1), dtype=object)
        new_attrs_inner = {k: np.empty((n_imgs, 1), dtype=object) for k in attr_keys}
        remap_inner: list[np.ndarray] = []

        for j in range(n_imgs):
            stem = str(img_inner[j, 0][0])
            img_path = args.wider_root / "images" / event_name / f"{stem}.jpg"
            image = cv2.imread(str(img_path))
            if image is None:
                total_unreadable += 1
                # Treat as empty; nothing to keep.
                new_bbx_inner[j, 0] = np.empty((0, 4), dtype=np.float64)
                for k in attr_keys:
                    new_attrs_inner[k][j, 0] = np.empty((0, 1), dtype=np.uint8)
                remap_inner.append(np.full(bbx_inner[j, 0].shape[0], -1, dtype=np.int32))
                continue

            h, w = image.shape[:2]
            side = min(h, w)
            top = (h - side) // 2
            left = (w - side) // 2
            scale = args.input_size / side

            boxes = bbx_inner[j, 0]
            n_faces = boxes.shape[0]
            kept_boxes: list[np.ndarray] = []
            kept_attrs: dict[str, list[int]] = {k: [] for k in attr_keys}
            remap = np.full(n_faces, -1, dtype=np.int32)

            for k in range(n_faces):
                tb = transform_face_row(boxes[k], left, top, side, scale, args.min_side)
                total_in += 1
                if tb is None:
                    if boxes[k, 2] == 0 or boxes[k, 3] == 0:
                        total_small += 1
                    else:
                        # Heuristic: if any part of the box was inside the original crop
                        # window we count it as "small" (it got clipped to nothing or
                        # below the min-side threshold). Otherwise it was outside.
                        x, y, ww, hh = boxes[k]
                        x_in_crop = (x + ww > left) and (x < left + side)
                        y_in_crop = (y + hh > top) and (y < top + side)
                        if x_in_crop and y_in_crop:
                            total_small += 1
                        else:
                            total_outside += 1
                    continue
                remap[k] = len(kept_boxes)
                kept_boxes.append(tb)
                for attr_key in attr_keys:
                    attr_val = attr_lists[attr_key][i, 0][j, 0]
                    if k < attr_val.shape[0]:
                        kept_attrs[attr_key].append(int(attr_val[k, 0]))
                    else:
                        kept_attrs[attr_key].append(0)

            total_kept += len(kept_boxes)
            new_bbx_inner[j, 0] = (np.stack(kept_boxes, axis=0) if kept_boxes
                                   else np.empty((0, 4), dtype=np.float64))
            for attr_key in attr_keys:
                vals = kept_attrs[attr_key]
                new_attrs_inner[attr_key][j, 0] = (
                    np.array(vals, dtype=np.uint8).reshape(-1, 1) if vals
                    else np.empty((0, 1), dtype=np.uint8)
                )
            remap_inner.append(remap)

        new_face_bbx_per_event.append(new_bbx_inner)
        for k in attr_keys:
            new_attrs_per_event[k].append(new_attrs_inner[k])
        remap_per_event.append(remap_inner)

        if (i + 1) % 10 == 0 or i + 1 == n_events:
            print(f"  event {i + 1}/{n_events}  ({event_name})")

    # Pack everything into the cell-of-cell shape MATLAB expects
    new_face_bbx_mat = build_object_array(new_face_bbx_per_event)
    new_attrs_mat = {k: build_object_array(new_attrs_per_event[k]) for k in attr_keys}

    out_val_mat = args.output_dir / "wider_face_val.mat"
    print(f"\nSaving {out_val_mat.name} ...")
    savemat(str(out_val_mat), {
        "event_list": event_list,
        "file_list": file_list,
        "face_bbx_list": new_face_bbx_mat,
        **new_attrs_mat,
    }, do_compression=True, oned_as="column")

    # Rewrite each difficulty .mat: filter and remap 1-based indices.
    for split in ("easy", "medium", "hard"):
        src = args.difficulty_dir / f"wider_{split}_val.mat"
        print(f"\nProcessing {src.name} ...")
        diff = loadmat(str(src))
        gt_list = diff["gt_list"]
        new_gt_per_event: list[np.ndarray] = []

        for i in range(n_events):
            inner = gt_list[i, 0]
            n_imgs = inner.shape[0]
            new_inner = np.empty((n_imgs, 1), dtype=object)
            for j in range(n_imgs):
                old_idx = inner[j, 0].flatten().astype(np.int64)  # 1-based
                if old_idx.size == 0:
                    new_inner[j, 0] = np.empty((0, 1), dtype=np.uint16)
                    continue
                remap = remap_per_event[i][j]
                # old_idx is 1-based; convert to 0-based, look up new, drop -1
                new_0based = remap[old_idx - 1]
                kept = new_0based[new_0based >= 0]
                new_inner[j, 0] = (kept.astype(np.uint16).reshape(-1, 1) + 1
                                   if kept.size else np.empty((0, 1), dtype=np.uint16))
            new_gt_per_event.append(new_inner)

        out_diff_mat = args.output_dir / f"wider_{split}_val.mat"
        savemat(str(out_diff_mat), {
            "event_list": event_list,
            "file_list": file_list,
            "face_bbx_list": new_face_bbx_mat,
            "gt_list": build_object_array(new_gt_per_event),
            **new_attrs_mat,
        }, do_compression=True, oned_as="column")
        print(f"  wrote {out_diff_mat}")

    print(f"\nDone.")
    print(f"  Faces in:      {total_in}")
    print(f"  Faces kept:    {total_kept}")
    print(f"  Dropped (outside crop): {total_outside}")
    print(f"  Dropped (< {args.min_side}px or zero-sized): {total_small}")
    if total_unreadable:
        print(f"  Unreadable images: {total_unreadable}", file=sys.stderr)
    print(f"\nPoint the eval toolkit at:")
    print(f"  --gt {args.output_dir}")


if __name__ == "__main__":
    main()
