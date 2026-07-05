"""Visualise the WIDER FACE -> 320x320 dataset transform.

Loads one image, draws its original ground-truth boxes on the full-resolution
source image, then applies the same Path-B preprocessing your training pipeline
uses (center crop to square -> resize to 320x320) and draws the rescaled boxes
on the result. Saves three PNGs:

    <out>/<stem>_original.png        full-resolution image + original boxes
    <out>/<stem>_320.png             center-cropped & resized + rescaled boxes
    <out>/<stem>_side_by_side.png    both above in one figure (for slides/paper)

Usage:
    python plot_wider_transform.py
    python plot_wider_transform.py --image "0--Parade/0_Parade_marchingband_1_799.jpg"
    python plot_wider_transform.py --split val
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# WIDER FACE dataset root — override via the WIDER_ROOT env var.
WIDER_ROOT = Path(os.environ.get("WIDER_ROOT",
                                 Path.home() / "Datasets" / "Wider_Faces"))

# Copied (not imported) from pruning/finetune/dataset.py so this script has no
# torch dependency — purely matplotlib + opencv.
def parse_blocks(path: Path):
    """Yield (image_relpath, list_of_boxes) from a WIDER-format .txt file."""
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


def center_crop_to_square_and_resize(image, target: int):
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


def draw_boxes(ax, boxes, color="lime", linewidth=2):
    for x, y, w, h in boxes:
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=linewidth,
                                       edgecolor=color, facecolor="none"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="0--Parade/0_Parade_marchingband_1_799.jpg",
                    help="WIDER image rel path, e.g. '0--Parade/0_Parade_marchingband_1_799.jpg'")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--target", type=int, default=320,
                    help="Crop/resize target side length (default 320).")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "wider_transform_demo",
                    help="Output dir for the PNGs.")
    args = ap.parse_args()

    if args.split == "train":
        orig_anno = WIDER_ROOT / "wider_face_split/wider_face_train_bbx_gt.txt"
        re_anno   = WIDER_ROOT / f"wider_face_split/wider_face_train_bbx_gt_{args.target}x{args.target}.txt"
        img_root  = WIDER_ROOT / "WIDER_train/images"
    else:
        orig_anno = WIDER_ROOT / "wider_face_split/wider_face_val_bbx_gt.txt"
        re_anno   = WIDER_ROOT / f"wider_face_split/wider_face_val_bbx_gt_{args.target}x{args.target}.txt"
        img_root  = WIDER_ROOT / "WIDER_val/images"

    # Sanity-check the input files
    for p in (orig_anno, re_anno, img_root):
        if not p.exists():
            sys.exit(f"missing: {p}")

    orig_boxes_map = dict(parse_blocks(orig_anno))
    re_boxes_map   = dict(parse_blocks(re_anno))

    if args.image not in orig_boxes_map:
        sys.exit(f"image not in original annotations ({orig_anno.name}): {args.image}")
    if args.image not in re_boxes_map:
        sys.exit(f"image not in {args.target}x{args.target} annotations ({re_anno.name}): {args.image}")

    img_path = img_root / args.image
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"could not read image: {img_path}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply the same Path-B transform the training pipeline uses
    img_t = center_crop_to_square_and_resize(img, args.target)
    img_t_rgb = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)

    orig_boxes = orig_boxes_map[args.image]
    re_boxes = re_boxes_map[args.image]

    args.out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # --- 1) original ---
    aspect = h / w
    fig, ax = plt.subplots(figsize=(9, 9 * aspect))
    ax.imshow(img_rgb)
    draw_boxes(ax, orig_boxes, linewidth=max(1.5, w / 500))
    ax.set_title(f"Original WIDER  ({w}×{h})  —  {len(orig_boxes)} faces", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    p1 = args.out / f"{stem}_original.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}")

    # --- 2) 320x320 ---
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(img_t_rgb)
    draw_boxes(ax, re_boxes, linewidth=1.6)
    ax.set_title(
        f"Center-crop + resize → {args.target}×{args.target}  —  {len(re_boxes)} faces",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    p2 = args.out / f"{stem}_{args.target}.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p2}")

    # --- 3) side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    axes[0].imshow(img_rgb)
    draw_boxes(axes[0], orig_boxes, linewidth=max(1.5, w / 500))
    axes[0].set_title(f"Original ({w}×{h}) — {len(orig_boxes)} faces", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(img_t_rgb)
    draw_boxes(axes[1], re_boxes, linewidth=1.6)
    axes[1].set_title(
        f"Center-crop + resize → {args.target}×{args.target} — {len(re_boxes)} faces",
        fontsize=11,
    )
    axes[1].axis("off")
    fig.suptitle(args.image, fontsize=10, y=0.99)
    fig.tight_layout()
    p3 = args.out / f"{stem}_side_by_side.png"
    fig.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p3}")


if __name__ == "__main__":
    main()
