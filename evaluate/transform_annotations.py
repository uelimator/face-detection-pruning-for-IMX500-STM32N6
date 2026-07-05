"""Transform WIDER FACE ground-truth annotations to match Path B preprocessing.

Mirrors the same geometric transform that run_yunet_wider.py and compare_outputs.py
apply to the images:
    1. Center-crop to the largest square (side = min(H, W))
    2. Resize the square to (target, target)

For each ground-truth bbox we:
    - Subtract the crop offset (left, top)
    - Drop boxes that fall entirely outside the crop region
    - Clip boxes that fall partially outside (to the visible portion)
    - Scale the result by target/side
    - Drop boxes that are too small after transform (--min-side filter)

The output keeps the official WIDER format so it stays compatible with the
standard eval toolkits.

Example:
    python transform_annotations.py \\
        --wider-root ~/Documents/Projects/Datasets/Wider_Faces/WIDER_val \\
        --annotations ~/Documents/Projects/Datasets/Wider_Faces/wider_face_split/wider_face_val_bbx_gt.txt \\
        --output ~/Documents/Projects/Datasets/Wider_Faces/wider_face_split/wider_face_val_bbx_gt_320x320.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_blocks(lines: list[str]):
    """Yield (image_rel_path, num_faces, face_lines) per image block.

    Handles the WIDER quirk where num_faces==0 still has one placeholder line
    of zeros.
    """
    i = 0
    while i < len(lines):
        path = lines[i].strip()
        if not path:
            i += 1
            continue
        i += 1
        n = int(lines[i].strip())
        i += 1
        if n == 0:
            # One placeholder line "0 0 0 0 0 0 0 0 0 0"
            i += 1
            yield path, 0, []
        else:
            face_lines = lines[i : i + n]
            i += n
            yield path, n, face_lines


def transform_box(x: float, y: float, w: float, h: float,
                  left: int, top: int, side: int, scale: float) -> tuple[float, float, float, float] | None:
    """Transform a single bbox; return None if it disappears under crop+filter."""
    # Crop-relative coords (still in original-pixel scale)
    x -= left
    y -= top
    x2, y2 = x + w, y + h

    # Clip to [0, side) square
    cx1 = max(0.0, x)
    cy1 = max(0.0, y)
    cx2 = min(float(side), x2)
    cy2 = min(float(side), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None  # entirely outside the crop

    # Scale to target pixel space
    return cx1 * scale, cy1 * scale, (cx2 - cx1) * scale, (cy2 - cy1) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wider-root", type=Path, required=True,
                        help="Path to WIDER_val or WIDER_train (containing images/)")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Original WIDER bbx_gt.txt file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the transformed annotations")
    parser.add_argument("--input-size", type=int, default=320,
                        help="Target square side (must match the preprocessing in run_yunet_wider.py)")
    parser.add_argument("--min-side", type=float, default=2.0,
                        help="Drop boxes whose width or height after transform is below this "
                             "(in target-pixel space). Boxes smaller than ~2 px are essentially "
                             "undetectable and would corrupt mAP curves.")
    args = parser.parse_args()

    lines = args.annotations.read_text().splitlines()
    out_lines: list[str] = []

    total_images = 0
    total_in = 0
    total_kept = 0
    total_dropped_outside = 0
    total_dropped_small = 0

    for rel_path, n, face_lines in parse_blocks(lines):
        img_path = args.wider_root / "images" / rel_path
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  skip (unreadable): {img_path}", file=sys.stderr)
            continue
        h, w = image.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        scale = args.input_size / side

        kept_faces: list[str] = []
        for line in face_lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, fw, fh = (float(v) for v in parts[:4])
            attrs = parts[4:]  # blur, expression, illumination, invalid, occlusion, pose
            total_in += 1

            tb = transform_box(x, y, fw, fh, left, top, side, scale)
            if tb is None:
                total_dropped_outside += 1
                continue
            nx, ny, nw, nh = tb
            if nw < args.min_side or nh < args.min_side:
                total_dropped_small += 1
                continue

            # Match WIDER format: integer pixel coords + original attribute flags
            kept_faces.append(
                f"{int(round(nx))} {int(round(ny))} {int(round(nw))} {int(round(nh))} "
                + " ".join(attrs)
            )
            total_kept += 1

        out_lines.append(rel_path)
        if not kept_faces:
            out_lines.append("0")
            out_lines.append("0 0 0 0 0 0 0 0 0 0")  # placeholder per WIDER convention
        else:
            out_lines.append(str(len(kept_faces)))
            out_lines.extend(kept_faces)

        total_images += 1
        if total_images % 500 == 0:
            print(f"  processed {total_images} images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")

    print(f"\nDone. {total_images} images written to {args.output}")
    print(f"  Faces in:        {total_in}")
    print(f"  Faces kept:      {total_kept}")
    print(f"  Dropped (outside crop): {total_dropped_outside}")
    print(f"  Dropped (<{args.min_side}px after resize): {total_dropped_small}")


if __name__ == "__main__":
    main()
