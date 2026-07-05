"""Build a deterministic calibration set for INT8 PTQ from WIDER FACE train.

Samples N images, applies Path B preprocessing (center-crop to square + resize
to target_size), saves each as a float32 .npy of shape (1, 3, H, W) in BGR
0-255 range. That's what `cv2.dnn.blobFromImage` produces and what the YuNet
ONNX expects (no normalization — img_norm_cfg has mean=0, std=1).

Why WIDER train, not val? Calibration data must be disjoint from the eval set
— using val leaks the eval distribution into the quantization scales and
inflates AP numbers.

Why deterministic (seed + sorted iteration)? Quantization scales depend on the
calibration sample. Pinning the sample makes accuracy drift between PTQ runs
reflect real changes, not sample noise.

Why Path B preprocessing? Deployment will see Path B inputs (center-cropped
+ resized at the DCMIPP / camera stage). Calibrating on the same distribution
the deployed model will see is the right choice.

Example:
    python build_calib_set.py \\
        --wider-train $WIDER_ROOT/WIDER_train \\
        --output ./calib_images_320 \\
        --num-images 200 \\
        --input-size 320
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def collect_image_paths(wider_train: Path) -> list[Path]:
    images_root = wider_train / "images"
    if not images_root.is_dir():
        sys.exit(f"Expected {images_root} to exist. Pass --wider-train pointing at WIDER_train/")
    return sorted(images_root.glob("*/*.jpg"))


def center_crop_to_square_and_resize(image: np.ndarray, target: int) -> np.ndarray:
    """Path B preprocessing — matches what run_yunet_wider.py and the deployment do."""
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square, resize to `size`, then blobify to (1, 3, size, size)."""
    cropped = center_crop_to_square_and_resize(image, size)
    return cv2.dnn.blobFromImage(cropped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wider-train", type=Path, required=True,
                        help="Path to WIDER_train (folder containing images/)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory to write .npy calibration tensors into")
    parser.add_argument("--num-images", type=int, default=200,
                        help="How many images to sample (200 is a sweet spot for vision PTQ)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for deterministic sampling")
    parser.add_argument("--input-size", type=int, default=320,
                        help="Target square size (must match what the ONNX model expects)")
    args = parser.parse_args()

    all_paths = collect_image_paths(args.wider_train)
    if len(all_paths) < args.num_images:
        sys.exit(f"Only found {len(all_paths)} images, but --num-images={args.num_images}")

    rng = random.Random(args.seed)
    sampled = rng.sample(all_paths, args.num_images)

    args.output.mkdir(parents=True, exist_ok=True)
    # Clear any prior .npy files so the set is exactly what we just sampled
    for stale in args.output.glob("*.npy"):
        stale.unlink()

    written = 0
    for i, img_path in enumerate(sampled):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  skip (unreadable): {img_path}", file=sys.stderr)
            continue
        blob = preprocess(image, args.input_size)
        out_path = args.output / f"{i:04d}_{img_path.stem}.npy"
        np.save(out_path, blob)
        written += 1

    print(f"Wrote {written} calibration tensors to {args.output}")
    print(f"  Each is shape (1, 3, {args.input_size}, {args.input_size}), float32, BGR 0-255")


if __name__ == "__main__":
    main()
