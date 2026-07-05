"""Smoke test: numerical comparison of FP32 vs INT8 YuNet outputs.

For each test image, runs both models via ONNX Runtime and reports, per output
tensor, cosine similarity and max absolute difference. This is the fastest way
to answer "did quantization break the model" before bothering with mAP.

Interpretation rules of thumb for INT8 PTQ of a well-behaved vision model:
  cosine >= 0.99 on classification/objectness heads        — healthy
  cosine 0.95-0.99                                          — usable, watch mAP
  cosine < 0.95                                             — quantization
                                                              breaking something;
                                                              check per-channel,
                                                              calib set size,
                                                              or try entropy
                                                              calibration

Regression-head tensors (bbox, kps) tend to have lower cosine because their
values span a wider numeric range — judge them on max-abs-diff instead, scaled
against the typical anchor cell size (8/16/32 pixels at stride 8/16/32).

Example:
  python compare_outputs.py \\
      --fp32 ../models/face_detection_yunet_2023mar.onnx \\
      --int8 ../models/int8/yunet_int8_onnxrt.onnx \\
      --test-images ~/Documents/Projects/Datasets/Wider_Faces/WIDER_val/images \\
      --num-samples 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def center_crop_to_square_and_resize(image: np.ndarray, target_size: int) -> np.ndarray:
    """Match the on-device DCMIPP pipeline: center-crop to square, then resize."""
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    cropped = center_crop_to_square_and_resize(image, size)
    return cv2.dnn.blobFromImage(cropped)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if np.array_equal(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def sample_test_images(images_root: Path, n: int, seed: int) -> list[Path]:
    all_paths = sorted(images_root.glob("*/*.jpg"))
    if not all_paths:
        raise SystemExit(f"No .jpg images under {images_root}")
    rng = random.Random(seed)
    return rng.sample(all_paths, min(n, len(all_paths)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True,
                        help="Directory containing event subfolders of .jpg images "
                             "(e.g. WIDER_val/images)")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Different from the calib seed so we don't test on calibration data")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Spatial size to feed both models. If omitted, inferred from "
                             "the INT8 model's declared input shape.")
    args = parser.parse_args()

    if args.input_size is None:
        # Auto-detect from the INT8 model so we always match its expected shape.
        probe = ort.InferenceSession(str(args.int8), providers=["CPUExecutionProvider"])
        probe_shape = probe.get_inputs()[0].shape
        args.input_size = int(probe_shape[2])
        print(f"Inferred --input-size {args.input_size} from {args.int8.name}")

    sess_fp = ort.InferenceSession(str(args.fp32), providers=["CPUExecutionProvider"])
    sess_q8 = ort.InferenceSession(str(args.int8), providers=["CPUExecutionProvider"])
    input_name = sess_fp.get_inputs()[0].name
    output_names = [o.name for o in sess_fp.get_outputs()]

    samples = sample_test_images(args.test_images, args.num_samples, args.seed)
    print(f"Comparing {len(samples)} images, {len(output_names)} output tensors each.\n")

    # Accumulators across all samples, per output tensor
    cos_acc: dict[str, list[float]] = {n: [] for n in output_names}
    diff_acc: dict[str, list[float]] = {n: [] for n in output_names}

    for path in samples:
        image = cv2.imread(str(path))
        blob = preprocess(image, args.input_size)
        out_fp = sess_fp.run(output_names, {input_name: blob})
        out_q8 = sess_q8.run(output_names, {input_name: blob})
        for name, a, b in zip(output_names, out_fp, out_q8):
            cos_acc[name].append(cosine_similarity(a, b))
            diff_acc[name].append(float(np.max(np.abs(a - b))))

    name_w = max(len(n) for n in output_names)
    print(f"{'output':<{name_w}}  {'mean cos':>10}  {'min cos':>10}  {'mean |Δ|':>10}  {'max |Δ|':>10}")
    print("-" * (name_w + 50))
    for n in output_names:
        cos = np.array(cos_acc[n])
        dif = np.array(diff_acc[n])
        print(f"{n:<{name_w}}  {cos.mean():>10.4f}  {cos.min():>10.4f}  {dif.mean():>10.4f}  {dif.max():>10.4f}")

    overall_min_cos = min(min(v) for v in cos_acc.values())
    verdict = ("LIKELY HEALTHY" if overall_min_cos >= 0.99
               else "USABLE, CHECK MAP" if overall_min_cos >= 0.95
               else "SUSPICIOUS — investigate")
    print(f"\nOverall worst cosine: {overall_min_cos:.4f}  →  {verdict}")


if __name__ == "__main__":
    main()
