"""Run a libfacedetection.train YuNet ONNX over WIDER val and emit predictions.

Assumes the 12-output YuNet head (cls_8/16/32, obj_*, bbox_*, kps_*). If the
ONNX has a different output structure, run inspect_onnx.py first and adapt
the OUTPUT_NAMES constant / decode logic below.

Writes WIDER-format predictions to <output-dir>/predictions/<event>/<image>.txt
in the same layout as the project's main run_yunet_wider.py, so the existing
WIDER eval toolkit can score it directly against the transformed .mat GT.

Example:
    python run_libfd_inference.py \\
        --model /path/to/libfd_yunet.onnx \\
        --wider-root $WIDER_ROOT/WIDER_val \\
        --output-dir ./out_libfd_320
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


STRIDES = (8, 16, 32)


def output_names(strides=STRIDES) -> list[str]:
    """12-tensor YuNet output schema: cls_*, obj_*, bbox_*, kps_* per stride."""
    return [f"{kind}_{s}" for kind in ("cls", "obj", "bbox", "kps") for s in strides]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def center_crop_to_square_and_resize(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


def generate_priors(input_size: int) -> dict[int, np.ndarray]:
    """Per-stride grid-cell positions (cx_grid, cy_grid) in feature-map units."""
    priors = {}
    for stride in STRIDES:
        n = input_size // stride
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        priors[stride] = np.stack([xs.flatten(), ys.flatten()], axis=-1).astype(np.float32)
    return priors


def decode(outputs: dict[str, np.ndarray], priors: dict[int, np.ndarray],
           apply_sigmoid: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode all 12 outputs into (boxes_xywh, scores, kps_10).

    Standard YuNet anchor-free decode:
        cx = (grid_x + bbox[0]) * stride
        cy = (grid_y + bbox[1]) * stride
         w = exp(bbox[2]) * stride
         h = exp(bbox[3]) * stride
        kps_i = (grid + kps[i]) * stride
    Score = cls * obj (already-probabilities) or sigmoid(cls) * sigmoid(obj)
    (raw logits). Determined by --apply-sigmoid flag — libfacedetection.train
    bakes sigmoid into the ONNX, opencv_zoo's export does not.
    """
    boxes_all, scores_all, kps_all = [], [], []
    for stride in STRIDES:
        cls = outputs[f"cls_{stride}"][0]   # (H*W, 1)
        obj = outputs[f"obj_{stride}"][0]   # (H*W, 1)
        bbox = outputs[f"bbox_{stride}"][0] # (H*W, 4)
        kps = outputs[f"kps_{stride}"][0]   # (H*W, 10)

        prior = priors[stride]  # (H*W, 2) grid positions

        if apply_sigmoid:
            score = (sigmoid(cls) * sigmoid(obj)).flatten()
        else:
            score = (cls * obj).flatten()

        cx = (prior[:, 0] + bbox[:, 0]) * stride
        cy = (prior[:, 1] + bbox[:, 1]) * stride
        w = np.exp(bbox[:, 2]) * stride
        h = np.exp(bbox[:, 3]) * stride
        x = cx - w / 2
        y = cy - h / 2
        boxes = np.stack([x, y, w, h], axis=-1)  # (H*W, 4)

        kps_pts = kps.reshape(-1, 5, 2)
        kps_pts[..., 0] = (prior[:, 0:1] + kps_pts[..., 0]) * stride
        kps_pts[..., 1] = (prior[:, 1:2] + kps_pts[..., 1]) * stride
        kps_flat = kps_pts.reshape(-1, 10)

        boxes_all.append(boxes)
        scores_all.append(score)
        kps_all.append(kps_flat)

    return (np.concatenate(boxes_all, axis=0),
            np.concatenate(scores_all, axis=0),
            np.concatenate(kps_all, axis=0))


def nms(boxes: np.ndarray, scores: np.ndarray, conf_thr: float,
        iou_thr: float, top_k: int) -> np.ndarray:
    """Apply confidence filter + NMS. Returns indices into the input arrays."""
    keep = scores >= conf_thr
    if not keep.any():
        return np.array([], dtype=int)
    boxes_f = boxes[keep]
    scores_f = scores[keep]
    indices_f = np.where(keep)[0]
    nms_idx = cv2.dnn.NMSBoxes(boxes_f.tolist(), scores_f.tolist(), conf_thr, iou_thr, top_k=top_k)
    if len(nms_idx) == 0:
        return np.array([], dtype=int)
    nms_idx = np.array(nms_idx).flatten()
    return indices_f[nms_idx]


def write_wider_prediction(path: Path, stem: str, boxes: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [stem, str(len(boxes))]
    for (x, y, w, h), s in zip(boxes, scores):
        lines.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} {float(s):.4f}")
    path.write_text("\n".join(lines) + "\n")


def iter_wider_images(wider_root: Path):
    images_root = wider_root / "images"
    if not images_root.is_dir():
        sys.exit(f"Expected {images_root} to exist.")
    for event_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for img_path in sorted(event_dir.glob("*.jpg")):
            yield event_dir.name, img_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to the libfacedetection.train YuNet .onnx")
    parser.add_argument("--wider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=320,
                        help="Square size for center-crop+resize (Path B preprocessing)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip Path B; feed images at native resolution (model input "
                             "shape must be dynamic for this to work)")
    parser.add_argument("--conf-threshold", type=float, default=0.02)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply-sigmoid", action="store_true",
                        help="Apply sigmoid to cls/obj outputs. Use this if the ONNX emits "
                             "raw logits (opencv_zoo's YuNet). Leave OFF for "
                             "libfacedetection.train which bakes sigmoid into the graph.")
    args = parser.parse_args()

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names_in_model = {o.name for o in sess.get_outputs()}
    expected = set(output_names())
    missing = expected - out_names_in_model
    if missing:
        sys.exit(
            f"Model outputs don't match the 12-output YuNet schema. "
            f"Missing: {sorted(missing)}\nGot: {sorted(out_names_in_model)}\n"
            f"Run inspect_onnx.py and adapt the decode in this script."
        )

    priors = generate_priors(args.input_size)
    predictions_root = args.output_dir / "predictions"

    n_done = 0
    n_dets = 0
    for event, img_path in iter_wider_images(args.wider_root):
        if args.limit is not None and n_done >= args.limit:
            break
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  skip (unreadable): {img_path}", file=sys.stderr)
            continue
        if not args.no_preprocess:
            image = center_crop_to_square_and_resize(image, args.input_size)
        blob = cv2.dnn.blobFromImage(image)  # (1, 3, H, W) float32 BGR 0-255

        outs = sess.run(list(output_names()), {in_name: blob})
        outputs = dict(zip(output_names(), outs))

        boxes, scores, kps = decode(outputs, priors, args.apply_sigmoid)
        keep = nms(boxes, scores, args.conf_threshold, args.nms_threshold, args.top_k)

        write_wider_prediction(
            predictions_root / event / f"{img_path.stem}.txt",
            img_path.stem,
            boxes[keep] if len(keep) else np.empty((0, 4)),
            scores[keep] if len(keep) else np.empty((0,)),
        )
        n_done += 1
        n_dets += len(keep)
        if n_done % 100 == 0:
            print(f"  {n_done} images, {n_dets} detections")

    print(f"\nDone. {n_done} images, {n_dets} detections.")
    print(f"  Predictions: {predictions_root}")


if __name__ == "__main__":
    main()
