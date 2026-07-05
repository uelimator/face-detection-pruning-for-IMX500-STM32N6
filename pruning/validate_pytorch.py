"""Validate a standalone PyTorch YuNet against WIDER val.

Generates WIDER-format predictions by running the model on every WIDER val
image, then optionally invokes the official WIDER eval toolkit and parses
its Easy / Medium / Hard AP from stdout.

Exposes three top-level functions so other scripts (e.g. prune_yunet.py)
can call validation programmatically:

    generate_predictions(model, wider_root, output_dir, ...)
        Run inference on WIDER val; write predictions/<event>/<image>.txt.

    run_wider_eval(predictions_dir, gt_dir)
        Invoke the WIDER eval toolkit; return parsed {easy, medium, hard} AP.

    validate_model(model, wider_root, output_dir, gt_dir)
        Both of the above in sequence; returns the AP dict.

Mirrors the decode/NMS/prediction-writing logic from
libfd_validation/run_libfd_inference.py. Differences:
  - Loads via PyTorch instead of ONNX Runtime
  - Skips the --apply-sigmoid flag (standalone head always sigmoids cls/obj)
  - Functions importable from other scripts, not just CLI

Example (CLI):
    python validate_pytorch.py \\
        --weights ../training/libfacedetection.train/weights/yunet_n.pth \\
        --wider-root $WIDER_ROOT/WIDER_val \\
        --output-dir ./out_pytorch_baseline \\
        --gt-dir $WIDER_ROOT/wider_face_split_320x320
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from yunet_standalone import YuNet


STRIDES = (8, 16, 32)
WIDER_EVAL_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "evaluate" / "WiderFace-Evaluation-master" / "evaluation.py"
)


# ---------------------------------------------------------------------------
# Decode + NMS — mirrors libfd_validation/run_libfd_inference.py, kept
# numpy-based so we don't need PyTorch ops here. The model emits sigmoid'd
# cls/obj, so the decode does NOT re-apply sigmoid.
# ---------------------------------------------------------------------------

def center_crop_to_square_and_resize(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top:top + side, left:left + side]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)


def generate_priors(input_size: int) -> dict[int, np.ndarray]:
    priors = {}
    for stride in STRIDES:
        n = input_size // stride
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        priors[stride] = np.stack([xs.flatten(), ys.flatten()], axis=-1).astype(np.float32)
    return priors


def decode(outputs: dict[str, np.ndarray], priors: dict[int, np.ndarray]
           ) -> tuple[np.ndarray, np.ndarray]:
    """12 raw output tensors -> (boxes_xywh, scores). cls/obj are already sigmoid'd."""
    boxes_all, scores_all = [], []
    for stride in STRIDES:
        cls = outputs[f"cls_{stride}"][0]
        obj = outputs[f"obj_{stride}"][0]
        bbox = outputs[f"bbox_{stride}"][0]
        prior = priors[stride]

        score = (cls * obj).flatten()
        cx = (prior[:, 0] + bbox[:, 0]) * stride
        cy = (prior[:, 1] + bbox[:, 1]) * stride
        w = np.exp(bbox[:, 2]) * stride
        h = np.exp(bbox[:, 3]) * stride
        boxes_all.append(np.stack([cx - w / 2, cy - h / 2, w, h], axis=-1))
        scores_all.append(score)
    return np.concatenate(boxes_all), np.concatenate(scores_all)


def nms(boxes: np.ndarray, scores: np.ndarray, conf_thr: float,
        iou_thr: float, top_k: int) -> np.ndarray:
    keep = scores >= conf_thr
    if not keep.any():
        return np.array([], dtype=int)
    boxes_f = boxes[keep]
    scores_f = scores[keep]
    indices_f = np.where(keep)[0]
    nms_idx = cv2.dnn.NMSBoxes(boxes_f.tolist(), scores_f.tolist(), conf_thr, iou_thr, top_k=top_k)
    if len(nms_idx) == 0:
        return np.array([], dtype=int)
    return indices_f[np.array(nms_idx).flatten()]


def write_wider_prediction(path: Path, stem: str, boxes: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [stem, str(len(boxes))]
    for (x, y, w, h), s in zip(boxes, scores):
        lines.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} {float(s):.4f}")
    path.write_text("\n".join(lines) + "\n")


def iter_wider_images(wider_root: Path):
    images_root = wider_root / "images"
    if not images_root.is_dir():
        raise SystemExit(f"Expected {images_root} to exist.")
    for event_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for img_path in sorted(event_dir.glob("*.jpg")):
            yield event_dir.name, img_path


# ---------------------------------------------------------------------------
# Public API — importable from other scripts (e.g. prune_yunet.py)
# ---------------------------------------------------------------------------

def generate_predictions(model: torch.nn.Module, wider_root: Path, output_dir: Path,
                         input_size: int = 320, conf_threshold: float = 0.02,
                         nms_threshold: float = 0.45, top_k: int = 5000,
                         limit: int | None = None, device: str | torch.device = "cpu",
                         verbose: bool = True) -> Path:
    """Run inference on WIDER val and write WIDER-format predictions.

    Returns the path of the predictions root directory
    (output_dir/predictions/<event>/<image>.txt).
    """
    model = model.to(device)
    model.eval()
    priors = generate_priors(input_size)
    output_names = [f"{kind}_{s}" for kind in ("cls", "obj", "bbox", "kps") for s in STRIDES]
    predictions_root = output_dir / "predictions"

    n_done = 0
    n_dets = 0
    for event, img_path in iter_wider_images(wider_root):
        if limit is not None and n_done >= limit:
            break
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  skip (unreadable): {img_path}", file=sys.stderr)
            continue
        image = center_crop_to_square_and_resize(image, input_size)
        blob = cv2.dnn.blobFromImage(image)  # (1, 3, H, W) float32 BGR 0-255

        with torch.no_grad():
            out_tensors = model(torch.from_numpy(blob).to(device))
        outputs = dict(zip(output_names, [t.detach().cpu().numpy() for t in out_tensors]))

        boxes, scores = decode(outputs, priors)
        keep = nms(boxes, scores, conf_threshold, nms_threshold, top_k)

        write_wider_prediction(
            predictions_root / event / f"{img_path.stem}.txt",
            img_path.stem,
            boxes[keep] if len(keep) else np.empty((0, 4)),
            scores[keep] if len(keep) else np.empty((0,)),
        )
        n_done += 1
        n_dets += len(keep)
        if verbose and n_done % 100 == 0:
            print(f"  {n_done} images, {n_dets} detections")

    if verbose:
        print(f"\nDone. {n_done} images, {n_dets} detections.")
        print(f"  Predictions: {predictions_root}")
    return predictions_root


def run_wider_eval(predictions_dir: Path, gt_dir: Path,
                   eval_script: Path = WIDER_EVAL_SCRIPT,
                   python_exe: str | None = None) -> dict[str, float]:
    """Invoke the WIDER eval toolkit and parse its stdout.

    The toolkit prints lines like:
        Easy   Val AP: 0.8826
        Medium Val AP: 0.8704
        Hard   Val AP: 0.6921
    Regex these out and return as {'easy': ..., 'medium': ..., 'hard': ...}.

    If the toolkit fails or output doesn't match, returns NaNs and prints
    the full stderr/stdout for debugging.
    """
    if python_exe is None:
        python_exe = sys.executable
    cmd = [
        str(python_exe), str(eval_script),
        "--pred", str(predictions_dir),
        "--gt", str(gt_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr

    aps = {"easy": float("nan"), "medium": float("nan"), "hard": float("nan")}
    for line in output.splitlines():
        m = re.search(r"(Easy|Medium|Hard)\s+Val AP:\s+(\d+\.\d+)", line)

        if m:
            aps[m.group(1).lower()] = float(m.group(2))

    if proc.returncode != 0 or any(v != v for v in aps.values()):  # NaN check
        print("WIDER eval output (for debugging):", file=sys.stderr)
        print(output, file=sys.stderr)

    return aps


def validate_model(model: torch.nn.Module, wider_root: Path, output_dir: Path,
                   gt_dir: Path, **kwargs) -> dict[str, float]:
    """End-to-end: generate predictions + run eval. Returns AP dict.

    All kwargs are forwarded to generate_predictions (input_size,
    conf_threshold, nms_threshold, top_k, limit, device, verbose).
    """
    predictions_dir = generate_predictions(model, wider_root, output_dir, **kwargs)
    print("Running WIDER eval ...")
    return run_wider_eval(predictions_dir, gt_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", type=Path, default=None,
                     help="libfacedetection.train .pth - instantiates fresh YuNet() and loads weights")
    src.add_argument("--checkpoint", type=Path, default=None,
                     help="PyTorch checkpoint dict with 'model_state_dict' key (weights_only=True)")
    src.add_argument("--pruned", type=Path, default=None,
                     help="Pruned model full pickle (torch.load with weights_only=False)")

    parser.add_argument("--wider-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="If supplied, run WIDER eval and print Easy/Medium/Hard AP")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--conf-threshold", type=float, default=0.02)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.weights is not None:
        model = YuNet()
        model.load_pretrained(str(args.weights))
    elif args.checkpoint is not None:
        model = YuNet()
        ckpt = torch.load(str(args.checkpoint), weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = torch.load(str(args.pruned), map_location="cpu", weights_only=False)

    if args.gt_dir is not None:
        aps = validate_model(
            model, args.wider_root, args.output_dir, args.gt_dir,
            input_size=args.input_size, conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold, top_k=args.top_k,
            limit=args.limit, device=args.device,
        )
        print(f"\nEasy:   {aps['easy']:.4f}")
        print(f"Medium: {aps['medium']:.4f}")
        print(f"Hard:   {aps['hard']:.4f}")
    else:
        generate_predictions(
            model, args.wider_root, args.output_dir,
            input_size=args.input_size, conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold, top_k=args.top_k,
            limit=args.limit, device=args.device,
        )
        print("(Skipped WIDER eval - pass --gt-dir to also score the predictions.)")


if __name__ == "__main__":
    main()
