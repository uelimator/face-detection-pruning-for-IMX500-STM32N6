"""Load the standalone YuNet from a libfacedetection.train .pth and verify vs ONNX.

This is the V2 of load_and_verify — uses the extracted pure-PyTorch model in
yunet_standalone/ instead of mmdet. No mmcv / mmdet install required.

Run from the modern Python 3.11 venv (FD/fdvenv) since the only dependencies
now are torch, numpy, onnx, onnxruntime.

Example:
    python load_and_verify.py \\
        --weights ../training/libfacedetection.train/weights/yunet_n.pth \\
        --onnx ../training/libfacedetection.train/onnx/yunet_n_320_320.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from yunet_standalone import YuNet


# Output names as declared in the libfacedetection.train ONNX export, in the
# same order our standalone head emits them.
ONNX_OUTPUT_NAMES = [
    "cls_8", "cls_16", "cls_32",
    "obj_8", "obj_16", "obj_32",
    "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
]


def run_onnx(onnx_path: Path, x: np.ndarray) -> dict[str, np.ndarray]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {in_name: x.astype(np.float32)})
    return dict(zip(out_names, outs))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if np.array_equal(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Pretrained .pth (e.g. weights/yunet_n.pth)")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="(Optional) ONNX file for numerical comparison")
    parser.add_argument("--input-size", type=int, default=320)
    args = parser.parse_args()

    print("=== Building standalone YuNet (yunet_n config) ===")
    model = YuNet()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_params:,}")
    print(f"Trainable params: {n_trainable:,}")

    print(f"\n=== Loading weights from {args.weights.name} ===")
    missing, unexpected = model.load_pretrained(str(args.weights), strict=False)
    print(f"  missing keys:    {len(missing)} (first 3: {missing[:3]})")
    print(f"  unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    if len(missing) > 5:
        print(f"  WARN: many missing keys — architecture may not match the checkpoint.")
    model.eval()

    print(f"\n=== Forward pass (zero input, {args.input_size}x{args.input_size}) ===")
    dummy = torch.zeros(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        outputs = model(dummy)
    if len(outputs) != len(ONNX_OUTPUT_NAMES):
        print(f"  WARN: expected {len(ONNX_OUTPUT_NAMES)} outputs, got {len(outputs)}")
    for name, t in zip(ONNX_OUTPUT_NAMES, outputs):
        arr = t.detach().numpy()
        print(f"  {name}: shape={tuple(arr.shape)} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")

    if args.onnx is not None:
        print(f"\n=== Comparing against {args.onnx.name} ===")
        onnx_outs = run_onnx(args.onnx, dummy.numpy())

        worst_cos = 1.0
        for name, t in zip(ONNX_OUTPUT_NAMES, outputs):
            pt = t.detach().numpy()
            ox = onnx_outs.get(name)
            if ox is None:
                print(f"  {name}: NOT IN ONNX")
                continue
            if pt.shape != ox.shape:
                print(f"  {name}: SHAPE MISMATCH pt={pt.shape} onnx={ox.shape}")
                continue
            cos = cosine_similarity(pt, ox)
            diff = float(np.max(np.abs(pt - ox)))
            worst_cos = min(worst_cos, cos)
            print(f"  {name}: cos={cos:.6f}  max|Δ|={diff:.6f}")

        if worst_cos >= 0.999:
            print(f"\n  Standalone model matches the ONNX (worst cos = {worst_cos:.6f}).")
            print("  Safe to use as the pruning starting point.")
        elif worst_cos >= 0.95:
            print(f"\n  Close but not identical (worst cos = {worst_cos:.4f}).")
            print("  Investigate before pruning — likely a BN-eval state or activation flag.")
        else:
            print(f"\n  Outputs DIFFER significantly (worst cos = {worst_cos:.4f}).")
            print("  Don't proceed to pruning. Most likely: architecture parameter mismatch.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
