"""Export a pruned YuNet (.pt pickle) back to ONNX in the libfacedetection.train format.

The exported ONNX has:
  - input name: "input", shape (1, 3, 320, 320), float32
  - 12 named outputs (cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32)
  - sigmoid baked into cls and obj (matches the libfd.train export)
  - opset 11 (matches the libfd.train ONNX; opset 13+ would also work)

This produces a file that drops directly into:
    libfd_validation/run_libfd_inference.py
to generate predictions on WIDER val, which can then be scored by the
existing WIDER eval toolkit against the 320x320-space ground truth.

Example:
    python export_pruned_to_onnx.py \\
        --pruned pruned_yunet_30pct.pt \\
        --output pruned_yunet_30pct.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ONNX_OUTPUT_NAMES = [
    "cls_8", "cls_16", "cls_32",
    "obj_8", "obj_16", "obj_32",
    "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if np.array_equal(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pruned", type=Path, required=True,
                        help="Pruned model .pt produced by prune_yunet.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the exported .onnx")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset (default 11 matches the libfd.train export)")
    args = parser.parse_args()

    # Make yunet_standalone importable for the pickle to deserialize.
    sys.path.insert(0, str(Path(__file__).parent))

    print(f"=== Loading pruned model: {args.pruned} ===")
    model = torch.load(str(args.pruned), map_location="cpu", weights_only=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    example = torch.zeros(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        pt_outputs = [t.numpy() for t in model(example)]
    print(f"  forward OK, {len(pt_outputs)} output tensors")

    print(f"\n=== Exporting to {args.output} (opset {args.opset}) ===")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["input"],
        output_names=ONNX_OUTPUT_NAMES,
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
        # No dynamic axes — fixed 320x320 input matches the deployment target
        # (and the existing quantize/eval pipeline already assumes this shape).
    )

    # Strip the allowzero attribute that PyTorch 2.x emits on every Reshape — it
    # trips up CubeAI / STEdgeAI parsers even when set to its default value.
    import onnx
    exported = onnx.load(str(args.output))
    for node in exported.graph.node:
      if node.op_type == "Reshape":
          to_keep = [a for a in node.attribute if a.name != "allowzero"]
          del node.attribute[:] 
          node.attribute.extend(to_keep)
    onnx.save(exported, str(args.output))

    size_kb = args.output.stat().st_size / 1024.0
    print(f"  Wrote {size_kb:.1f} KB")

    print("\n=== Verifying ONNX matches PyTorch ===")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names_in_model = [o.name for o in sess.get_outputs()]
    if out_names_in_model != ONNX_OUTPUT_NAMES:
        print(f"  WARN: ONNX outputs in unexpected order: {out_names_in_model}")
    onnx_outputs = sess.run(out_names_in_model, {in_name: example.numpy()})

    worst_cos = 1.0
    worst_diff = 0.0
    for name, pt, ox in zip(out_names_in_model, pt_outputs, onnx_outputs):
        if pt.shape != ox.shape:
            print(f"  {name}: SHAPE MISMATCH pt={pt.shape} onnx={ox.shape}")
            continue
        cos = cosine(pt, ox)
        diff = float(np.max(np.abs(pt - ox)))
        worst_cos = min(worst_cos, cos)
        worst_diff = max(worst_diff, diff)
        print(f"  {name}: cos={cos:.6f}  max|Δ|={diff:.6f}")

    print(f"\n  worst cos: {worst_cos:.6f}, worst max|Δ|: {worst_diff:.6e}")
    if worst_cos >= 0.9999 and worst_diff < 1e-4:
        print("  ONNX export is faithful to the PyTorch model.")
    else:
        print("  WARN: ONNX outputs drift from PyTorch — check export options.")

    print("\n=== Done ===")
    print("To eval on WIDER val using the existing pipeline:")
    print(f"  python ../libfd_validation/run_libfd_inference.py \\")
    print(f"    --model {args.output.resolve()} \\")
    print(f"    --wider-root $WIDER_ROOT/WIDER_val \\")
    print(f"    --output-dir ./out_pruned_30pct")
    print("Then score the resulting predictions against the 320x320 GT.")


if __name__ == "__main__":
    main()
