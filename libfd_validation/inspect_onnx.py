"""Quick inspector for a YuNet-ish ONNX of unknown provenance.

Prints inputs and outputs (names, shapes, dtypes) plus a sample inference on a
zero tensor so you can see what the decode head emits. Use this before running
the inference script to confirm the output structure matches what the decode
code assumes (12 outputs: cls_*, obj_*, bbox_*, kps_* per stride 8/16/32).

If the names/shapes look different (fewer outputs, different naming, decode
already fused in, etc.), tell me what you see and we'll adapt the inference
script.

Example:
    python inspect_onnx.py --model /path/to/libfd_yunet.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    m = onnx.load(str(args.model))
    print(f"=== {args.model.name} ===\n")
    print(f"opset: {next(imp.version for imp in m.opset_import if imp.domain in ('', 'ai.onnx'))}")
    print(f"ir_version: {m.ir_version}\n")

    print("Inputs:")
    for i in m.graph.input:
        dims = [d.dim_value if d.dim_value else (d.dim_param or '?') for d in i.type.tensor_type.shape.dim]
        print(f"  {i.name}: {dims}")

    print("\nOutputs:")
    for o in m.graph.output:
        dims = [d.dim_value if d.dim_value else (d.dim_param or '?') for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {dims}")

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    # Substitute any non-int dim with 1 or 320 for the test
    concrete_shape = []
    for d in in_shape:
        if isinstance(d, int) and d > 0:
            concrete_shape.append(d)
        elif d in ("batch", "N", None):
            concrete_shape.append(1)
        else:
            concrete_shape.append(320)
    print(f"\nDummy inference at shape {concrete_shape}:")
    dummy = np.zeros(concrete_shape, dtype=np.float32)
    outs = sess.run(None, {in_name: dummy})
    for o, arr in zip(sess.get_outputs(), outs):
        print(f"  {o.name}: shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
