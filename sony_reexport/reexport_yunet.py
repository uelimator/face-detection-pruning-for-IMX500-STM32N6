#!/usr/bin/env python3
"""
Re-export an existing YuNet ONNX through PyTorch >= 2.0 so it is accepted
by the Sony IMX500 converter (imxconv-pt).

Pipeline:
    OpenCV-Zoo YuNet ONNX  (old PyTorch 1.7 export)
        |  onnx2torch  -> nn.Module
    PyTorch nn.Module
        |  torch.onnx.export (>=2.0)
    Re-exported ONNX (producer = pytorch, version >= 2.0)
        |  imxconv-pt
    packerOut.zip

Requirements (in your existing venv):
    pip install 'torch>=2.0' onnx onnx2torch onnxsim

Usage:
    # 1) Download the float OpenCV Zoo model:
    #    https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    # 2) Run:
    python reexport_yunet.py \
        --input face_detection_yunet_2023mar.onnx \
        --output yunet_pt2.onnx \
        --input-size 320 320
"""

import argparse
import sys

import numpy as np
import onnx
import torch
from onnx2torch import convert
from onnxsim import simplify


def reexport(input_path: str, output_path: str, w: int, h: int, opset: int):
    # ---- 1. Load original ONNX and simplify it ---------------------------
    print(f"[1/4] Loading {input_path}")
    model_onnx = onnx.load(input_path)

    print("[2/4] Simplifying with onnx-simplifier (folds constants, "
          "removes dead branches)")
    """
    try:
        model_onnx, ok = simplify(model_onnx)
        if not ok:
            print("      onnxsim reported validation failed; "
                  "continuing with original graph", file=sys.stderr)
    except Exception as e:
        print(f"      onnxsim failed ({e}); continuing without simplification",
              file=sys.stderr)"""

    # ---- 2. Convert ONNX -> torch.nn.Module ------------------------------
    print("[3/4] Converting ONNX graph -> PyTorch nn.Module via onnx2torch")
    torch_model = convert(model_onnx)
    torch_model.eval()

    # Sanity check: forward pass on dummy input
    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)
    with torch.no_grad():
        out = torch_model(dummy)
    if isinstance(out, (list, tuple)):
        print(f"      Model produced {len(out)} output tensors:")
        for i, t in enumerate(out):
            print(f"        out[{i}].shape = {tuple(t.shape)}")
    else:
        print(f"      Single output, shape = {tuple(out.shape)}")

    # ---- 3. Re-export via torch.onnx.export ------------------------------
    print(f"[4/4] Exporting to {output_path} "
          f"(opset {opset}, fixed shape 1x3x{h}x{w}, no dynamic axes)")
    torch.onnx.export(
        torch_model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        # We don't name outputs because there can be several and onnx2torch
        # gives them generic names; the converter doesn't care about names.
        dynamic_axes=None,                # IMX500 requires fixed shapes
        do_constant_folding=True,
        export_params=True,
        dynamo = False,
    )

    # ---- 4. Verify the new file is what we want --------------------------
    new_model = onnx.load(output_path)
    print(f"\nDone. New model metadata:")
    print(f"  producer_name    = {new_model.producer_name}")
    print(f"  producer_version = {new_model.producer_version}")
    print(f"  ir_version       = {new_model.ir_version}")
    print(f"  opset            = "
          f"{[(o.domain or 'ai.onnx', o.version) for o in new_model.opset_import]}")
    print(f"\nNext step:")
    print(f"  imxconv-pt -i {output_path} -o ./converter_out "
          f"--no-input-persistency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to the existing YuNet .onnx file")
    ap.add_argument("--output", required=True,
                    help="Path to write the re-exported .onnx file")
    ap.add_argument("--input-size", nargs=2, type=int, default=[320, 320],
                    metavar=("W", "H"),
                    help="Network input size (default 320 320)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (default 17; 15-20 are safe)")
    args = ap.parse_args()
    reexport(args.input, args.output, args.input_size[0], args.input_size[1],
             args.opset)


if __name__ == "__main__":
    main()
