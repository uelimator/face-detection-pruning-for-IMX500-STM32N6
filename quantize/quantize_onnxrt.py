"""Static INT8 PTQ of an ONNX model via ONNX Runtime.

Reads pre-computed calibration tensors from a directory (built by
calibration/build_calib_set.py), runs them through the FP32 model to collect
activation ranges, and writes a quantized model with INT8 weights and
activations.

Three project-specific choices baked in (each one was learned by hitting the
opposite default and failing on a vendor compiler):

  - QDQ format (not QOperator). Required by STEdgeAI (STM32N6) and imxconv-pt
    (IMX500). They prefer QDQ because it keeps original op identities visible
    and gives the vendor compiler freedom in fusion strategy.

  - Signed INT8 activations (NOT UInt8). Both X-CUBE-AI and imxconv-pt reject
    UInt8 activations in QDQ with "quantized unsigned integer not supported."
    ORT defaults to UInt8 because that's faster on CPU/CUDA; signed INT8 is
    the portable choice for NPU deployment.

  - Auto-reshape input to match calibration data + auto-upgrade opset to 13.
    Per-channel QDQ emits DequantizeLinear with an `axis` attribute that only
    exists from opset 13 onward. The libfacedetection.train ONNX is opset 11;
    the pruned ONNX exports follow whatever opset was used. We upgrade if
    needed before quantization to avoid a later "Unrecognized attribute: axis"
    error from ONNX Runtime.

Example:
    python quantize_onnxrt.py \\
        --fp32 ../pruning/pruned_yunet_structured.onnx \\
        --calib-dir ../calibration/calib_images_320 \\
        --output ../pruning/pruned_yunet_structured_int8.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference, version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

MIN_OPSET_FOR_PER_CHANNEL_QDQ = 13


def reshape_input(model: onnx.ModelProto, h: int, w: int) -> onnx.ModelProto:
    """Force the model's input to (1, 3, h, w) and re-infer all downstream shapes.

    YuNet's ONNX exports usually have a fixed input shape baked in. If the
    calibration data was prepared at a different spatial size, the model
    needs its input dim rewritten before quantization can use that data.
    """
    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    for d, v in zip(dims, (1, 3, h, w)):
        d.Clear()
        d.dim_value = v
    # Wipe cached shape info on outputs / intermediate tensors so they get
    # recomputed from the new input shape.
    for out in model.graph.output:
        out.type.tensor_type.ClearField("shape")
    model.graph.ClearField("value_info")
    return shape_inference.infer_shapes(model)


class NpyCalibrationReader(CalibrationDataReader):
    """Yields {input_name: np.ndarray} dicts from .npy files in a directory."""

    def __init__(self, calib_dir: Path, input_name: str) -> None:
        self.input_name = input_name
        self.files = sorted(calib_dir.glob("*.npy"))
        if not self.files:
            raise SystemExit(f"No .npy files in {calib_dir} — run build_calib_set.py first")
        self._iter = iter(self.files)

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: np.load(path)}

    def rewind(self):
        self._iter = iter(self.files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp32", type=Path, required=True,
                        help="Path to the FP32 ONNX")
    parser.add_argument("--calib-dir", type=Path, required=True,
                        help="Directory of .npy calibration tensors")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the INT8 ONNX")
    parser.add_argument("--input-name", type=str, default="input",
                        help="Name of the model's input tensor (default: 'input')")
    parser.add_argument("--per-channel", action="store_true", default=True,
                        help="Per-channel weight quantization (default on; safer for vision)")
    parser.add_argument("--calib-method", choices=["minmax", "entropy", "percentile"],
                        default="minmax",
                        help="Activation range collection method")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(args.fp32))
    suffix_parts: list[str] = []

    # 1. Reshape input if calibration data uses a different spatial size than
    #    the model declares.
    sample_files = sorted(args.calib_dir.glob("*.npy"))
    if not sample_files:
        raise SystemExit(f"No .npy files in {args.calib_dir} — run build_calib_set.py first")
    sample_shape = np.load(sample_files[0]).shape  # (1, 3, H, W)
    calib_h, calib_w = sample_shape[2], sample_shape[3]
    model_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if model_dims[2] != calib_h or model_dims[3] != calib_w:
        print(f"Reshaping input {model_dims[2]}x{model_dims[3]} -> {calib_h}x{calib_w}")
        model = reshape_input(model, calib_h, calib_w)
        suffix_parts.append(f"{calib_h}x{calib_w}")

    # 2. Upgrade opset if needed (per-channel QDQ requires opset >= 13).
    current_opset = next((imp.version for imp in model.opset_import
                          if imp.domain in ("", "ai.onnx")), None)
    if current_opset is None or current_opset < MIN_OPSET_FOR_PER_CHANNEL_QDQ:
        print(f"Upgrading opset {current_opset} -> {MIN_OPSET_FOR_PER_CHANNEL_QDQ}")
        model = version_converter.convert_version(model, MIN_OPSET_FOR_PER_CHANNEL_QDQ)
        suffix_parts.append(f"op{MIN_OPSET_FOR_PER_CHANNEL_QDQ}")

    fp32_for_quant = args.fp32
    if suffix_parts:
        fp32_for_quant = args.fp32.with_name(f"{args.fp32.stem}_{'_'.join(suffix_parts)}.onnx")
        onnx.save(model, str(fp32_for_quant))
        print(f"Saved adapted FP32 model: {fp32_for_quant.name}")

    reader = NpyCalibrationReader(args.calib_dir, args.input_name)

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }

    print(f"\nQuantizing {fp32_for_quant.name}")
    print(f"  Calibration: {len(reader.files)} samples from {args.calib_dir}")
    print(f"  Method: {args.calib_method}, per_channel={args.per_channel}")
    print(f"  Weight type: INT8 (signed), Activation type: INT8 (signed)")

    quantize_static(
        model_input=str(fp32_for_quant),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,    # Signed — NOT UInt8. See module docstring.
        calibrate_method=method_map[args.calib_method],
    )

    src_mb = args.fp32.stat().st_size / 1e6
    dst_mb = args.output.stat().st_size / 1e6
    print(f"\nDone.")
    print(f"  FP32: {src_mb:.2f} MB")
    print(f"  INT8: {dst_mb:.2f} MB  ({dst_mb / src_mb:.0%} of FP32)")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
