#!/usr/bin/env python3
"""
Quantize a float YuNet ONNX model for IMX500 using Sony MCT.

Usage:
    python quantize_yunet.py \
        --input yunet_float.onnx \
        --output yunet_mct_quant.onnx \
        --calib-dir ./calib_images \
        --input-size 320 320
"""

import argparse
import glob
import os

import numpy as np

import model_compression_toolkit as mct
from model_compression_toolkit.core import CoreConfig, QuantizationConfig
#from model_compression_toolkit.target_platform_capabilities import (
#    get_target_platform_capabilities,
#)

from edgemdt_tpc import get_target_platform_capabilities


def _patch_onnx2torch_resize_for_fx():
    """
    onnx2torch's OnnxResize picks the torch interpolate mode from
    input_tensor.dim() - 2. Under MCT's torch.fx symbolic trace that dim is a
    Proxy, not an int, so the (mode, dim) lookup misses and raises. YuNet's
    Resize ops are all on 4D NCHW feature maps, so force 2 spatial dims when the
    dim isn't a concrete int.
    """
    import onnx2torch.node_converters.resize as rz

    orig = rz._onnx_mode_to_torch_mode

    def patched(onnx_mode, dim_size):
        if not isinstance(dim_size, int):
            dim_size = 2
        return orig(onnx_mode, dim_size)

    rz._onnx_mode_to_torch_mode = patched


def _register_resize_v18():
    """
    onnx2torch's Resize converter is registered for versions 10/11/13.
    PyTorch 2.x exports default to opset 18, where Resize is at v18 — same
    semantics as v13 plus an `antialias` attribute (default 0, unused by YuNet).
    Alias v18 to the existing v13 converter so opset-18 ONNX files load.
    """
    from onnx2torch.node_converters import resize as _rz  # noqa: F401  (forces decorators to run)
    from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription

    v13 = OperationDescription(domain='', operation_type='Resize', version=13)
    v18 = OperationDescription(domain='', operation_type='Resize', version=18)
    if v13 in _CONVERTER_REGISTRY and v18 not in _CONVERTER_REGISTRY:
        _CONVERTER_REGISTRY[v18] = _CONVERTER_REGISTRY[v13]


def _force_legacy_onnx_export():
    """
    MCT's exporter calls torch.onnx.export without dynamo=, so newer torch
    routes through the dynamo exporter, which mistranslates MCT's dynamic_axes
    (tuple inputs vs dict dynamic_shapes) and fails. Force the legacy
    TorchScript exporter, same as reexport_yunet.py does.
    """
    import torch

    orig = torch.onnx.export

    def wrapped(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return orig(*args, **kwargs)

    torch.onnx.export = wrapped


def make_representative_dataset(calib_dir, input_w, input_h, n_iter=200):
    """
    Yields preprocessed face images one batch at a time.

    The calibration set is pre-baked .npy tensors, already in the exact form
    YuNet expects: NCHW (1, 3, H, W), float32, raw 0-255 BGR, no normalization.
    So we load and yield them as-is rather than re-doing imread/resize/transpose.
    """
    paths = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))
    if not paths:
        raise SystemExit(f"No .npy calibration tensors found in {calib_dir}")
    paths = paths[:n_iter]
    print(f"Using {len(paths)} calibration tensors")

    def gen():
        for p in paths:
            arr = np.load(p).astype(np.float32)
            if arr.ndim == 3:                 # CHW -> NCHW
                arr = arr[np.newaxis]
            if arr.shape != (1, 3, input_h, input_w):
                print(f"      skipping {os.path.basename(p)}: "
                      f"shape {arr.shape} != (1, 3, {input_h}, {input_w})")
                continue
            yield [arr]

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Float ONNX model")
    ap.add_argument("--output", required=True, help="Output quantized ONNX")
    ap.add_argument("--calib-dir", required=True, help="Folder of calibration images")
    ap.add_argument("--input-size", nargs=2, type=int, default=[320, 320],
                    metavar=("W", "H"))
    ap.add_argument("--n-iter", type=int, default=200)
    args = ap.parse_args()

    w, h = args.input_size

    # IMX500 target platform — this is what makes the output compatible
    tpc = get_target_platform_capabilities(
    tpc_version="1.0", device_type="imx500"
    )


    repr_gen = make_representative_dataset(args.calib_dir, w, h, args.n_iter)

    # MCT's pytorch PTQ needs an nn.Module, not an ONNX proto. Convert the
    # ONNX graph to torch via onnx2torch (same path reexport_yunet.py uses).
    import onnx
    from onnx2torch import convert
    _register_resize_v18()
    _patch_onnx2torch_resize_for_fx()
    float_model = convert(onnx.load(args.input))
    float_model.eval()

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=repr_gen,
        core_config=CoreConfig(quantization_config=QuantizationConfig()),
        target_platform_capabilities=tpc,
    )

    # Export to ONNX with producer_name=pytorch so imxconv-pt accepts it
    _force_legacy_onnx_export()
    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=args.output,
        repr_dataset=repr_gen,
    )

    print(f"\nWrote quantized model to {args.output}")
    print("Now run:  imxconv-pt -i {} -o ./converter_out --no-input-persistency"
          .format(args.output))


if __name__ == "__main__":
    main()
