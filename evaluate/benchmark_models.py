"""Benchmark ONNX model variants for size / params / MACs / inference latency.

Produces a comparison table across however many models you point at. Useful
for comparing FP32-vs-INT8, unpruned-vs-pruned, or any other variant set.

Metrics:
  - File size      (bytes on disk)
  - Params         (sum of initializer element counts — the "weight count")
  - MACs           (multiply-accumulates for Conv / Gemm / MatMul ops, estimated
                    from declared input/output/weight shapes)
  - CPU latency    (ONNX Runtime on CPU; mean / p50 / p95 / min over N runs,
                    after a warmup phase)

The MAC estimate is approximate — it counts the dominant conv ops correctly
but ignores small contributions from element-wise ops, NMS, etc. Good for
comparing variants of the same architecture; not a substitute for a profiler
on the actual deployment target.

Example:
    python benchmark_models.py \\
        --model "FP32 baseline:../training/libfacedetection.train/onnx/yunet_n_320_320.onnx" \\
        --model "INT8 baseline:../pruning/yunet_int8_320.onnx" \\
        --model "FP32 pruned:../pruning/pruned_yunet_structured_20_06.onnx" \\
        --model "INT8 pruned:../pruning/pruned_yunet_structured_int8.onnx"
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


def file_size_bytes(path: Path) -> int:
    """Total bytes — for ONNX files where weights got externalized into a
    .onnx.data sidecar, sum both."""
    total = path.stat().st_size
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total


def count_params(model: onnx.ModelProto) -> int:
    """Total element count across ALL initializers in the ONNX graph.

    Walks every entry in `graph.initializer`, which includes:
      - Conv2d / Linear / MatMul weight tensors
      - Conv / Linear biases
      - BatchNorm gamma, beta, running_mean, running_var (folded if Conv+BN was fused)
      - For QDQ-quantised graphs: per-channel scales (FP32) + zero-points (INT8/INT32)
      - Any constant tensors emitted by ONNX export
    This is the *whole-model* parameter count — not just convs.
    """
    n = 0
    for init in model.graph.initializer:
        prod = 1
        for d in init.dims:
            prod *= d
        n += prod
    return n


# ONNX TensorProto.data_type -> bytes per element. From onnx.TensorProto.DataType.
_DTYPE_BYTES = {
    1: 4,    # FLOAT
    2: 1,    # UINT8
    3: 1,    # INT8
    4: 2,    # UINT16
    5: 2,    # INT16
    6: 4,    # INT32
    7: 8,    # INT64
    9: 1,    # BOOL
    10: 2,   # FLOAT16
    11: 8,   # DOUBLE
    12: 4,   # UINT32
    13: 8,   # UINT64
    16: 2,   # BFLOAT16
}


def count_init_bytes(model: onnx.ModelProto) -> int:
    """Actual storage bytes across all initializers (params × dtype_size).

    This is the real weight footprint — independent of file format overhead,
    external-data sidecars, and graph metadata, all of which inflate the on-disk
    .onnx file size in ways that don't reflect the model's true cost.
    """
    total = 0
    for init in model.graph.initializer:
        prod = 1
        for d in init.dims:
            prod *= d
        total += prod * _DTYPE_BYTES.get(init.data_type, 4)
    return total


def build_shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    """Map tensor_name -> [dim0, dim1, ...] using both value_info and initializers."""
    shapes: dict[str, list[int]] = {}
    for source in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in source:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value else 0)
            shapes[vi.name] = dims
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def count_macs(model: onnx.ModelProto) -> int:
    """Estimate MACs from Conv / Gemm / MatMul nodes.

    Conv:  MACs = Hout * Wout * Cout * (Cin/G) * Kh * Kw
    Gemm:  MACs = M * N * K  (for output MxN built from MxK . KxN)
    MatMul: same as Gemm, derived from last two dims of inputs.
    """
    model = shape_inference.infer_shapes(model)
    shapes = build_shape_map(model)
    macs = 0

    for node in model.graph.node:
        if node.op_type == "Conv":
            if len(node.input) < 2:
                continue
            in_shape = shapes.get(node.input[0], [])
            w_shape = shapes.get(node.input[1], [])
            out_shape = shapes.get(node.output[0], [])
            if len(in_shape) != 4 or len(out_shape) != 4 or len(w_shape) != 4:
                continue
            _, _, h_out, w_out = out_shape
            c_out, cin_per_g, kh, kw = w_shape
            macs += h_out * w_out * c_out * cin_per_g * kh * kw

        elif node.op_type == "Gemm":
            # A: (M, K) or (K, M) depending on transA; B: (K, N) or (N, K)
            if len(node.input) < 2:
                continue
            a_shape = shapes.get(node.input[0], [])
            b_shape = shapes.get(node.input[1], [])
            if len(a_shape) != 2 or len(b_shape) != 2:
                continue
            trans_a = trans_b = 0
            for attr in node.attribute:
                if attr.name == "transA":
                    trans_a = attr.i
                elif attr.name == "transB":
                    trans_b = attr.i
            m, k = (a_shape[1], a_shape[0]) if trans_a else (a_shape[0], a_shape[1])
            _, n = (b_shape[1], b_shape[0]) if trans_b else (b_shape[0], b_shape[1])
            macs += m * k * n

        elif node.op_type == "MatMul":
            a_shape = shapes.get(node.input[0], [])
            b_shape = shapes.get(node.input[1], [])
            if len(a_shape) < 2 or len(b_shape) < 2:
                continue
            # Last two dims drive the matmul; leading dims are broadcast
            m = a_shape[-2]
            k = a_shape[-1]
            n = b_shape[-1]
            # Multiply by leading-batch product for the larger of the two
            lead_a = 1
            for d in a_shape[:-2]:
                lead_a *= max(d, 1)
            lead_b = 1
            for d in b_shape[:-2]:
                lead_b *= max(d, 1)
            macs += m * k * n * max(lead_a, lead_b)

    return macs


def _make_session(onnx_path: Path) -> ort.InferenceSession:
    """Create an ORT session, falling back to mct_quantizers' custom-op
    registration if the plain CPU EP doesn't know an op (MCT-quantized models
    use custom WeightsQuantizer / ActivationQuantizer ops)."""
    try:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as first_err:
        try:
            from mct_quantizers import get_ort_session_options
        except ImportError:
            raise first_err
        opts = get_ort_session_options()
        return ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )


def measure_latency(
    onnx_path: Path, input_size: int, n_runs: int, n_warmup: int
) -> dict[str, float]:
    """Run N forward passes through ONNX Runtime on CPU, return latency stats (ms)."""
    sess = _make_session(onnx_path)
    in_meta = sess.get_inputs()[0]
    # Use the model's declared input shape if known; otherwise fall back to (1, 3, input_size, input_size).
    declared = []
    for d in in_meta.shape:
        if isinstance(d, int) and d > 0:
            declared.append(d)
        elif d in (None, "batch", "N"):
            declared.append(1)
        else:
            declared.append(input_size)
    if len(declared) != 4:
        declared = [1, 3, input_size, input_size]
    dummy = np.random.rand(*declared).astype(np.float32) * 255.0  # BGR-like range

    in_name = in_meta.name
    out_names = [o.name for o in sess.get_outputs()]

    # Warmup
    for _ in range(n_warmup):
        sess.run(out_names, {in_name: dummy})

    # Measure
    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(out_names, {in_name: dummy})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    return {
        "min": times_ms[0],
        "mean": statistics.mean(times_ms),
        "p50": times_ms[n // 2],
        "p95": times_ms[min(int(n * 0.95), n - 1)],
        "p99": times_ms[min(int(n * 0.99), n - 1)],
        "max": times_ms[-1],
    }


def _is_mct_model(model: onnx.ModelProto) -> bool:
    """True if the graph uses mct_quantizers custom ops. ORT can run these, but
    the reference kernels are unoptimised and dominate wall-clock — the latency
    has no relation to how the model executes on the IMX500 NPU."""
    return any(n.domain == "mct_quantizers" for n in model.graph.node)


def benchmark_one(
    name: str, path: Path, input_size: int, n_runs: int, n_warmup: int
) -> dict:
    if not path.exists():
        return {"name": name, "error": f"file not found: {path}"}

    model = onnx.load(str(path))
    size = file_size_bytes(path)
    init_bytes = count_init_bytes(model)
    params = count_params(model)
    macs = count_macs(model)

    base = {
        "name": name,
        "path": str(path),
        "size_bytes": size,
        "init_bytes": init_bytes,
        "params": params,
        "macs": macs,
    }

    # MCT models: skip CPU latency — it's not deployment-representative and the
    # reference quantizer kernels take orders of magnitude longer than QDQ ops.
    if _is_mct_model(model):
        base["latency_skipped"] = "MCT custom ops (CPU latency not representative)"
        return base

    base["latency_ms"] = measure_latency(path, input_size, n_runs, n_warmup)
    return base


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.2f} MB"


def format_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}G"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def print_table(results: list[dict]) -> None:
    headers = [
        "Variant",
        "File",
        "Weights",
        "All Params",
        "MACs",
        "Mean ms",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "min ms",
        "max ms",
    ]
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["name"], r["error"], "", "", "", "", "", "", "", "", ""])
            continue
        if "latency_skipped" in r:
            rows.append(
                [
                    r["name"],
                    format_size(r["size_bytes"]),
                    format_size(r["init_bytes"]),
                    format_count(r["params"]),
                    format_count(r["macs"]),
                    "—", "—", "—", "—", "—", "—",
                ]
            )
            continue
        rows.append(
            [
                r["name"],
                format_size(r["size_bytes"]),
                format_size(r["init_bytes"]),
                format_count(r["params"]),
                format_count(r["macs"]),
                f"{r['latency_ms']['mean']:.2f}",
                f"{r['latency_ms']['p50']:.2f}",
                f"{r['latency_ms']['p95']:.2f}",
                f"{r['latency_ms']['p99']:.2f}",
                f"{r['latency_ms']['min']:.2f}",
                f"{r['latency_ms']['max']:.2f}",
            ]
        )

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    sep = "  "

    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for row in rows:
        print(sep.join(v.ljust(w) for v, w in zip(row, widths)))


def write_csv(results: list[dict], path: Path) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "path",
                "size_bytes",
                "init_bytes",
                "all_params",
                "macs",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "latency_min_ms",
                "latency_max_ms",
            ]
        )
        for r in results:
            if "error" in r:
                continue
            if "latency_skipped" in r:
                w.writerow(
                    [r["name"], r["path"], r["size_bytes"], r["init_bytes"],
                     r["params"], r["macs"], "", "", "", "", "", ""]
                )
                continue
            w.writerow(
                [
                    r["name"],
                    r["path"],
                    r["size_bytes"],
                    r["init_bytes"],
                    r["params"],
                    r["macs"],
                    r["latency_ms"]["mean"],
                    r["latency_ms"]["p50"],
                    r["latency_ms"]["p95"],
                    r["latency_ms"]["p99"],
                    r["latency_ms"]["min"],
                    r["latency_ms"]["max"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="`Name:path/to/model.onnx`. Can be repeated for multiple models.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=320,
        help="Square spatial size to feed if the model has dynamic dims (default 320).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timed inference runs per model (default 200).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed warmup runs before measuring (default 20).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional: write results to this CSV file.",
    )
    args = parser.parse_args()

    if not args.model:
        sys.exit("Pass at least one --model 'Name:path' pair.")

    specs = []
    for s in args.model:
        if ":" not in s:
            sys.exit(f"--model arg must be 'Name:path', got {s!r}")
        name, p = s.split(":", 1)
        specs.append((name.strip(), Path(p.strip())))

    print(
        f"Benchmarking {len(specs)} model(s), {args.runs} runs each (after {args.warmup} warmup)...\n"
    )
    results = []
    for name, path in specs:
        print(f"  [{name}] {path.name} ...", end=" ", flush=True)
        r = benchmark_one(name, path, args.input_size, args.runs, args.warmup)
        if "error" in r:
            print(f"SKIP: {r['error']}")
        elif "latency_skipped" in r:
            print(f"latency skipped ({r['latency_skipped']})")
        else:
            print(f"{r['latency_ms']['mean']:.2f} ms (mean)")
        results.append(r)

    print()
    print_table(results)

    if args.csv:
        write_csv(results, args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
