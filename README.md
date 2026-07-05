# face-detection-pruning

Structured channel pruning + knowledge-distillation recovery + target-neutral / vendor-specific INT8 quantization for YuNet face detection, targeting deployment on the Sony IMX500 (Raspberry Pi AI Camera) and STM32N6 Neural-ART NPU.

## Repository layout

```
pruning/            structured filter pruning + KD-recovery finetune
quantize/           target-neutral ONNX Runtime QDQ INT8 quantization
sony_reexport/      IMX500-targeted quantization via Sony MCT
calibration/        build calibration tensor sets from WIDER FACE
evaluate/           PC-side benchmarks + before/after plots + WIDER Face eval toolkit
libfd_validation/   WIDER-mAP validation harness for ONNX exports
IMX500/             Raspberry Pi AI Camera streaming app + metrics parser
```

## External dependencies

The scripts reference three things outside this repo:

1. **libfacedetection.train** — the FP32 baseline YuNet checkpoint and ONNX export.
   Clone from https://github.com/ShiqiYu/libfacedetection.train and note the paths to
   `weights/yunet_n.pth` and `onnx/yunet_n_320_320.onnx`. Several scripts default
   to looking for these at `../training/libfacedetection.train/...` — override via
   CLI flags.
2. **WIDER FACE dataset** — http://shuoyang1213.me/WIDERFACE/. Set the `WIDER_ROOT`
   environment variable to your download location. Expected layout:
   ```
   $WIDER_ROOT/WIDER_train/            (training images)
   $WIDER_ROOT/WIDER_val/               (validation images)
   $WIDER_ROOT/wider_face_split/        (annotations)
   $WIDER_ROOT/ground_truth_320/        (regenerated 320x320 GT .mat files — see evaluate/transform_annotations_mat.py)
   ```
3. **STM32N6 model zoo** (optional, deployment only) — https://github.com/STMicroelectronics/stm32ai-modelzoo-services

## Python environments

Two environment "flavours" are useful because the Sony MCT path has heavy deps
that not every user needs:

```bash
# Core pruning + quantization
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Additionally, for IMX500-targeted MCT quantization
pip install -r requirements-sony.txt
```

`requirements.txt` should cover: `torch`, `torch_pruning`, `numpy`, `onnx`,
`onnxruntime`, `opencv-python`, `matplotlib`, `pandas`, `tqdm`, `scipy`.

`requirements-sony.txt` adds: `model_compression_toolkit`, `edgemdt_tpc`,
`mct_quantizers`, `onnx2torch`, `OpenEXR`.

## Workflow (end-to-end)

```
                    FP32 ONNX (from libfacedetection.train)
                              │
                    ┌─────────┼──────────────────┐
                    ↓         ↓                  ↓
             quantize/  pruning/yunet_prune.py  (baseline eval)
                │             │
                │             ↓  (KD-recovered, structurally pruned)
                │      pruning/export_pruned_to_onnx.py
                │             │
                ↓             ↓
      ORT QDQ INT8 ONNX  → both go through quantize_onnxrt.py
                                │
                                ├──────► STM32N6 (via ST cloud benchmark or on-target deploy)
                                │
                                └──────► IMX500 via sony_reexport/quantize_yunet.py
                                                → imxconv-pt → .rpk
```

## Key scripts

### Pruning
- `pruning/yunet_prune.py` — iterative structured pruning with knowledge-distillation recovery
- `pruning/prune_yunet.py` — single-shot variant, no finetune
- `pruning/plot_pruning_run.py` — training-curve plots from `pruning/logs/`
- `pruning/plot_pruned_architecture.py` — before/after channel + parameter diagrams

### Quantization
- `quantize/quantize_onnxrt.py` — ORT QDQ static INT8 (target-neutral, works on STM32N6 and PC)
- `sony_reexport/quantize_yunet.py` — Sony MCT INT8 with IMX500 target platform capabilities

### Evaluation
- `evaluate/benchmark_models.py` — PC-side ONNX size / params / MACs / latency (p50/p95/p99) comparison
- `libfd_validation/run_libfd_inference.py` + `evaluate/WiderFace-Evaluation-master/evaluation.py` — WIDER Face mAP scoring
- `evaluate/plot_wider_transform.py` — visualize the WIDER → 320x320 dataset transformation

### Deployment (IMX500 side)
- `IMX500/model_stream.py` — picamera2 app: MJPEG stream over HTTP with face-detection overlays + performance HUD (avg/p95/p99 DNN latency)
- `IMX500/parse_metrics.py` — session-level statistics from the metrics log
