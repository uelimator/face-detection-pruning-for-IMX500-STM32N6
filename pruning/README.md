# Pruning workspace

Standalone-PyTorch pruning experiments for YuNet. Self-contained — no mmdet
or mmcv install required, just torch + numpy + onnx + onnxruntime.

## Layout

```
pruning/
    yunet_standalone/       Pure-PyTorch port of libfacedetection.train's YuNet
        __init__.py
        layers.py           ConvDPUnit, Conv_head, Conv4layerBlock
        backbone.py         YuNetBackbone (6 stages, strides 8/16/32)
        neck.py             TFPN (top-down feature pyramid)
        head.py             YuNet_Head (cls/obj/bbox/kps × 3 strides)
        model.py            YuNet wrapper + YUNET_N_CFG architecture params
    load_and_verify.py      Build model, load .pth, compare vs ONNX
    README.md               This file
```

## Setup

Add torch to your existing 3.11 venv (everything else is already there):

```bash
.venv/bin/pip install torch
```

That's it — no mmdet, no mmcv-full, no pinned-version compatibility juggling.

## Verify the load

```bash
cd /pruning

python load_and_verify.py \
    --weights ../training/libfacedetection.train/weights/yunet_n.pth \
    --onnx ../training/libfacedetection.train/onnx/yunet_n_320_320.onnx
```

Expected output: state_dict loads with 0 missing/unexpected keys, all 12
output tensors match the ONNX with cosine ≥ 0.999. If you see this, the
standalone model is faithful to the deployed ONNX — safe pruning starting
point.

## Architecture summary (yunet_n config)

- **Backbone**: 6 stages, [3,16,16] → [16,64] → 4× [64,64] with selective
  2× downsampling. Outputs at stages 3/4/5 give feature maps at strides
  8 / 16 / 32 from a 320×320 input.
- **Neck (TFPN)**: top-down 3-level feature pyramid with depthwise lateral
  convs (`ConvDPUnit`). Same channel count (64) at every level.
- **Head**: one shared `ConvDPUnit` per scale, then four parallel
  per-scale sub-heads producing cls (1 ch), obj (1 ch), bbox (4 ch),
  kps (10 ch). Sigmoid is applied to cls and obj inside the forward —
  matching the ONNX export — so the 12 output tensors are directly
  comparable to the deployed model.
- **Total params**: ~75,856 (very small, hence the "nano" variant name).

## Next: write the pruning script

`yunet_standalone.YuNet` is already a clean `nn.Module` with a single
`forward(x) -> tuple[Tensor, ...]` signature. Torch-Pruning can trace it
without any wrapper:

```python
import torch
import torch_pruning as tp
from yunet_standalone import YuNet

model = YuNet()
model.load_pretrained("../training/libfacedetection.train/weights/yunet_n.pth")
model.eval()

example = torch.zeros(1, 3, 320, 320)
imp = tp.importance.MagnitudeImportance(p=2)

# Protect the final per-stride output convs — pruning their output channels
# would change the model's contract (1/4/10 channels per head).
ignored = list(model.bbox_head.multi_level_cls) \
        + list(model.bbox_head.multi_level_obj) \
        + list(model.bbox_head.multi_level_bbox) \
        + list(model.bbox_head.multi_level_kps)

pruner = tp.pruner.MagnitudePruner(
    model, example_inputs=example, importance=imp,
    pruning_ratio=0.3, ignored_layers=ignored,
)
pruner.step()
```

Hold off on running the actual pruning until you've decided on a finetune
loop (the model needs to recover from any non-trivial pruning ratio, and
WIDER train labels need to be wired into a dataloader). That's the
next-next step.

## Re-export to ONNX after pruning

The standalone model can export back to ONNX with names matching the
opencv_zoo / N6 deployment schema:

```python
torch.onnx.export(
    model, torch.zeros(1, 3, 320, 320), "pruned_yunet.onnx",
    input_names=["input"],
    output_names=[
        "cls_8", "cls_16", "cls_32",
        "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32",
        "kps_8", "kps_16", "kps_32",
    ],
    opset_version=11,
)
```

The exported ONNX feeds directly into your existing quantize → eval →
STEdgeAI / imx500-converter pipeline.
