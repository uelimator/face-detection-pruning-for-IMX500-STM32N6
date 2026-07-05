# libfacedetection.train validation

Isolated harness for running a YuNet ONNX exported by
[libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
through WIDER val and emitting predictions in the same format as the main
project pipeline (`run_yunet_wider.py`). This lets the existing WIDER eval
toolkit score it directly — useful for cross-checking the training repo's
model against the opencv_zoo distribution before doing any pruning work.

## Why this isn't in the main pipeline

`run_yunet_wider.py` uses `cv2.FaceDetectorYN`, which is hardcoded to the
opencv_zoo YuNet output schema. The libfacedetection.train export *may* match
it (same 12-output structure) or *may not* (different names, fused decode,
different anchor convention). If it matches, you could in principle reuse the
main script — but until that's verified, this isolated script is safer.

## Workflow

1. **Inspect the ONNX**:
   ```bash
   python inspect_onnx.py \
     --model /path/to/libfd_yunet.onnx
   ```
   Confirm the output structure: should be 12 outputs named
   `cls_8 / cls_16 / cls_32 / obj_* / bbox_* / kps_*`.

2. **Run inference + write predictions**:
   ```bash
   python run_libfd_inference.py \
     --model /path/to/libfd_yunet.onnx \
     --wider-root $WIDER_ROOT/WIDER_val \
     --output-dir ./out_libfd_320
   ```

3. **Score with the existing eval toolkit** (same one used for the main
   pipeline, same 320-space GT):
   ```bash
   cd ../evaluate/WiderFace-Evaluation-master
   python evaluation.py \
     --pred ../../libfd_validation/out_libfd_320/predictions \
     --gt $WIDER_ROOT/wider_face_split_320x320
   ```

   Compare the resulting Easy / Medium / Hard AP against your existing INT8
   320 numbers from `outputMISC/predictions`. Roughly similar → the training
   repo's model is consistent with what you've been deploying.

## If the inspection step shows a different schema

The decode logic in `run_libfd_inference.py` assumes the standard YuNet
anchor-free formulation:

```
cx = (grid_x + bbox[0]) * stride
cy = (grid_y + bbox[1]) * stride
 w = exp(bbox[2]) * stride
 h = exp(bbox[3]) * stride
score = sigmoid(cls) * sigmoid(obj)
```

If the libfacedetection.train export uses a different convention (different
`bbox` semantics, sigmoid baked into the ONNX itself, decode head fused into
the graph, etc.), adapt the `decode()` function. The script will fail fast
with a clear error if the output names don't match the expected schema, so
you'll know if intervention is needed before the inference loop starts.
