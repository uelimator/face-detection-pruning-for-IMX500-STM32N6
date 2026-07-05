#!/usr/bin/python3

# Mostly copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).

# This version uses the software JPEG encoder, so on Pi 4 or earlier devices,
# mjpeg_server_2.py, which will use the hardware encoder, is probably better.

import io
import logging
import socketserver
import argparse
import sys
import time
import cv2
import numpy as np

from collections import deque
from datetime import datetime
from functools import lru_cache
from http import server
from threading import Condition

from picamera2 import Picamera2, MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="320" height="320" />
</body>
</html>
"""




last_detections = []


class Detection:
    def __init__(self, coords, conf, metadata, landmarks=None):
        """Create a Detection object, recording the box, confidence and 5 landmarks."""
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)
        # Map each landmark point to main-stream pixels via a degenerate (y,x,y,x) box.
        self.landmarks = []
        for ny, nx in (landmarks or []):
            px, py, _, _ = imx500.convert_inference_coords((ny, nx, ny, nx), metadata, picam2)
            self.landmarks.append((px, py))


# YuNet is anchor-free over three feature-map strides.
# For 320x320 input the grids are 40x40 (1600), 20x20 (400), 10x10 (100).
STRIDES = (8, 16, 32)


@lru_cache(maxsize=4)
def _yunet_priors(input_w: int, input_h: int):
    """Grid-cell centres (col, row) and stride for every prior, concatenated in
    stride order so they line up with np.concatenate of the per-stride tensors."""
    cxs, cys, strides = [], [], []
    for stride in STRIDES:
        fw, fh = input_w // stride, input_h // stride
        xv, yv = np.meshgrid(np.arange(fw, dtype=np.float32),
                             np.arange(fh, dtype=np.float32))  # row-major (r outer, c inner)
        cxs.append(xv.ravel())
        cys.append(yv.ravel())
        strides.append(np.full(fw * fh, stride, dtype=np.float32))
    return np.concatenate(cxs), np.concatenate(cys), np.concatenate(strides)


def parse_detections(metadata: dict):
    """Decode the raw YuNet tensors into faces, run NMS, scale to the ISP output."""
    global last_detections
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections
    input_w, input_h = imx500.get_input_size()

    # Group the 12 tensors by type, concatenated across the three strides.
    cls  = np.concatenate([np_outputs[i][0][:, 0] for i in (0, 1, 2)])   # (2100,)
    obj  = np.concatenate([np_outputs[i][0][:, 0] for i in (3, 4, 5)])   # (2100,)
    bbox = np.concatenate([np_outputs[i][0]       for i in (6, 7, 8)])   # (2100, 4)
    lmk  = np.concatenate([np_outputs[i][0]       for i in (9, 10, 11)]) # (2100, 10)

    # Combined face score. cls/obj are assumed already sigmoid-activated in-model;
    # if scores look saturated/odd, they may be logits needing a sigmoid first.
    scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))

    # Anchor-free box decode -> top-left + size in INPUT pixels.
    cx_g, cy_g, strides = _yunet_priors(input_w, input_h)
    cx = (cx_g + bbox[:, 0]) * strides
    cy = (cy_g + bbox[:, 1]) * strides
    w  = np.exp(bbox[:, 2]) * strides
    h  = np.exp(bbox[:, 3]) * strides
    x = cx - w / 2.0
    y = cy - h / 2.0

    # Landmark decode (5 points): same anchor-free offset as the box centres.
    lmk_x = (cx_g[:, None] + lmk[:, 0::2]) * strides[:, None]  # (2100, 5) input px
    lmk_y = (cy_g[:, None] + lmk[:, 1::2]) * strides[:, None]  # (2100, 5)

    keep = scores > args.threshold
    if not np.any(keep):
        last_detections = []
        return last_detections
    x, y, w, h, scores = x[keep], y[keep], w[keep], h[keep], scores[keep]
    lmk_x, lmk_y = lmk_x[keep], lmk_y[keep]

    # Non-maximum suppression (boxes as x,y,w,h in input pixels).
    boxes_xywh = np.stack([x, y, w, h], axis=1)
    idxs = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(),
                            args.threshold, args.iou)
    idxs = np.array(idxs).reshape(-1)[:args.max_detections]

    # Hand normalized corners (y0, x0, y1, x1) and landmarks to convert_inference_coords.
    dets = []
    for i in idxs:
        nx0, ny0 = x[i] / input_w, y[i] / input_h
        nx1, ny1 = (x[i] + w[i]) / input_w, (y[i] + h[i]) / input_h
        points = [(lmk_y[i, k] / input_h, lmk_x[i, k] / input_w) for k in range(5)]
        dets.append(Detection((ny0, nx0, ny1, nx1), float(scores[i]), metadata, points))
    last_detections = dets
    return last_detections


def draw_detections(request, detections, stream="main"):
    if detections is None:
        return
    # eyes, nose, mouth-corners -> distinct colours (BGR)
    lm_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    with MappedArray(request, stream) as m:
        for d in detections:
            x, y, w, h = d.box
            label = f"face {d.conf:.2f}"
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(m.array, label, (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for (px, py), color in zip(d.landmarks, lm_colors):
                cv2.circle(m.array, (px, py), 2, color, -1)


# ---- performance metrics (runs in the camera callback thread) ----
_frame_intervals = deque(maxlen=60)   # rolling window for FPS
_last_frame_t = None
# HUD rolling window for DNN ms — bigger than FPS window so p99 is meaningful.
_hud_dnn = deque(maxlen=200)          # ~7 s at 30 fps
# Per-flush accumulators: lists so we can compute p95/p99 at flush time.
_log_accum = {"dnn": [], "dsp": [], "pp": [], "faces": [], "n": 0}
_last_log_t = None
_log_fh = None


def metrics_init(path):
    """Open the metrics log (append) and start the averaging window."""
    global _log_fh, _last_log_t
    _log_fh = open(path, "a", buffering=1)  # line-buffered
    _log_fh.write(f"# session start {datetime.now().isoformat(timespec='seconds')}\n")
    _last_log_t = time.perf_counter()


def _avg(vs):
    return sum(vs) / len(vs) if vs else 0.0


def _pct(vs, p):
    """Nearest-rank percentile (no interpolation) — robust at any sample size."""
    if not vs:
        return 0.0
    s = sorted(vs)
    idx = min(len(s) - 1, int(len(s) * p / 100))
    return s[idx]


def _maybe_flush_log(now, fps):
    """Every --log-interval seconds, append per-window stats and reset."""
    global _last_log_t
    if _log_fh is None or now - _last_log_t < args.log_interval:
        return
    dnn, dsp, pp, faces = _log_accum["dnn"], _log_accum["dsp"], _log_accum["pp"], _log_accum["faces"]
    _log_fh.write(
        f"{datetime.now().isoformat(timespec='seconds')} "
        f"dnn_avg={_avg(dnn):.2f}ms dnn_p95={_pct(dnn,95):.2f}ms dnn_p99={_pct(dnn,99):.2f}ms "
        f"dsp_avg={_avg(dsp):.2f}ms "
        f"pp_avg={_avg(pp):.2f}ms pp_p95={_pct(pp,95):.2f}ms pp_p99={_pct(pp,99):.2f}ms "
        f"fps={fps:.1f} "
        f"faces_avg={_avg(faces):.2f} "
        f"frames={_log_accum['n']}\n"
    )
    _log_accum["dnn"].clear()
    _log_accum["dsp"].clear()
    _log_accum["pp"].clear()
    _log_accum["faces"].clear()
    _log_accum["n"] = 0
    _last_log_t = now


def inference_callback(request):
    """Parse + time detections, draw boxes/landmarks, overlay HUD, feed the logger."""
    global _last_frame_t
    metadata = request.get_metadata()

    t0 = time.perf_counter()
    detections = parse_detections(metadata)
    pp_ms = (time.perf_counter() - t0) * 1000

    kpi = imx500.get_kpi_info(metadata)            # (dnn_ms, dsp_ms) or None
    dnn_ms, dsp_ms = kpi if kpi else (0.0, 0.0)

    now = time.perf_counter()
    if _last_frame_t is not None:
        _frame_intervals.append(now - _last_frame_t)
    _last_frame_t = now
    fps = len(_frame_intervals) / sum(_frame_intervals) if _frame_intervals else 0.0

    _hud_dnn.append(dnn_ms)
    _log_accum["dnn"].append(dnn_ms)
    _log_accum["dsp"].append(dsp_ms)
    _log_accum["pp"].append(pp_ms)
    _log_accum["faces"].append(len(detections))
    _log_accum["n"] += 1
    _maybe_flush_log(now, fps)

    # HUD: rolling DNN avg/p95/p99 from _hud_dnn window so the numbers are stable.
    hud_window = list(_hud_dnn)
    dnn_avg_hud = _avg(hud_window)
    dnn_p95_hud = _pct(hud_window, 95)
    dnn_p99_hud = _pct(hud_window, 99)

    draw_detections(request, detections)
    with MappedArray(request, "main") as m:
        hud = (f"DNN avg {dnn_avg_hud:4.1f} p95 {dnn_p95_hud:4.1f} p99 {dnn_p99_hud:4.1f} ms"
               f" | DSP {dsp_ms:4.1f} PP {pp_ms:4.1f} | {fps:4.1f}fps")
        cv2.putText(m.array, hud, (5, m.array.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default="./out/network.rpk",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    # TODO(NMS): wired up once parse_detections runs NMS on the raw YuNet boxes
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold for NMS")
    parser.add_argument("--max-detections", type=int, default=10, help="Max detections to keep after NMS")
    parser.add_argument(
        "-r",
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="preserve the pixel aspect ratio of the input tensor",
    )
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")
    parser.add_argument("--log-file", type=str, default="inference_metrics.log",
                        help="Append average performance metrics to this file")
    parser.add_argument("--log-interval", type=float, default=5.0,
                        help="Seconds between averaged metric log lines")
    return parser.parse_args()





# Here is streaming output. Use then for streaming model to server
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


"""
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    picam2.stop_recording()"""

if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args (single-class face model needs no labels file)
    for key, value in vars(args).items():
        if hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()


    #here we change some stuff so it gets streamed
    picam2 = Picamera2(imx500.camera_num)
    #What is the difference between create preview and video?
    #config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    #config = picam2.create_video_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12,main={"size": (320,320)})
    config = picam2.create_video_configuration(
        main={"size": (320, 320), "format": "RGB888"},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )
    output = StreamingOutput()

    imx500.show_network_fw_progress_bar()
    #Here we change this:
    #picam2.start(config, show_preview=True)
    metrics_init(args.log_file)
    picam2.pre_callback = inference_callback
    picam2.configure(config)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    picam2.start_recording(JpegEncoder(), FileOutput(output))


    print(f"Streaming on http://<pi-ip>:8000/  (Ctrl-C to stop)")
    print(f"Logging averaged metrics every {args.log_interval}s to {args.log_file}")
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        picam2.stop_recording()
        if _log_fh:
            _log_fh.close()



"""
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    while True:
        last_results = parse_detections(picam2.capture_metadata())"""
