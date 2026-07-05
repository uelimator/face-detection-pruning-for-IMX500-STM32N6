#!/usr/bin/python3
"""Summarise an inference_metrics.log produced by model_stream.py.

Each data line looks like:
  2026-05-26T14:03:15 dnn_avg=4.21ms dsp_avg=1.08ms pp_avg=0.79ms fps=29.7 faces_avg=1.30 frames=148

Sessions are delimited by '# session start <timestamp>' markers. For every
session (and the whole file) we report min / mean / max per metric. The mean is
weighted by the 'frames' count of each window, so it reflects the true average
inference time rather than an average-of-averages.
"""
import argparse
import re
from collections import defaultdict

KV = re.compile(r"(\w+)=([0-9.]+)")
METRICS = ["dnn_avg", "dsp_avg", "pp_avg", "fps", "faces_avg"]
LABELS = {"dnn_avg": "dnn (ms)", "dsp_avg": "dsp (ms)", "pp_avg": "pp  (ms)",
          "fps": "fps", "faces_avg": "faces"}


def parse(path):
    """Return [(session_name, [window_dicts...]), ...] in file order."""
    sessions = []
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# session start"):
                current = (line[len("# session start"):].strip() or "unnamed", [])
                sessions.append(current)
                continue
            kv = {k: float(v) for k, v in KV.findall(line)}
            if "frames" not in kv:
                continue
            if current is None:                       # data before any marker
                current = ("unnamed", [])
                sessions.append(current)
            current[1].append(kv)
    return sessions


def summarise(windows):
    """min / weighted-mean / max per metric over a list of window dicts."""
    out = {}
    total_frames = sum(w.get("frames", 0) for w in windows) or 1
    for m in METRICS:
        vals = [w[m] for w in windows if m in w]
        if not vals:
            continue
        weighted = sum(w[m] * w.get("frames", 0) for w in windows if m in w)
        out[m] = (min(vals), weighted / total_frames, max(vals))
    return out, int(total_frames)


def print_block(title, windows):
    stats, frames = summarise(windows)
    print(f"\n{title}  ({len(windows)} windows, {frames} frames)")
    if not stats:
        print("  (no data)")
        return
    print(f"  {'metric':<10}{'min':>8}{'mean':>9}{'max':>9}")
    for m in METRICS:
        if m in stats:
            lo, mean, hi = stats[m]
            print(f"  {LABELS[m]:<10}{lo:>8.2f}{mean:>9.2f}{hi:>9.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="inference_metrics.log",
                    help="metrics log file (default: inference_metrics.log)")
    ap.add_argument("--per-session", action="store_true",
                    help="also break the summary down per session")
    args = ap.parse_args()

    try:
        sessions = parse(args.log)
    except FileNotFoundError:
        raise SystemExit(f"Log file not found: {args.log}")
    all_windows = [w for _, ws in sessions for w in ws]
    if not all_windows:
        print(f"No metric data found in {args.log}")
        return

    if args.per_session:
        for name, windows in sessions:
            if windows:
                print_block(f"Session {name}", windows)

    print_block(f"OVERALL ({args.log})", all_windows)


if __name__ == "__main__":
    main()
