"""Visualise what structured pruning removed from YuNet.

Loads the unpruned baseline checkpoint and the pruned-model pickle, walks every
Conv2d layer in declaration order, and produces:

    architecture/channels.png         out_channels per conv: baseline vs pruned
    architecture/ratio.png            per-layer % channel reduction
    architecture/params.png           per-layer parameter count (baseline vs pruned)
    architecture/summary.png          all three + totals/text panel
    architecture/layer_table.csv      raw data behind the plots

Usage:
    python plot_pruned_architecture.py
    python plot_pruned_architecture.py --baseline yunet_n_baseline.ckpt --pruned pruned_yunet_structured_20_06.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# yunet_standalone is in the same dir; ensure it's importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yunet_standalone import YuNet  # noqa: E402


# ---------------------------------------------------------------------------

def _stage_of(name: str) -> str:
    """Group a layer name into a coarse architectural stage for grouping/colour."""
    if name.startswith("backbone.model"):
        return name.split(".")[1]  # model0, model1, ...
    if name.startswith("neck"):
        return "neck"
    if name.startswith("head"):
        return "head"
    return "other"


def model_totals(model: nn.Module) -> dict:
    """Whole-model stats: total params (all layers, not just convs) + per-type leaf-module counts."""
    total_params = sum(p.numel() for p in model.parameters())
    types = Counter(type(m).__name__
                    for m in model.modules() if len(list(m.children())) == 0)
    leaf_modules = sum(types.values())
    return {
        "total_params": total_params,
        "leaf_modules": leaf_modules,
        "conv2d":       types.get("Conv2d", 0),
        "bn2d":         types.get("BatchNorm2d", 0),
        "relu":         types.get("ReLU", 0) + types.get("ReLU6", 0),
        "linear":       types.get("Linear", 0),
        "by_type":      dict(types),
    }


def onnx_init_totals(onnx_path: Path | None) -> dict | None:
    """Count initializer elements + bytes from an ONNX file (the same numbers
    benchmark_models.py reports). Returns None if path is missing.

    These differ from torch model.parameters() because ONNX export folds Conv+BN,
    absorbing BN gamma/beta into Conv weights. For INT8 QDQ models, the count
    additionally includes per-channel scale + zero-point initializers.
    """
    if onnx_path is None or not onnx_path.exists():
        return None
    try:
        import onnx as _onnx
    except ImportError:
        return None
    DT_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
    m = _onnx.load(str(onnx_path))
    elems = 0
    bytes_ = 0
    for init in m.graph.initializer:
        prod = 1
        for d in init.dims:
            prod *= d
        elems += prod
        bytes_ += prod * DT_BYTES.get(init.data_type, 4)
    return {"path": str(onnx_path), "elems": elems, "bytes": bytes_}


def collect_conv_info(model: nn.Module) -> list[dict]:
    """Walk a model and return a list of dicts describing each Conv2d in declaration order."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            params = sum(p.numel() for p in m.parameters())
            rows.append({
                "name": name,
                "in_channels": m.in_channels,
                "out_channels": m.out_channels,
                "kh": m.kernel_size[0],
                "kw": m.kernel_size[1],
                "params": params,
                "stage": _stage_of(name),
            })
    return rows


def build_dataframe(baseline_info: list[dict], pruned_info: list[dict]) -> pd.DataFrame:
    """Pair baseline and pruned conv lists by name and produce a DataFrame."""
    base_by_name = {r["name"]: r for r in baseline_info}
    pruned_by_name = {r["name"]: r for r in pruned_info}

    common = [n for n in base_by_name if n in pruned_by_name]
    only_base = set(base_by_name) - set(common)
    only_pruned = set(pruned_by_name) - set(common)
    if only_base or only_pruned:
        print(f"  warning: layer-name mismatch — "
              f"baseline-only: {len(only_base)}, pruned-only: {len(only_pruned)}")

    rows = []
    for name in common:
        b = base_by_name[name]
        p = pruned_by_name[name]
        rows.append({
            "name": name,
            "stage": b["stage"],
            "kernel": f"{b['kh']}×{b['kw']}",
            "baseline_in":  b["in_channels"],
            "baseline_out": b["out_channels"],
            "baseline_params": b["params"],
            "pruned_in":  p["in_channels"],
            "pruned_out": p["out_channels"],
            "pruned_params": p["params"],
        })
    df = pd.DataFrame(rows)
    df["delta_out"] = df["baseline_out"] - df["pruned_out"]
    df["ratio_out"] = 1.0 - df["pruned_out"] / df["baseline_out"]
    df["delta_params"] = df["baseline_params"] - df["pruned_params"]
    df["ratio_params"] = np.where(df["baseline_params"] > 0,
                                  1.0 - df["pruned_params"] / df["baseline_params"], 0.0)
    df["layer_idx"] = range(len(df))
    return df


# --- plotting --------------------------------------------------------------

_STAGE_PALETTE = {
    "model0": "#7fbfff", "model1": "#5fa4e6", "model2": "#3f88cc",
    "model3": "#246cb3", "model4": "#175499", "model5": "#0d3d80",
    "neck":   "#ff9248",
    "head":   "#7a7a7a",   # untouched in our pruning recipe — visually neutral
}


def _stage_color(stage: str) -> str:
    return _STAGE_PALETTE.get(stage, "#bbbbbb")


def _stage_boundaries(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Return [(start_idx, stage_label), ...] for drawing background spans."""
    boundaries = []
    cur = None
    for i, s in enumerate(df["stage"]):
        if s != cur:
            boundaries.append((i, s))
            cur = s
    return boundaries


def _draw_stage_spans(ax, df: pd.DataFrame, ymax_fraction: float = 0.04) -> None:
    """Shade backgrounds for each stage and label them above the axis."""
    bounds = _stage_boundaries(df)
    bounds_with_end = bounds + [(len(df), None)]
    ymax = ax.get_ylim()[1]
    for (start, stage), (end, _) in zip(bounds, bounds_with_end[1:]):
        ax.axvspan(start - 0.5, end - 0.5, color=_stage_color(stage), alpha=0.12, zorder=0)
        ax.text((start + end - 1) / 2, ymax * (1 - ymax_fraction), stage,
                ha="center", va="top", fontsize=8, color="#444444")


def plot_channels(df: pd.DataFrame, ax) -> None:
    width = 0.42
    x = df["layer_idx"].values
    ax.bar(x - width/2, df["baseline_out"], width=width,
           color="lightgray", edgecolor="gray", linewidth=0.5, label="baseline")
    ax.bar(x + width/2, df["pruned_out"], width=width,
           color=[_stage_color(s) for s in df["stage"]], edgecolor="black",
           linewidth=0.3, label="after pruning")
    ax.set_xlabel("conv layer (declaration order)")
    ax.set_ylabel("out_channels")
    ax.set_title("out_channels per Conv2d: baseline vs pruned")
    ax.set_xlim(-1, len(df))
    ax.grid(True, alpha=0.3, axis="y")
    _draw_stage_spans(ax, df)
    ax.legend(loc="upper right", framealpha=0.9)


def plot_ratio(df: pd.DataFrame, ax) -> None:
    x = df["layer_idx"].values
    ratios = df["ratio_out"].values * 100
    colors = [_stage_color(s) if r > 0 else "#dddddd"
              for s, r in zip(df["stage"], ratios)]
    ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("conv layer (declaration order)")
    ax.set_ylabel("channels removed (%)")
    ax.set_title("per-layer channel reduction (0% = untouched)")
    ax.set_xlim(-1, len(df))
    ax.set_ylim(0, max(40, ratios.max() * 1.1))
    ax.grid(True, alpha=0.3, axis="y")
    _draw_stage_spans(ax, df)


def build_stage_digest(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stage table: max + sum out_channels, params, reduction percentages."""
    agg = df.groupby("stage", sort=False).agg(
        layers=("name", "count"),
        baseline_max_ch=("baseline_out", "max"),
        pruned_max_ch=("pruned_out", "max"),
        baseline_sum_ch=("baseline_out", "sum"),
        pruned_sum_ch=("pruned_out", "sum"),
        baseline_params=("baseline_params", "sum"),
        pruned_params=("pruned_params", "sum"),
    )
    agg["ch_reduction"] = 1 - agg["pruned_max_ch"] / agg["baseline_max_ch"]
    agg["param_reduction"] = 1 - agg["pruned_params"] / agg["baseline_params"]
    return agg


def render_stage_digest_text(digest: pd.DataFrame) -> str:
    """Plain-text per-stage digest — print and also save to disk."""
    lines = [
        f"{'stage':<10} {'layers':>6}  {'max ch (base->pruned)':>25}  {'params (base->pruned)':>26}  {'ch red.':>8}  {'param red.':>10}",
        "-" * 95,
    ]
    for stage, row in digest.iterrows():
        lines.append(
            f"{stage:<10} {int(row['layers']):>6}  "
            f"{int(row['baseline_max_ch']):>10} -> {int(row['pruned_max_ch']):>10}  "
            f"{int(row['baseline_params']):>11,} -> {int(row['pruned_params']):>11,}  "
            f"{row['ch_reduction']:>7.1%}  {row['param_reduction']:>9.1%}"
        )
    return "\n".join(lines)


def plot_stage_digest(digest: pd.DataFrame, ax) -> None:
    """Horizontal grouped bars per stage — baseline max-channels vs pruned max-channels."""
    y = np.arange(len(digest))
    bw = 0.4
    ax.barh(y - bw/2, digest["baseline_max_ch"], height=bw,
            color="lightgray", edgecolor="gray", linewidth=0.5, label="baseline")
    ax.barh(y + bw/2, digest["pruned_max_ch"], height=bw,
            color=[_stage_color(s) for s in digest.index], edgecolor="black",
            linewidth=0.3, label="after pruning")
    ax.set_yticks(y)
    ax.set_yticklabels(digest.index)
    ax.invert_yaxis()   # so first stage shows at top
    ax.set_xlabel("max out_channels in stage")
    ax.set_title("per-stage channel width: baseline vs pruned")
    # Annotate each pair with absolute numbers
    for i, (_, row) in enumerate(digest.iterrows()):
        ax.text(row["baseline_max_ch"] + 1, i - bw/2,
                f"{int(row['baseline_max_ch'])}",
                va="center", fontsize=8, color="gray")
        ax.text(row["pruned_max_ch"] + 1, i + bw/2,
                f"{int(row['pruned_max_ch'])}  ({row['ch_reduction']:.0%}↓)",
                va="center", fontsize=8, color="black", weight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0, digest["baseline_max_ch"].max() * 1.3)


def plot_tower_diagram(digest: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side 'tower' diagram: rectangle widths ∝ channel count.

    Backbone stages stacked vertically; neck + head shown at bottom as wide blocks.
    Two columns (baseline left, pruned right) for direct visual comparison.
    """
    backbone = [s for s in digest.index if s.startswith("model")]
    rest = [s for s in digest.index if not s.startswith("model")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
    max_ch = max(digest["baseline_max_ch"].max(), digest["pruned_max_ch"].max())
    # use a unit scale so the widest box is ~6 units; same scale on both axes
    scale = 6.0 / max_ch

    def draw_tower(ax, ch_col: str, title: str) -> None:
        ax.set_xlim(-7, 7)
        ax.set_ylim(0, len(backbone) + 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        # Hide spines for a cleaner paper-style look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # input arrow at top
        ax.annotate("input 3×320×320", xy=(0, len(backbone) + 1.9),
                    ha="center", va="bottom", fontsize=9, color="#444")
        ax.annotate("", xy=(0, len(backbone) + 1.6), xytext=(0, len(backbone) + 1.9),
                    arrowprops=dict(arrowstyle="->", color="#444"))

        # backbone tower (top to bottom)
        for i, stage in enumerate(backbone):
            row = digest.loc[stage]
            ch = int(row[ch_col])
            half_w = ch * scale / 2
            y = len(backbone) - i + 0.5
            color = _stage_color(stage)
            rect = plt.Rectangle((-half_w, y - 0.4), 2 * half_w, 0.7,
                                  facecolor=color, edgecolor="black", linewidth=0.6)
            ax.add_patch(rect)
            ax.text(0, y - 0.05, f"{stage}", ha="center", va="center",
                    fontsize=10, weight="bold", color="white" if ch > 30 else "black")
            ax.text(half_w + 0.3, y - 0.05, f"{ch} ch",
                    ha="left", va="center", fontsize=9)
            # arrow to next stage
            if i < len(backbone) - 1:
                ax.annotate("", xy=(0, y - 0.5), xytext=(0, y - 0.4),
                            arrowprops=dict(arrowstyle="->", color="#777"))

        # neck + head as a single wider block beneath the backbone
        neck_y = 0.6
        if "neck" in digest.index:
            row = digest.loc["neck"]
            ch = int(row[ch_col])
            half_w = ch * scale / 2
            ax.add_patch(plt.Rectangle((-half_w, neck_y - 0.2), 2 * half_w, 0.5,
                                       facecolor=_stage_color("neck"),
                                       edgecolor="black", linewidth=0.6))
            ax.text(0, neck_y + 0.05, "neck (TFPN)", ha="center", va="center",
                    fontsize=9, weight="bold")
            ax.text(half_w + 0.3, neck_y + 0.05, f"{ch} ch",
                    ha="left", va="center", fontsize=9)
        # head label
        ax.text(0, 0.1, "head (cls/obj/bbox/kps × strides 8/16/32)",
                ha="center", va="center", fontsize=8,
                style="italic", color="#555")

    draw_tower(axes[0], "baseline_max_ch", "BASELINE")
    draw_tower(axes[1], "pruned_max_ch",   "PRUNED")
    fig.suptitle("YuNet macro architecture: baseline vs pruned (box width ∝ max channels)",
                 fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_params(df: pd.DataFrame, ax) -> None:
    width = 0.42
    x = df["layer_idx"].values
    ax.bar(x - width/2, df["baseline_params"] / 1000, width=width,
           color="lightgray", edgecolor="gray", linewidth=0.5, label="baseline")
    ax.bar(x + width/2, df["pruned_params"] / 1000, width=width,
           color=[_stage_color(s) for s in df["stage"]], edgecolor="black",
           linewidth=0.3, label="after pruning")
    ax.set_xlabel("conv layer (declaration order)")
    ax.set_ylabel("parameters (×10³)")
    ax.set_title("parameter count per Conv2d: baseline vs pruned")
    ax.set_xlim(-1, len(df))
    ax.grid(True, alpha=0.3, axis="y")
    _draw_stage_spans(ax, df)
    ax.legend(loc="upper right", framealpha=0.9)


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, default=HERE / "yunet_n_baseline.ckpt")
    ap.add_argument("--pruned",   type=Path, default=HERE / "pruned_yunet_structured_20_06.pt")
    ap.add_argument("--baseline-onnx", type=Path,
                    default=HERE.parent / "training/libfacedetection.train/onnx/yunet_n_320_320.onnx",
                    help="Optional: the corresponding FP32 ONNX file to also report ONNX-init param counts (matches benchmark_models.py).")
    ap.add_argument("--pruned-onnx", type=Path,
                    default=HERE / "pruned_yunet_structured.onnx",
                    help="Optional: the corresponding FP32 pruned ONNX file.")
    ap.add_argument("--out",      type=Path, default=HERE / "architecture")
    args = ap.parse_args()

    if not args.baseline.exists():
        sys.exit(f"baseline not found: {args.baseline}")
    if not args.pruned.exists():
        sys.exit(f"pruned not found: {args.pruned}")

    # Baseline = state_dict in a checkpoint
    print(f"loading baseline:  {args.baseline.name}")
    ckpt = torch.load(args.baseline, weights_only=True, map_location="cpu")
    baseline = YuNet()
    baseline.load_state_dict(ckpt["model_state_dict"])

    # Pruned = full module pickle (torch.save(model)). Needs weights_only=False
    # because torch_pruning swapped in new Conv2d objects whose architecture
    # diverges from YUNET_N_CFG and can't be reconstructed from state_dict alone.
    print(f"loading pruned:    {args.pruned.name}")
    pruned = torch.load(args.pruned, weights_only=False, map_location="cpu")

    base_info = collect_conv_info(baseline)
    pruned_info = collect_conv_info(pruned)
    print(f"baseline convs: {len(base_info)}   pruned convs: {len(pruned_info)}")

    base_tot = model_totals(baseline)
    pruned_tot = model_totals(pruned)
    print(f"baseline total params: {base_tot['total_params']:,}   leaf modules: {base_tot['leaf_modules']}")
    print(f"pruned   total params: {pruned_tot['total_params']:,}   leaf modules: {pruned_tot['leaf_modules']}")

    base_onnx = onnx_init_totals(args.baseline_onnx)
    pruned_onnx = onnx_init_totals(args.pruned_onnx)
    if base_onnx:
        print(f"baseline ONNX init elems (matches benchmark): {base_onnx['elems']:,}")
    if pruned_onnx:
        print(f"pruned   ONNX init elems (matches benchmark): {pruned_onnx['elems']:,}")

    df = build_dataframe(base_info, pruned_info)
    args.out.mkdir(parents=True, exist_ok=True)

    # Raw data alongside the plots
    df.to_csv(args.out / "layer_table.csv", index=False)
    print(f"wrote {args.out / 'layer_table.csv'}")

    # Individual figures
    for name, fn in [("channels", plot_channels), ("ratio", plot_ratio),
                     ("params",   plot_params)]:
        fig, ax = plt.subplots(figsize=(14, 5))
        fn(df, ax)
        fig.tight_layout()
        p = args.out / f"{name}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print(f"wrote {p}")

    # Per-stage digest: text + figure
    digest = build_stage_digest(df)
    digest_text = render_stage_digest_text(digest)
    print("\n" + digest_text + "\n")
    (args.out / "stage_digest.txt").write_text(digest_text + "\n")
    print(f"wrote {args.out / 'stage_digest.txt'}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_stage_digest(digest, ax)
    fig.tight_layout()
    p = args.out / "stage_digest.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"wrote {p}")

    # Node-graph / tower diagram
    plot_tower_diagram(digest, args.out / "tower_diagram.png")
    print(f"wrote {args.out / 'tower_diagram.png'}")

    # Combined summary
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    plot_channels(df, axes[0])
    plot_ratio(df, axes[1])
    plot_params(df, axes[2])

    # Totals text overlay in the corner of the channels plot
    total_b_conv_params = df["baseline_params"].sum()
    total_p_conv_params = df["pruned_params"].sum()
    total_b_ch     = df["baseline_out"].sum()
    total_p_ch     = df["pruned_out"].sum()
    layers_pruned  = int((df["ratio_out"] > 0).sum())
    summary_lines = [
        f"{'':<24} {'baseline':>10}  ->  {'pruned':>10}",
        "-" * 56,
        f"{'TORCH VIEW':<24}",
        f"{'  total model params':<24} {base_tot['total_params']:>10,}  ->  {pruned_tot['total_params']:>10,}"
        f"   ({1 - pruned_tot['total_params']/base_tot['total_params']:>5.1%})",
        f"{'  conv2d params':<24} {total_b_conv_params:>10,}  ->  {total_p_conv_params:>10,}"
        f"   ({1 - total_p_conv_params/total_b_conv_params:>5.1%})",
        f"{'  sum of out_channels':<24} {total_b_ch:>10,}  ->  {total_p_ch:>10,}"
        f"   ({1 - total_p_ch/total_b_ch:>5.1%})",
    ]
    if base_onnx and pruned_onnx:
        summary_lines += [
            "",
            f"{'ONNX VIEW (BN folded)':<24}",
            f"{'  init params':<24} {base_onnx['elems']:>10,}  ->  {pruned_onnx['elems']:>10,}"
            f"   ({1 - pruned_onnx['elems']/base_onnx['elems']:>5.1%})",
            f"{'  init bytes':<24} {base_onnx['bytes']:>10,}  ->  {pruned_onnx['bytes']:>10,}"
            f"   ({1 - pruned_onnx['bytes']/base_onnx['bytes']:>5.1%})",
            "  (these are what benchmark_models.py reports as 'All Params' / 'Weights')",
        ]
    summary_lines += [
        "",
        f"{'LAYER COUNTS':<24}",
        f"{'  leaf modules total':<24} {base_tot['leaf_modules']:>10,}  ->  {pruned_tot['leaf_modules']:>10,}",
        f"{'  Conv2d layers':<24} {base_tot['conv2d']:>10,}  ->  {pruned_tot['conv2d']:>10,}",
        f"{'  BatchNorm2d layers':<24} {base_tot['bn2d']:>10,}  ->  {pruned_tot['bn2d']:>10,}",
        f"{'  ReLU layers':<24} {base_tot['relu']:>10,}  ->  {pruned_tot['relu']:>10,}",
        "",
        f"layers actually pruned: {layers_pruned:>3d} of {len(df):d} Conv2d",
    ]
    summary = "\n".join(summary_lines)
    axes[0].text(0.99, 0.97, summary, transform=axes[0].transAxes, fontsize=8,
                 family="monospace", ha="right", va="top",
                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    fig.suptitle("YuNet structured pruning: per-layer before/after", fontsize=14)
    fig.tight_layout()
    p = args.out / "summary.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
