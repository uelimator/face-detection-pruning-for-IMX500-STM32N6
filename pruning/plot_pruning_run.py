"""Plot the loss curve and AP-vs-sparsity story from a pruning run's logs.

Reads loss_log.csv and metrics_log.csv (the two files yunet_prune.py writes
under logs/<TIMESTAMP>/), and produces four PNGs into <run>/plots/:

    loss_curve.png        smoothed distillation loss over training steps
    ap_vs_sparsity.png    headline pruning plot (Easy/Medium/Hard/Mean AP)
    finetune_recovery.png AP recovery during finetune, per iteration
    summary.png           all three in one 2x2 figure

Usage:
    python plot_pruning_run.py                          # defaults to the newest run
    python plot_pruning_run.py --run logs/20260625_143543

Needs matplotlib + pandas. fdvenv typically has both; if not:
    fdvenv/bin/pip install matplotlib pandas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --- loading ---------------------------------------------------------------

def load_run(run_dir: Path):
    loss = pd.read_csv(run_dir / "loss_log.csv")
    metrics = pd.read_csv(run_dir / "metrics_log.csv")
    # A monotonic step index across the whole run is more useful than the
    # per-epoch step that resets — use it as the x-axis for loss.
    loss["global_step"] = range(len(loss))
    return loss, metrics


# --- individual plots ------------------------------------------------------

def plot_loss(loss: pd.DataFrame, ax, smooth_window: int = 100) -> None:
    """One line per iteration; rolling-mean smoothing makes the trend visible."""
    for it, sub in loss.groupby("iteration"):
        smoothed = sub["loss_total"].rolling(smooth_window, min_periods=1).mean()
        ax.plot(sub["global_step"], smoothed, label=f"iter {it}", linewidth=1.4)
    ax.set_xlabel("training step (global)")
    ax.set_ylabel(f"distillation loss (rolling mean, w={smooth_window})")
    ax.set_title("KD loss over training")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_ap_vs_sparsity(metrics: pd.DataFrame, ax) -> None:
    """Scatter Easy/Medium/Hard/Mean against sparsity, with phase as marker shape."""
    phase_markers = {
        "initial": ("o", 110),
        "after_prune": ("x", 80),
        "after_finetune": ("D", 90),
    }
    series = [("ap_easy", "Easy"), ("ap_medium", "Medium"),
              ("ap_hard", "Hard"), ("ap_mean", "Mean")]
    colors = {"ap_easy": "C0", "ap_medium": "C1", "ap_hard": "C2", "ap_mean": "k"}

    for col, label in series:
        for phase, (marker, size) in phase_markers.items():
            sub = metrics[metrics["phase"] == phase]
            if sub.empty:
                continue
            # Unfilled markers (like 'x') don't accept edgecolors — only set on filled ones.
            extra = {"edgecolors": "white", "linewidths": 0.5} if marker in {"o", "D"} else {}
            ax.scatter(sub["sparsity"], sub[col],
                       marker=marker, s=size, color=colors[col],
                       label=f"{label} ({phase})" if phase == "after_finetune" else None,
                       **extra)
        # connect the after_finetune points for each series to show the recovery line
        ft = metrics[metrics["phase"] == "after_finetune"].sort_values("sparsity")
        ax.plot([0, *ft["sparsity"]],
                [metrics[metrics["phase"] == "initial"][col].iloc[0], *ft[col]],
                color=colors[col], alpha=0.5, linewidth=1)

    ax.set_xlabel("sparsity (param reduction vs baseline)")
    ax.set_ylabel("Average Precision")
    ax.set_title("AP vs sparsity (○ initial, × after-prune, ◇ after-finetune)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(left=-0.02)
    ax.set_ylim(0, 1)


def plot_finetune_recovery(metrics: pd.DataFrame, ax) -> None:
    """Per-iteration AP recovery vs finetune epoch — shows whether 14 epochs is enough."""
    mid = metrics[metrics["phase"].str.startswith("finetune_ep")].copy()
    mid["epoch"] = mid["phase"].str.extract(r"finetune_ep(\d+)").astype(int)

    for it, sub in mid.groupby("iteration"):
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub["ap_mean"], marker="o",
                label=f"iter {it} (sparsity {sub['sparsity'].iloc[0]:.1%})")
        # also overlay the after_prune mean AP at epoch 0 as the starting point
        ap0 = metrics[(metrics["iteration"] == it) & (metrics["phase"] == "after_prune")]
        if not ap0.empty:
            ax.scatter(0, ap0["ap_mean"].iloc[0], marker="x", s=80,
                       color=ax.lines[-1].get_color())

    ax.set_xlabel("finetune epoch within iteration")
    ax.set_ylabel("mean AP (Easy+Medium+Hard)/3")
    ax.set_title("AP recovery during finetune (× = after-prune starting point)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None,
                    help="Path to a logs/<TIMESTAMP>/ dir. Default: newest under ./logs.")
    ap.add_argument("--smooth", type=int, default=100,
                    help="Rolling-window size for loss smoothing (default 100).")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    if args.run is None:
        runs_root = here / "logs"
        runs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
        if not runs:
            raise SystemExit(f"no runs under {runs_root}")
        args.run = runs[-1]
    if not (args.run / "loss_log.csv").exists():
        raise SystemExit(f"{args.run} doesn't look like a pruning run dir")

    print(f"plotting from: {args.run}")
    loss, metrics = load_run(args.run)
    out = args.run / "plots"
    out.mkdir(exist_ok=True)

    # Individual figures
    for name, fn, kwargs in [
        ("loss_curve",        plot_loss,             {"smooth_window": args.smooth}),
        ("ap_vs_sparsity",    plot_ap_vs_sparsity,   {}),
        ("finetune_recovery", plot_finetune_recovery,{}),
    ]:
        fig, axis = plt.subplots(figsize=(9, 5.5))
        if name == "loss_curve":
            fn(loss, axis, **kwargs)
        else:
            fn(metrics, axis, **kwargs)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=140)
        plt.close(fig)
        print(f"  wrote {out / (name + '.png')}")

    # Combined summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_loss(loss, axes[0, 0], smooth_window=args.smooth)
    plot_ap_vs_sparsity(metrics, axes[0, 1])
    plot_finetune_recovery(metrics, axes[1, 0])
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95,
                    f"Pruning run summary\n"
                    f"Run dir: {args.run.name}\n"
                    f"Loss steps: {len(loss):,}\n"
                    f"Iterations: {loss['iteration'].nunique()}\n"
                    f"Final sparsity: {metrics['sparsity'].max():.2%}\n"
                    f"Initial mean AP: {metrics[metrics['phase']=='initial']['ap_mean'].iloc[0]:.3f}\n"
                    f"Final mean AP:   {metrics[metrics['phase']=='after_finetune']['ap_mean'].iloc[-1]:.3f}\n"
                    f"AP retained:     {metrics[metrics['phase']=='after_finetune']['ap_mean'].iloc[-1] / metrics[metrics['phase']=='initial']['ap_mean'].iloc[0]:.1%}",
                    fontsize=11, family="monospace", va="top", transform=axes[1, 1].transAxes)
    fig.suptitle(f"Pruning run: {args.run.name}", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'summary.png'}")


if __name__ == "__main__":
    main()
