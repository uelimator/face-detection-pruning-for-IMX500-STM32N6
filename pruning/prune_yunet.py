"""Apply structured channel pruning to the standalone YuNet.

Uses Torch-Pruning's MagnitudePruner with the ignored_layers list set so
that the per-stride output convs keep their fixed output channels
(1 for cls, 1 for obj, 4 for bbox, 10 for kps). Pruning their input
channels is allowed and desirable — that's how the head shrinks.

After pruning the model is saved as a full nn.Module pickle (torch.save(model))
since the pruned architecture diverges from the YUNET_N_CFG defaults and
can't be reinstantiated by re-running YuNet() + load_state_dict(). Loading
later requires this file's directory in sys.path so the yunet_standalone
module is importable.

What's intentionally NOT here: finetuning. This script prunes once and stops.
Use prune_then_finetune.py (TBD) for the iterative recipe.

Example:
    python prune_yunet.py \\
        --weights ../training/libfacedetection.train/weights/yunet_n.pth \\
        --output pruned_yunet_30pct.pt \\
        --pruning-ratio 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_pruning as tp

from yunet_standalone import YuNet


def collect_head_output_convs(model: YuNet) -> list[torch.nn.Module]:
    """Return the per-stride output convs whose output channels must NOT be pruned.

    These define the model's interface contract: 1 channel for cls, 1 for obj,
    4 for bbox, 10 for kps (5 landmarks × 2). Pruning their output dims would
    change the API and break downstream code that expects fixed channel counts.

    Torch-Pruning interprets `ignored_layers` as 'do not prune output channels of
    these layers' — input pruning of the same layer is still allowed and is how
    we shrink the head from upstream.
    """
    head = model.bbox_head
    ignored: list[torch.nn.Module] = []
    for module_list in (head.multi_level_cls, head.multi_level_obj,
                        head.multi_level_bbox, head.multi_level_kps):
        for unit in module_list:
            # Each multi_level_* entry is a ConvDPUnit. The OUTPUT convs are
            # conv2 (3x3 depthwise) — depthwise weights have shape (C, 1, 3, 3)
            # so groups=C. We ignore both conv1 (pointwise) and conv2 of these
            # final output ConvDPUnits because they together define the head's
            # output channel count.
            ignored.append(unit.conv1)
            ignored.append(unit.conv2)
    return ignored


def summarize(model: torch.nn.Module, example: torch.Tensor, label: str) -> dict[str, float]:
    """Compute and print parameter count + MACs for a model."""
    n_params = sum(p.numel() for p in model.parameters())
    # tp.utils.count_ops_and_params returns (MACs, params); MACs ~ FLOPs / 2.
    macs, _ = tp.utils.count_ops_and_params(model, example)
    print(f"{label:<10}  params: {n_params:>10,}   MACs: {macs:>15,.0f}")
    return {"params": float(n_params), "macs": float(macs)}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().double()
    bf = b.flatten().double()
    denom = float(torch.linalg.norm(af) * torch.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if torch.equal(af, bf) else 0.0
    return float(torch.dot(af, bf) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Pretrained .pth from libfacedetection.train")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to save the pruned model (torch.save(module))")
    parser.add_argument("--pruning-ratio", type=float, default=0.3,
                        help="Fraction of channels to remove from each prunable layer (default 0.3)")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--importance", choices=["l1", "l2", "random", "bn"], default="l2",
                        help="Channel importance criterion (default L2 magnitude). "
                             "'bn' uses BatchNorm gamma scales — often better for "
                             "depthwise-separable architectures like YuNet.")
    parser.add_argument("--global-pruning", action="store_true",
                        help="Apply pruning ratio globally across all prunable layers, "
                             "rather than per-layer. Often produces better accuracy/size trade-offs.")
    parser.add_argument("--validate", action="store_true",
                        help="After pruning, run WIDER val and print Easy/Medium/Hard AP. "
                             "Requires --wider-root and --gt-dir.")
    parser.add_argument("--wider-root", type=Path, default=None,
                        help="WIDER_val/ root, used only when --validate is set")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="320x320-space WIDER GT directory, used only when --validate is set")
    parser.add_argument("--validate-output-dir", type=Path, default=None,
                        help="Where to put prediction files for the validation run. "
                             "Defaults to <output dir>/validation_preds.")
    args = parser.parse_args()

    if args.validate and (args.wider_root is None or args.gt_dir is None):
        parser.error("--validate requires both --wider-root and --gt-dir")

    print("=== Loading standalone YuNet ===")
    model = YuNet()
    missing, unexpected = model.load_pretrained(str(args.weights), strict=False)
    if missing or unexpected:
        print(f"  WARN: state_dict had missing={len(missing)} unexpected={len(unexpected)} keys")
    model.eval()

    example = torch.zeros(1, 3, args.input_size, args.input_size)

    # Snapshot the unpruned outputs for a post-prune comparison
    with torch.no_grad():
        outputs_before = [t.clone() for t in model(example)]

    print("\n=== Before pruning ===")
    stats_before = summarize(model, example, "FP32 base")

    print("\n=== Configuring pruner ===")
    if args.importance == "l1":
        importance = tp.importance.MagnitudeImportance(p=1)
    elif args.importance == "l2":
        importance = tp.importance.MagnitudeImportance(p=2)
    elif args.importance == "bn":
        importance = tp.importance.BNScaleImportance()
    else:
        importance = tp.importance.RandomImportance()

    ignored = collect_head_output_convs(model)
    print(f"  protected (output convs): {len(ignored)} modules")
    print(f"  importance: {args.importance}")
    print(f"  pruning ratio: {args.pruning_ratio}")
    print(f"  global pruning: {args.global_pruning}")

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example,
        importance=importance,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored,
        global_pruning=args.global_pruning,
    )

    print("\n=== Running pruner.step() ===")
    pruner.step()

    print("\n=== After pruning ===")
    stats_after = summarize(model, example, "Pruned")

    print("\n=== Reduction ===")
    print(f"  params: {stats_before['params']:.0f} -> {stats_after['params']:.0f} "
          f"({100.0 * (1 - stats_after['params'] / stats_before['params']):.1f}% reduction)")
    print(f"  MACs:   {stats_before['macs']:.0f} -> {stats_after['macs']:.0f} "
          f"({100.0 * (1 - stats_after['macs'] / stats_before['macs']):.1f}% reduction)")

    print("\n=== Output preservation check ===")
    print("(Comparing pruned-no-finetune vs original. Expect noticeable drift — that's the")
    print(" accuracy hit you'll need to recover by finetuning.)")
    model.eval()
    with torch.no_grad():
        outputs_after = list(model(example))

    expected_names = [
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32",
    ]
    for name, b, a in zip(expected_names, outputs_before, outputs_after):
        if b.shape != a.shape:
            print(f"  {name}: SHAPE CHANGED b={tuple(b.shape)} a={tuple(a.shape)}")
            continue
        cos = cosine(b, a)
        print(f"  {name}: shape {tuple(a.shape)}  cos={cos:.4f}")

    print(f"\n=== Saving pruned model to {args.output} ===")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save the full nn.Module (pickle). Loading requires yunet_standalone on PYTHONPATH.
    torch.save(model, str(args.output))
    size_kb = args.output.stat().st_size / 1024.0
    print(f"  Wrote {size_kb:.1f} KB")

    print("\n=== Reload sanity check ===")
    reloaded = torch.load(str(args.output), map_location="cpu", weights_only=False)
    reloaded.eval()
    with torch.no_grad():
        reloaded_outs = reloaded(example)
    all_match = all(torch.allclose(a, b) for a, b in zip(outputs_after, reloaded_outs))
    print(f"  Reloaded forward matches saved-model forward: {all_match}")

    if args.validate:
        from validate_pytorch import validate_model
        validate_output = args.validate_output_dir or (args.output.parent / "validation_preds")
        print(f"\n=== Validating pruned model on WIDER val ===")
        print(f"  Predictions -> {validate_output}")
        aps = validate_model(
            model, args.wider_root, validate_output, args.gt_dir,
            input_size=args.input_size,
        )
        print(f"\n  Easy:   {aps['easy']:.4f}")
        print(f"  Medium: {aps['medium']:.4f}")
        print(f"  Hard:   {aps['hard']:.4f}")

    print("\n=== Done ===")
    print("Next step: wire up a WIDER finetune loop and run prune_then_finetune.py")
    print("(not yet written). The current pruned model has structural compression")
    print("but no accuracy recovery — running WIDER eval on it now will show the drop.")


if __name__ == "__main__":
    main()
