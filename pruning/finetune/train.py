"""Finetune a (pruned) YuNet on WIDER train.

Loads either an unpruned `yunet_n.pth` (via the standalone YuNet class) or a
pruned model file (full nn.Module pickle from prune_yunet.py). Runs AdamW
finetuning against the WIDER train annotations transformed into 320x320
coordinate space.

Designed as a smoke-testable scaffold rather than a polished production
trainer. Run with --max-steps to limit iteration count for sanity-checking
that the loss goes down.

Example (smoke test):
    python -m finetune.train \\
        --pruned ../pruned_yunet_30pct.pt \\
        --wider-root $WIDER_ROOT/WIDER_train \\
        --annotations $WIDER_ROOT/wider_face_split/wider_face_train_bbx_gt_320x320.txt \\
        --output ../pruned_yunet_30pct_ft.pt \\
        --max-steps 20 --batch-size 8

Example (real finetune):
    python -m finetune.train \\
        --pruned ../pruned_yunet_30pct.pt \\
        --wider-root ... --annotations ... \\
        --output ../pruned_yunet_30pct_ft.pt \\
        --epochs 5 --batch-size 16 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make 'yunet_standalone' importable when running this file as a module.
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune.dataset import WiderFaceTrain, collate_fn  # noqa: E402
from finetune.loss import YuNetLoss, LossWeights         # noqa: E402


def load_model(pruned_path: Path | None, weights_path: Path | None) -> torch.nn.Module:
    if pruned_path is not None:
        return torch.load(str(pruned_path), map_location="cpu", weights_only=False)
    from yunet_standalone import YuNet
    model = YuNet()
    model.load_pretrained(str(weights_path), strict=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pruned", type=Path, default=None,
                        help="Pruned model .pt to finetune (full nn.Module pickle)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Pretrained .pth (use instead of --pruned to finetune the unpruned baseline)")
    parser.add_argument("--wider-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="320x320-space WIDER train annotations (.txt)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to save the finetuned model")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many iterations regardless of epoch count (smoke test)")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.pruned is None and args.weights is None:
        sys.exit("Pass either --pruned (full nn.Module .pt) or --weights (raw .pth state dict)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model ...")
    model = load_model(args.pruned, args.weights).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    print("Building dataset ...")
    dataset = WiderFaceTrain(
        wider_root=args.wider_root,
        annotations=args.annotations,
        input_size=args.input_size,
        horizontal_flip=True,
        skip_empty=True,
    )
    print(f"  {len(dataset)} images")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    criterion = YuNetLoss(input_size=args.input_size, weights=LossWeights())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    total_steps = args.epochs * len(loader)
    t_start = time.time()

    for epoch in range(args.epochs):
        for batch in loader:
            images = batch["images"].to(device)
            gt_boxes = [b.to(device) for b in batch["boxes"]]

            outputs = model(images)
            losses = criterion(outputs, gt_boxes)
            loss = losses["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            step += 1
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t_start
                rate = step / max(elapsed, 1e-6)
                print(f"  ep{epoch} step{step:4d}/{total_steps}  "
                      f"loss={loss.item():.4f}  "
                      f"cls={losses['cls'].item():.4f}  "
                      f"obj={losses['obj'].item():.4f}  "
                      f"bbox={losses['bbox'].item():.4f}  "
                      f"({rate:.1f} it/s)")

            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

        # End-of-epoch save (skipped in smoke-test mode)
        if args.max_steps is None:
            epoch_path = args.output.with_name(f"{args.output.stem}_ep{epoch}{args.output.suffix}")
            torch.save(model, str(epoch_path))
            print(f"  saved checkpoint: {epoch_path.name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, str(args.output))
    print(f"\nSaved final model: {args.output}")


if __name__ == "__main__":
    main()
