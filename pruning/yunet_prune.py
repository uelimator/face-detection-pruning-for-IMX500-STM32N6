import csv
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch_pruning as tp
from finetune.dataset import WiderFaceTrain, collate_fn

# from finetune.loss import YuNetLoss   # replaced by DistillationLoss below
from finetune.loss import DistillationLoss
from torch import nn
from torch.utils.data import DataLoader
from yunet_standalone import YuNet

if __name__ == "__main__":
    # WIDER FACE dataset root — override with the WIDER_ROOT env var.
    # Expected layout:
    #   $WIDER_ROOT/WIDER_train/            (original training images)
    #   $WIDER_ROOT/WIDER_val/               (original val images)
    #   $WIDER_ROOT/ground_truth_320/        (320x320-transformed GT .mat files)
    #   $WIDER_ROOT/wider_face_split/wider_face_train_bbx_gt_320x320.txt
    WIDER_ROOT = Path(os.environ.get("WIDER_ROOT",
                                     Path.home() / "Datasets" / "Wider_Faces"))
    if not WIDER_ROOT.exists():
        raise SystemExit(
            f"WIDER FACE dataset not found at {WIDER_ROOT}. "
            "Set the WIDER_ROOT env var or place the dataset there.")

    GT_DIR   = WIDER_ROOT / "ground_truth_320"
    VAL_ROOT = WIDER_ROOT / "WIDER_val"

    # train_loader, just defines how and where to load the data for training
    train_dataset = WiderFaceTrain(
        wider_root=str(WIDER_ROOT / "WIDER_train"),
        annotations=str(WIDER_ROOT / "wider_face_split" / "wider_face_train_bbx_gt_320x320.txt"),
        input_size=320,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,  # critical: each image has variable number of GT boxes
        drop_last=True,
    )

    # I have no idea what this does :(
    """
    batch = next(iter(train_loader))
    batch["images"]  # (B, 3, 320, 320) tensor
    batch["boxes"]   # list of B tensors, each (N_i, 4)
    """

    # Set Device to MPS (apple)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # 1. Instantiate the class (needs to match the architecture the checkpoint was saved with)
    # model = YuNet().to(device=device)
    model = YuNet()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 2. Load — weights_only=True is the PyTorch 2.x security flag
    checkpoint = torch.load("yunet_n_baseline.ckpt", weights_only=True)

    # 3. Apply state dicts
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    # module = model.backbone.model0.conv1
    # print(list(module.named_parameters()))

    # ---- Frozen teacher (unpruned) for distillation ----
    # Same architecture, same weights loaded from the same checkpoint, but
    # never gets pruned or updated. Used by fine_tune as the target.
    teacher = YuNet()
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    """
    model.eval()   # or model.train() for finetuning

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_cfg": "yunet_n",
    }, f"yunet_n_epoch{epoch}.ckpt")
    """

    class IterativePruner:
        def __init__(self, model, train_loader, optimizer, device, teacher=None):
            self.model = model
            self.train_loader = train_loader
            # self.val_loader=val_loader
            self.optimizer = optimizer
            self.device = device
            self.teacher = teacher  # frozen reference for distillation
            self.model.to(device)
            self.baseline_params = sum(p.numel() for p in self.model.parameters())

            # ---- Logging: per-step loss + per-eval metrics, CSV, line-buffered ----
            log_root = Path(__file__).parent / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            log_root.mkdir(parents=True, exist_ok=True)
            self._log_root = log_root
            self._loss_fh = open(log_root / "loss_log.csv", "w", newline="", buffering=1)
            self._loss_csv = csv.writer(self._loss_fh)
            self._loss_csv.writerow(["timestamp", "iteration", "epoch", "step", "loss_total"])
            self._metrics_fh = open(log_root / "metrics_log.csv", "w", newline="", buffering=1)
            self._metrics_csv = csv.writer(self._metrics_fh)
            self._metrics_csv.writerow(["timestamp", "iteration", "phase", "sparsity",
                                        "ap_easy", "ap_medium", "ap_hard", "ap_mean"])
            self._current_iteration = -1   # set by iterative_prune
            print(f"[log] writing to {log_root}")

        def _log_loss(self, epoch, step, loss_value):
            self._loss_csv.writerow([
                datetime.now().isoformat(timespec="seconds"),
                self._current_iteration, epoch, step, float(loss_value),
            ])

        def _log_metrics(self, phase, sparsity, aps):
            mean_ap = (aps["easy"] + aps["medium"] + aps["hard"]) / 3.0
            self._metrics_csv.writerow([
                datetime.now().isoformat(timespec="seconds"),
                self._current_iteration, phase, sparsity,
                aps["easy"], aps["medium"], aps["hard"], mean_ap,
            ])

        def evaluate(self):
            self.model.eval()
            from validate_pytorch import validate_model

            return validate_model(
                self.model,
                wider_root=VAL_ROOT,
                output_dir=Path("/tmp/iter_val"),
                gt_dir=GT_DIR,
                verbose=False,
                device=self.device,
            )
            """aps = validate_model(
                self.model, wider_root=VAL_ROOT, output_dir=Path("/tmp/iter_val"),
                gt_dir=GT_DIR, verbose=False, device=self.device,
            )
            return (aps["easy"] + aps["medium"] + aps["hard"]) / 3   # mean AP as the single "accuracy" number"""

        def fine_tune(self, epochs):
            """Fine tune model after pruning, via knowledge distillation from
            the frozen unpruned teacher."""
            self.model.train()
            # Keep BN running stats frozen — they were calibrated on the full
            # training set and shouldn't drift from small-batch finetune stats.
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            criterion = DistillationLoss()

            for epoch in range(epochs):
                steps = 0
                for batch in self.train_loader:
                    images = batch["images"].to(self.device)
                    # gt_boxes still loaded by the dataset but unused for distillation:
                    # _ = [b.to(self.device) for b in batch["boxes"]]

                    self.optimizer.zero_grad()

                    # Teacher forward — no gradient tracking, eval mode.
                    with torch.no_grad():
                        teacher_outputs = self.teacher(images)

                    # Student forward — gradient tracked.
                    student_outputs = self.model(images)

                    losses = criterion(student_outputs, teacher_outputs)
                    loss = losses["total"]
                    loss.backward()
                    self.optimizer.step()
                    steps += 1
                    self._log_loss(epoch, steps, loss.item())
                    if (steps + 1) % 50 == 0:
                        print(f"Loss = {loss:.6f}")

                if (epoch + 1) % 2 == 0:
                    aps = self.evaluate()
                    print(
                        f"Easy: {aps['easy']:.4f}  Medium: {aps['medium']:.4f}  Hard: {aps['hard']:.4f}"
                    )
                    self._log_metrics(f"finetune_ep{epoch+1}",
                                      self._calculate_sparsity_structured(), aps)

        # ====================================================================
        # OLD CODE — supervised finetune with YuNetLoss. Replaced by the
        # distillation version above. Kept here for reference.
        # ====================================================================
        """
        def fine_tune(self, epochs):
            self.model.train()
            criterion = YuNetLoss(input_size=320)
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for epoch in range(epochs):
                steps = 0
                for batch in self.train_loader:
                    images = batch["images"].to(self.device)
                    gt_boxes = [b.to(self.device) for b in batch["boxes"]]
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    losses = criterion(outputs, gt_boxes)
                    loss = losses["total"]
                    loss.backward()
                    self.optimizer.step()
                    steps += 1
                    if (steps + 1) % 50 == 0:
                        print(f"Loss = {loss:.4f}")
                if (epoch + 1) % 2 == 0:
                    aps = self.evaluate()
                    print(f"Easy: {aps['easy']:.4f}  Medium: {aps['medium']:.4f}  Hard: {aps['hard']:.4f}")
        """

        def iterative_prune(self, target_sparsity, num_iterations, fine_tune_epochs):
            """Perform iterative pruning with fine-tuning."""
            # Calculate per-iteration pruning rate
            per_iter_rate = 1 - (1 - target_sparsity) ** (1 / num_iterations)

            # print(f"Initial accuracy: {self.evaluate()}")
            init_acc = self.evaluate()
            print(
                f"initial performance: {init_acc['easy']:.4f}  Medium: {init_acc['medium']:.4f}  Hard: {init_acc['hard']:.4f}"
            )
            self._log_metrics("initial", 0.0, init_acc)

            for iteration in range(num_iterations):
                self._current_iteration = iteration
                print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

                # Apply pruning
                # self._apply_magnitude_pruning(per_iter_rate)
                self._applyStructuredPruning(per_iter_rate)

                # Calculate current sparsity
                # sparsity = self._calculate_sparsity()
                sparsity = self._calculate_sparsity_structured()
                print(f"Current sparsity: {sparsity:.2%}")

                # Evaluate after pruning
                acc_after_prune = self.evaluate()
                print(
                    f"Performance after prune, Easy: {acc_after_prune['easy']:.4f}  Medium: {acc_after_prune['medium']:.4f}  Hard: {acc_after_prune['hard']:.4f}"
                )
                self._log_metrics("after_prune", sparsity, acc_after_prune)

                # Fine-tune
                print("Fine-tuning...")
                self.fine_tune(fine_tune_epochs)

                # Evaluate after fine-tuning
                acc_after_ft = self.evaluate()
                print(
                    f"Performance after finetuning, Easy: {acc_after_ft['easy']:.4f}  Medium: {acc_after_ft['medium']:.4f}  Hard: {acc_after_ft['hard']:.4f}"
                )
                self._log_metrics("after_finetune", sparsity, acc_after_ft)

            # Make pruning permanent
            # self._remove_pruning_masks()--> Not needed for structured pruning

            return self.model

        # This is very useful for unstructured pruning but works less well for structured pruning
        def _apply_magnitude_pruning(self, amount):
            """Apply magnitude-based pruning."""
            import torch.nn.utils.prune as prune

            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name="weight", amount=amount)
                    # prune.ln_structured(module,name='weight',n=2, amount=amount,dim=1)# This is too crude, no real benefit
                    # Re-create optimizer with the new parameter set
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-6, weight_decay=1e-4
            )

        def _applyStructuredPruning(self, amount):
            "Apply structured pruning to select stages"
            example = torch.zeros(1, 3, 320, 320).to(device)
            # Collect target convs from your chosen stages
            target_convs = []
            for stage in [
                model.backbone.model0,
                model.backbone.model1,
                model.backbone.model2,
                model.backbone.model3,
                model.backbone.model4,
                model.backbone.model5,
                model.neck,
            ]:
                for m in stage.modules():
                    if isinstance(m, nn.Conv2d):
                        target_convs.append(m)

            # Everything ELSE prunable is ignored
            ignored = [
                m
                for _, m in model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear)) and m not in target_convs
            ]

            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs=example,
                # importance=tp.importance.BNScaleImportance(),
                importance=tp.importance.MagnitudeImportance(p=2),
                pruning_ratio=0.06,
                round_to=8,
                ignored_layers=ignored,
            )
            pruner.step()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=5e-4, weight_decay=1e-4
            )

        def _calculate_sparsity(self):
            """Calculate overall model sparsity."""
            total_params = 0
            zero_params = 0

            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    total_params += module.weight.nelement()
                    zero_params += (module.weight == 0).sum().item()

            return zero_params / total_params

        def _calculate_sparsity_structured(self):
            """Override: report channel/param reduction vs baseline, not weight-zero ratio."""
            current = sum(p.numel() for p in self.model.parameters())
            return 1.0 - current / self.baseline_params

        def _remove_pruning_masks(self):
            """Remove pruning masks to make pruning permanent."""
            import torch.nn.utils.prune as prune

            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    try:
                        prune.remove(module, "weight")
                    except ValueError:
                        pass

    class StructuredPruner:
        def __init__(self, model):
            self.model = model

        def compute_filter_importance(self, conv_layer):
            """Compute importance scores for each filter using L1 norm."""
            weights = conv_layer.weight.data
            # Sum absolute values across spatial dimensions and input channels
            importance = torch.sum(torch.abs(weights), dim=(1, 2, 3))
            return importance

        def prune_conv_layer(self, conv_layer, bn_layer, prune_ratio):
            """Prune filters from a convolutional layer."""
            importance = self.compute_filter_importance(conv_layer)
            num_filters = conv_layer.out_channels
            num_to_prune = int(num_filters * prune_ratio)
            num_to_keep = num_filters - num_to_prune

            # Get indices of filters to keep (highest importance)
            _, keep_indices = torch.topk(importance, num_to_keep)
            keep_indices = keep_indices.sort()[0]

            # Create new smaller conv layer
            new_conv = nn.Conv2d(
                in_channels=conv_layer.in_channels,
                out_channels=num_to_keep,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                bias=conv_layer.bias is not None,
            )

            # Copy selected filters
            new_conv.weight.data = conv_layer.weight.data[keep_indices]
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data[keep_indices]

            # Create new batch norm if present
            new_bn = None
            if bn_layer is not None:
                new_bn = nn.BatchNorm2d(num_to_keep)
                new_bn.weight.data = bn_layer.weight.data[keep_indices]
                new_bn.bias.data = bn_layer.bias.data[keep_indices]
                new_bn.running_mean = bn_layer.running_mean[keep_indices]
                new_bn.running_var = bn_layer.running_var[keep_indices]

            return new_conv, new_bn, keep_indices

        def adjust_next_layer(self, next_conv, keep_indices):
            """Adjust the input channels of the next layer."""
            new_conv = nn.Conv2d(
                in_channels=len(keep_indices),
                out_channels=next_conv.out_channels,
                kernel_size=next_conv.kernel_size,
                stride=next_conv.stride,
                padding=next_conv.padding,
                bias=next_conv.bias is not None,
            )

            new_conv.weight.data = next_conv.weight.data[:, keep_indices]
            if next_conv.bias is not None:
                new_conv.bias.data = next_conv.bias.data

            return new_conv

    # pruner=StructuredPruner(model)
    # model.backbone.model0.
    # Hierarchy view
    """    print(model.backbone.model1)


  # Just the conv layers with their pixel shapes — usually more useful
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"{name:60}  {tuple(m.weight.shape)}")"""

    pruner = IterativePruner(model, train_loader, optimizer, device, teacher=teacher)

    pruned_model = pruner.iterative_prune(
        target_sparsity=0.2, num_iterations=2, fine_tune_epochs=14
    )

    pruned_model.to("cpu")  # cleaner for portability; ONNX export prefers CPU tensors
    pruned_model.eval()
    torch.save(pruned_model, "pruned_yunet_structured_20_06.pt")

    pruner._loss_fh.close()
    pruner._metrics_fh.close()
