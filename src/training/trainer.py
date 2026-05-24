"""
Training pipeline for 3D liver segmentation.

Key design choices:
  Optimiser  : AdamW — weight decay decoupled from gradient updates.
  Loss       : Soft-Dice + BCE (combined; weights in config).
  Scheduler  : CosineAnnealingLR — smooth LR decay without discrete steps.
  AMP        : torch.amp mixed-precision on CUDA; skipped gracefully on MPS/CPU.
  Grad clip  : L2 norm clipped at config.training.grad_clip to stabilise 3-D
               convolution training on large patch batches.
  Checkpoints: 'best.pth' (highest val Dice) and 'last.pth' (most recent).
  Metrics log: CSV with epoch, train_loss, val_dice, val_hd95, lr.
"""

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils.metrics import dice_score, hausdorff_95


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss computed on sigmoid-activated logits.

    The 'soft' formulation uses continuous probabilities rather than a
    hard threshold, making the loss fully differentiable and better suited
    to class-imbalanced volumetric segmentation tasks.

    Args:
        smooth: Laplace smoothing constant to prevent 0/0 on empty masks.
    """

    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten to 1-D to compute global Dice across the entire batch volume
        p = probs.reshape(-1)
        t = targets.reshape(-1)
        numerator = 2.0 * (p * t).sum() + self.smooth
        denominator = p.sum() + t.sum() + self.smooth
        return 1.0 - numerator / denominator


class CombinedLoss(nn.Module):
    """
    Weighted sum of soft-Dice loss and binary cross-entropy from logits.

    Dice handles class imbalance; BCE ensures per-voxel calibration.
    Both weights are configurable via config.yaml.

    Args:
        dice_weight: Coefficient for the Dice term.
        bce_weight:  Coefficient for the BCE term.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()  # numerically stable (fuses sigmoid + BCE)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.bce_weight * self.bce(logits, targets)
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training and validation lifecycle.

    Args:
        model:      Initialised UNet3D (untrained).
        config:     Full config dict parsed from config.yaml.
        device:     Target device (cuda / mps / cpu).
        output_dir: Directory where checkpoints and metrics.csv are written.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
        output_dir: str,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg_tr = config["training"]
        cfg_sch = cfg_tr["scheduler"]
        cfg_val = config["validation"]

        self.criterion = CombinedLoss(
            dice_weight=cfg_tr["dice_weight"],
            bce_weight=cfg_tr["bce_weight"],
        )
        self.optimiser = AdamW(
            model.parameters(),
            lr=cfg_tr["lr"],
            weight_decay=cfg_tr["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=cfg_tr["epochs"],
            eta_min=cfg_sch["eta_min"],
        )

        # Mixed precision is only available (and useful) on CUDA
        self.use_amp: bool = device.type == "cuda"
        self.scaler: Optional[GradScaler] = GradScaler("cuda") if self.use_amp else None

        self.epochs: int = cfg_tr["epochs"]
        self.grad_clip: float = cfg_tr["grad_clip"]
        self.val_interval: int = cfg_val["val_interval"]
        self.hd_percentile: int = cfg_val["hausdorff_percentile"]

        self.best_dice: float = -1.0
        self.metrics_path = self.output_dir / config["logging"]["metrics_csv"]
        self._init_csv()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader) -> None:
        """Run the full training loop for config.training.epochs epochs."""
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics: Dict = {}

            if epoch % self.val_interval == 0:
                val_metrics = self._validate(val_loader, epoch)
                if self.config["checkpoint"]["save_best"]:
                    self._maybe_save_checkpoint(val_metrics["dice"], epoch)

            self.scheduler.step()
            self._log_metrics(epoch, train_loss, val_metrics)

    # ------------------------------------------------------------------
    # Private: training
    # ------------------------------------------------------------------

    def _train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        amp_ctx = autocast("cuda") if self.use_amp else nullcontext()

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimiser.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Unscale before grad-clipping so the clip threshold is in
                # real gradient units (not AMP-scaled units)
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimiser.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Private: validation
    # ------------------------------------------------------------------

    def _validate(self, loader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        dice_scores, hd95_scores = [], []
        amp_ctx = autocast("cuda") if self.use_amp else nullcontext()

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)
            for batch in pbar:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                with amp_ctx:
                    logits = self.model(images)

                # Hard threshold at 0.5 for binary Dice and HD95
                preds = (torch.sigmoid(logits) > 0.5).float()

                d = dice_score(preds, masks)
                h = hausdorff_95(
                    preds.cpu().numpy(),
                    masks.cpu().numpy(),
                    percentile=self.hd_percentile,
                )
                dice_scores.append(d)
                hd95_scores.append(h)
                pbar.set_postfix(dice=f"{d:.4f}", hd95=f"{h:.2f}")

        mean_dice = float(sum(dice_scores) / len(dice_scores))
        # Exclude inf values (empty prediction / ground-truth) from HD95 mean
        valid_hd = [h for h in hd95_scores if h != float("inf")]
        mean_hd95 = float(sum(valid_hd) / len(valid_hd)) if valid_hd else float("inf")

        print(
            f"  Epoch {epoch:03d} │ val Dice {mean_dice:.4f} │ "
            f"val HD95 {mean_hd95:.2f} mm"
        )
        return {"dice": mean_dice, "hd95": mean_hd95}

    # ------------------------------------------------------------------
    # Private: checkpointing and logging
    # ------------------------------------------------------------------

    def _maybe_save_checkpoint(self, val_dice: float, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_dice": val_dice,
            "config": self.config,
        }
        torch.save(ckpt, self.output_dir / "last.pth")

        if val_dice > self.best_dice:
            self.best_dice = val_dice
            torch.save(ckpt, self.output_dir / "best.pth")
            print(f"  ✓ New best Dice {val_dice:.4f} — saved to best.pth")

    def _init_csv(self) -> None:
        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_dice", "val_hd95", "lr"]
            )

    def _log_metrics(
        self, epoch: int, train_loss: float, val_metrics: Dict
    ) -> None:
        lr = self.optimiser.param_groups[0]["lr"]
        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_metrics['dice']:.4f}" if val_metrics else "",
                f"{val_metrics['hd95']:.2f}" if val_metrics else "",
                f"{lr:.2e}",
            ])
