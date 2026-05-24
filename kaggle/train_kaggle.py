"""
Kaggle entry point for 3D liver segmentation training.

Usage (inside a Kaggle notebook cell):
    !git clone https://github.com/AdebanjiAdelowo/abdominal-ct-segmentation \
        /kaggle/working/abdominal-ct-segmentation
    %cd /kaggle/working/abdominal-ct-segmentation
    !pip install -r requirements.txt -q
    !python kaggle/train_kaggle.py

Expected Kaggle dataset path: /kaggle/input/datasets/zeynepzelk/task03-liver-npy-dataset/
All outputs (checkpoints, metrics CSV, visualisations) go to /kaggle/working/.

Dataset directory layout inferred from the Kaggle dataset page:
    /kaggle/input/task03-liver-npy-dataset/
    ├── imagesTr/    ← pre-processed CT volumes as float32 .npy
    └── labelsTr/    ← binary liver masks as int .npy
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so `src.*` imports resolve correctly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml

from src.data.dataset import build_dataloaders
from src.inference.predict import predict_from_file
from src.inference.visualise import save_segmentation_figure
from src.models.unet3d import build_model
from src.training.trainer import Trainer
from src.utils.device import get_device

# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

DATASET_DIR = "/kaggle/input/datasets/zeynepzelk/task03-liver-npy-dataset/"
OUTPUT_DIR = "/kaggle/working/"
CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Config ───────────────────────────────────────────────────────────────
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Override paths for the Kaggle runtime environment
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    metrics_csv = os.path.join(OUTPUT_DIR, config["logging"]["metrics_csv"])
    config["checkpoint"]["save_dir"] = checkpoint_dir
    config["logging"]["metrics_csv"] = metrics_csv

    # ── Device ───────────────────────────────────────────────────────────────
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[data] Building DataLoaders …")
    train_loader, val_loader = build_dataloaders(config, DATASET_DIR)
    print(
        f"  Train batches : {len(train_loader)}"
        f"  (batch_size={config['training']['batch_size']})"
    )
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[model] Building UNet3D …")
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Depth         : {config['model']['depth']}")
    print(f"  Base features : {config['model']['base_features']}")
    print(f"  Residual      : {config['model']['residual']}")
    print(f"  Trainable params: {n_params:,}")

    # ── Training ─────────────────────────────────────────────────────────────
    print("\n[train] Starting training …")
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=checkpoint_dir,
    )
    trainer.fit(train_loader, val_loader)

    print("\n[train] Training complete.")
    print(f"  Best val Dice : {trainer.best_dice:.4f}")
    print(f"  Checkpoints   : {checkpoint_dir}")
    print(f"  Metrics CSV   : {metrics_csv}")

    # ── Post-training visualisation ───────────────────────────────────────────
    # Load the best checkpoint and produce segmentation figures for the first
    # three validation volumes as a quick qualitative sanity check.
    print("\n[vis] Generating segmentation figures …")
    import torch
    from src.models.unet3d import build_model as _build

    best_ckpt_path = os.path.join(checkpoint_dir, "best.pth")
    if os.path.exists(best_ckpt_path):
        vis_model = _build(config).to(device)
        ckpt = torch.load(best_ckpt_path, map_location=device)
        vis_model.load_state_dict(ckpt["model_state"])
        vis_model.eval()

        vis_dir = os.path.join(OUTPUT_DIR, "visualisations")
        images_dir = (
            Path(DATASET_DIR) / config["dataset"]["images_subdir"]
        )
        sample_paths = sorted(images_dir.glob("*.npy"))[:3]

        for img_path in sample_paths:
            lbl_path = (
                Path(DATASET_DIR)
                / config["dataset"]["labels_subdir"]
                / img_path.name
            )
            volume, pred = predict_from_file(
                vis_model, str(img_path), config, device
            )
            gt = None
            if lbl_path.exists():
                import numpy as _np
                gt = (_np.load(str(lbl_path)) > 0).astype(_np.uint8)

            out_png = os.path.join(vis_dir, f"{img_path.stem}.png")
            save_segmentation_figure(
                volume, pred, out_png, gt_mask=gt, case_name=img_path.stem
            )
    else:
        print("  No best.pth found — skipping visualisation.")

    print("\nAll outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
