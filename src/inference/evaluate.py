"""
Full-volume evaluation of a trained checkpoint on the held-out validation split.

The per-epoch metrics logged during training are computed on a single
centre-cropped 128^3 patch per volume.  This script instead runs MONAI
sliding-window inference over each *entire* validation volume and scores the
full-volume prediction with Dice and the corrected surface-based HD95
(``src.utils.metrics.hausdorff_95``).

The validation cases are the ones ``build_dataloaders`` held out (fixed
seed 42, see ``split_case_names``), so the numbers are comparable to the
training-time Dice.  Model selection (``best.pth`` = highest validation Dice)
used this same split, so it is still an internal validation result, not an
independent test set.

Units: the ``.npy`` volumes carry no voxel-spacing metadata, so HD95 is
reported in voxel units unless ``--spacing`` is given.  Pass the spacing of
the source NIfTI files (MSD Task03 spacing varies per case) only if it is
known for the volumes actually loaded.

Usage:
    python -m src.inference.evaluate \\
        --checkpoint results/checkpoints/best.pth \\
        --data-dir /path/to/task03-liver-npy-dataset/ \\
        --out results/full_volume_metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.dataset import split_case_names
from src.inference.predict import predict_from_file
from src.models.unet3d import build_model
from src.utils.device import get_device
from src.utils.metrics import dice_score, hausdorff_95


def evaluate_case(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: int = 95,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Dice and HD95 for one full-volume binary prediction / ground-truth pair."""
    dice = dice_score(
        torch.from_numpy(np.ascontiguousarray(pred)),
        torch.from_numpy(np.ascontiguousarray(target)),
    )
    hd95 = hausdorff_95(pred, target, percentile=percentile, voxel_spacing=voxel_spacing)
    return {"dice": dice, "hd95": hd95}


def evaluate_split(
    model: nn.Module,
    config: Dict,
    data_dir: str,
    device: torch.device,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    case_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Run full-volume inference and scoring over the validation cases."""
    cfg_ds = config["dataset"]
    percentile = config["validation"]["hausdorff_percentile"]
    if case_names is None:
        _, case_names = split_case_names(config, data_dir)

    rows: List[Dict[str, object]] = []
    for i, name in enumerate(case_names, start=1):
        img_path = Path(data_dir) / cfg_ds["images_subdir"] / f"{name}{cfg_ds.get('images_suffix', '')}.npy"
        lbl_path = Path(data_dir) / cfg_ds["labels_subdir"] / f"{name}{cfg_ds.get('labels_suffix', '')}.npy"
        if not lbl_path.exists():
            raise FileNotFoundError(f"Ground-truth mask missing for {name}: {lbl_path}")

        _, pred = predict_from_file(model, str(img_path), config, device)
        # Same binarisation as LiverCTDataset: every positive label is foreground.
        gt = (np.load(str(lbl_path)) > 0).astype(np.uint8)

        scores = evaluate_case(pred, gt, percentile, voxel_spacing)
        rows.append({"case": name, "shape": "x".join(map(str, gt.shape)), **scores})
        print(f"  [{i:02d}/{len(case_names)}] {name}: Dice {scores['dice']:.4f}  HD95 {scores['hd95']:.2f}")
    return rows


def summarise(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-case rows; infinite HD95 (empty mask) is counted, not averaged."""
    dice = np.array([r["dice"] for r in rows], dtype=float)
    hd = np.array([r["hd95"] for r in rows], dtype=float)
    finite = hd[np.isfinite(hd)]
    return {
        "n_cases": len(rows),
        "dice_mean": float(dice.mean()),
        "dice_std": float(dice.std(ddof=1)) if len(dice) > 1 else 0.0,
        "dice_median": float(np.median(dice)),
        "dice_min": float(dice.min()),
        "hd95_mean": float(finite.mean()) if finite.size else float("inf"),
        "hd95_std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "hd95_median": float(np.median(finite)) if finite.size else float("inf"),
        "hd95_max": float(finite.max()) if finite.size else float("inf"),
        "hd95_n_infinite": int((~np.isfinite(hd)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth (must contain 'model_state')")
    parser.add_argument("--data-dir", required=True, help="Dataset root holding the image/ and liverMask/ folders")
    parser.add_argument("--config", default=None, help="Override the config stored in the checkpoint")
    parser.add_argument("--out", default="full_volume_metrics.csv", help="Per-case CSV; a .summary.json is written beside it")
    parser.add_argument(
        "--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), metavar=("D", "H", "W"),
        help="Voxel spacing in mm. Default 1 1 1 means HD95 is in VOXELS, not mm.",
    )
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    elif "config" in ckpt:
        config = ckpt["config"]
    else:
        parser.error("checkpoint has no embedded config; pass --config")

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    spacing = tuple(args.spacing)
    unit = "mm" if spacing != (1.0, 1.0, 1.0) else "voxels"
    print(f"Checkpoint epoch {ckpt.get('epoch')}, logged patch-level val Dice {ckpt.get('val_dice')}")
    print(f"Full-volume evaluation on the held-out split; HD95 unit: {unit}")

    rows = evaluate_split(model, config, args.data_dir, device, spacing)
    summary = summarise(rows)
    summary.update({
        "hd95_unit": unit,
        "voxel_spacing": list(spacing),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_logged_patch_val_dice": ckpt.get("val_dice"),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "shape", "dice", "hd95"])
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDice  {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f} (median {summary['dice_median']:.4f}, min {summary['dice_min']:.4f})")
    print(f"HD95  {summary['hd95_mean']:.2f} +/- {summary['hd95_std']:.2f} {unit} (median {summary['hd95_median']:.2f}, max {summary['hd95_max']:.2f}, {summary['hd95_n_infinite']} empty-mask case(s) excluded)")
    print(f"Wrote {out} and {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
