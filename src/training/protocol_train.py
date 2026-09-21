"""
Training for the leakage-controlled protocol (NIfTI, three-way split).

This trains a NEW model.  It is not the historical 0.9886-Dice model and its
numbers must never be reported in place of it.

Isolation and reproducibility guarantees
    * Only the train and validation case lists are loaded (``load_splits`` with
      ``include_test=False``); the test case names never enter this process.
    * ``best.pth`` is chosen by patch-level validation Dice on the validation
      split.  Validation may also drive early stopping and model-development
      decisions; the test split may not (see docs/EVALUATION_PROTOCOL.md).
    * Training refuses to start unless the split file is committed and
      unmodified, and (unless ``--allow-dirty``) the working tree is clean, so
      the recorded git commit and split fingerprint describe what actually ran.
    * All RNGs are seeded.  GPU training is not guaranteed bit-reproducible.
    * Checkpoints are written atomically with seed / fingerprint / commit /
      timestamp / model config and can be mirrored to persistent storage.

Usage:
    python -m src.training.protocol_train --data-dir /path/Task03_Liver \
        --splits splits/msd_task03_v1.json --out-dir runs/protocol_v1 [--mirror-dir /content/drive/...]
"""

import argparse
import copy
import random
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import NiftiLiverDataset
from src.data.nifti import list_cases
from src.data.splits import load_splits
from src.models.unet3d import build_model
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.provenance import atomic_write_text, git_state, is_tracked_and_clean, mirror_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "config.yaml"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_loaders(config: Dict, cases: Dict, splits: Dict, max_train: Optional[int] = None, max_val: Optional[int] = None):
    """Train/val loaders built ONLY from the case ids in ``splits`` (which must not contain a 'test' key)."""
    assert "test" not in splits, "test case names must not be loaded for training"
    cfg_tr, cfg_ds = config["training"], config["dataset"]
    train_ids = list(splits["train"])[:max_train]
    val_ids = list(splits["val"])[:max_val]
    train_ds = NiftiLiverDataset({c: cases[c] for c in train_ids}, "train", config)
    val_ds = NiftiLiverDataset({c: cases[c] for c in val_ids}, "val", config)
    train_loader = DataLoader(train_ds, batch_size=cfg_tr["batch_size"], shuffle=True,
                              drop_last=len(train_ds) >= cfg_tr["batch_size"],
                              num_workers=cfg_ds["num_workers"], pin_memory=cfg_ds["pin_memory"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg_ds["num_workers"], pin_memory=cfg_ds["pin_memory"])
    return train_loader, val_loader


def check_reproducibility_state(splits_path: str, allow_dirty: bool) -> Dict[str, object]:
    """Refuse to train from an uncommitted split file or (unless allowed) a dirty working tree."""
    if not allow_dirty:
        if not is_tracked_and_clean(splits_path):
            raise SystemExit(f"Refusing to train: {splits_path} is not committed (or has local edits). "
                             "Commit the frozen split file first, so the test set is fixed before any training.")
        state = git_state()
        if state["dirty"] is not False:
            raise SystemExit("Refusing to train from a working tree with uncommitted changes to tracked files "
                             "(the recorded git commit would not describe the code that ran). Use --allow-dirty only for experiments.")
    return git_state()


def run_training(
    data_dir: str,
    splits_path: str,
    out_dir: str,
    seed: int = 0,
    config_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
    mirror_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    stop_after_epoch: Optional[int] = None,
    allow_dirty: bool = False,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    smoke_test: bool = False,
    device: Optional[torch.device] = None,
) -> Trainer:
    """Train under the protocol and return the ``Trainer`` (its ``best_dice`` is a VALIDATION value)."""
    check_reproducibility_state(splits_path, allow_dirty)
    with open(config_path or DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f)
    for section, values in (overrides or {}).items():
        config.setdefault(section, {}).update(values)

    splits = load_splits(splits_path, include_test=False)
    out = Path(out_dir)
    ckpt_dir = out / "checkpoints"
    config["protocol"] = {
        "protocol_version": splits.get("protocol_version"),
        "splits_sha256": splits["sha256"],
        "seed": seed,
        "selection_split": "val",
        "n_train": len(splits["train"][:max_train]),
        "n_val": len(splits["val"][:max_val]),
        "smoke_test": smoke_test,
    }
    config["checkpoint"]["save_dir"] = str(ckpt_dir)
    config["checkpoint"]["mirror_dir"] = mirror_dir
    config["logging"]["metrics_csv"] = "metrics.csv"

    # Record exactly what ran, next to the checkpoints (and on persistent storage if mirrored).
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "run_config.yaml", yaml.safe_dump(copy.deepcopy(config)))
    shutil.copyfile(splits_path, out / Path(splits_path).name)
    if mirror_dir:
        mirror_file(out / "run_config.yaml", mirror_dir, verify_hash=False)
        mirror_file(out / Path(splits_path).name, mirror_dir, verify_hash=False)

    seed_everything(seed)
    device = device or get_device()
    cases = list_cases(data_dir)
    train_loader, val_loader = build_loaders(config, cases, splits, max_train, max_val)
    print(f"[protocol] train={config['protocol']['n_train']} val={config['protocol']['n_val']} "
          f"splits sha256={splits['sha256'][:12]} seed={seed}{' SMOKE TEST' if smoke_test else ''}")

    trainer = Trainer(build_model(config), config, device, str(ckpt_dir), resume_from=resume_from)
    trainer.fit(train_loader, val_loader, stop_after_epoch=stop_after_epoch)
    print(f"[train] best patch-level val Dice {trainer.best_dice:.4f} (validation split; not a test result)")
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="MSD Task03_Liver root (imagesTr/, labelsTr/)")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mirror-dir", default=None, help="Persistent directory (e.g. Google Drive) for verified checkpoint copies")
    parser.add_argument("--resume", default=None, help="Path to last.pth to continue an interrupted run")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="End this session after N epochs (resume later)")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run_training(args.data_dir, args.splits, args.out_dir, seed=args.seed, config_path=args.config,
                 mirror_dir=args.mirror_dir, resume_from=args.resume, stop_after_epoch=args.stop_after_epoch,
                 allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
