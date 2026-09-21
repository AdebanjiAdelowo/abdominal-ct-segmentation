"""
End-to-end infrastructure smoke test for the protocol.  NOT A RESULT.

Runs, on a handful of cases and 1-2 epochs, the same code paths a real run uses:

    NIfTI loading -> orientation / geometry handling -> train / val loading
    -> training -> checkpoint saving (+ optional persistent mirror)
    -> checkpoint reload + split-fingerprint verification (+ resume check)
    -> full-volume sliding-window inference -> Dice -> physical-mm HD95
    -> result serialisation.

Only the train and validation case lists are loaded (the test names are never
read).  Every number it produces comes from a barely trained model on a few
cases; it exists to prove the plumbing works and must never be reported.

Usage:
    python -m src.smoke_test --data-dir /content/data/Task03_Liver --splits splits/msd_task03_v1.json \
        --out-dir /content/drive/MyDrive/liver/smoke [--mirror-dir ...]
    python -m src.smoke_test --synthetic --out-dir /tmp/smoke     # no data needed
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
import torch

from src.data import splits as sp
from src.data.nifti import list_cases, load_case
from src.inference.evaluate_protocol import run_evaluation
from src.training.protocol_train import run_training
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.provenance import atomic_write_text, load_verified_checkpoint, sha256_file

BANNER = "SMOKE TEST: infrastructure check only. Nothing here is a model result."

SYNTHETIC_SPACING = (3.0, 0.7, 0.9)   # mm along the file axes (all different on purpose)
SYNTHETIC_OVERRIDES = {
    "dataset": {"num_workers": 0, "pin_memory": False},
    "preprocessing": {"patch_size": [16, 16, 16]},
    "model": {"base_features": 4, "depth": 2},
    "training": {"batch_size": 2, "epochs": 2},
    "inference": {"patch_size": [16, 16, 16], "sw_batch_size": 2},
}


def make_synthetic_dataset(root: Path, n_cases: int = 10, shape=(40, 36, 32)) -> Path:
    """Tiny MSD-layout NIfTI dataset: anisotropic spacing, deliberately non-canonical orientation."""
    affine = np.zeros((4, 4))
    affine[2, 0], affine[0, 1], affine[1, 2], affine[3, 3] = SYNTHETIC_SPACING[0], SYNTHETIC_SPACING[1], SYNTHETIC_SPACING[2], 1
    rng = np.random.default_rng(0)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    for i in range(n_cases):
        lo = rng.integers(4, 8, 3)
        size = rng.integers(16, 22, 3)
        label = np.zeros(shape, np.uint8)
        label[lo[0]:lo[0] + size[0], lo[1]:lo[1] + size[1], lo[2]:lo[2] + size[2]] = 1
        image = (np.where(label > 0, 80.0, -60.0) + rng.normal(0, 15, shape)).astype(np.float32)
        nib.save(nib.Nifti1Image(image, affine), str(root / "imagesTr" / f"liver_{i}.nii.gz"))
        nib.save(nib.Nifti1Image(label, affine), str(root / "labelsTr" / f"liver_{i}.nii.gz"))
    return root


def run_smoke_test(
    data_dir: str,
    splits_path: str,
    out_dir: str,
    max_train: int = 4,
    max_val: int = 2,
    epochs: int = 2,
    overrides: Optional[Dict] = None,
    mirror_dir: Optional[str] = None,
    allow_dirty: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    device = device or get_device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "NOT_A_RESULT.txt").write_text(BANNER + "\n")
    steps: Dict[str, str] = {}

    def ok(name: str, detail: str = "") -> None:
        steps[name] = "ok" + (f": {detail}" if detail else "")
        print(f"[smoke] {name}: ok {detail}")

    print("[smoke] " + BANNER)
    splits = sp.load_splits(splits_path, include_test=False)         # test names never loaded
    assert "test" not in splits
    ok("splits", f"fingerprint {splits['sha256'][:12]}, test names not loaded")

    cases = list_cases(data_dir)
    subset = list(splits["train"])[:max_train] + list(splits["val"])[:max_val]
    geometry = {}
    for cid in subset:
        c = load_case(str(cases[cid][0]), str(cases[cid][1]))       # raises on geometry mismatch / shear
        geometry[cid] = [round(float(x), 4) for x in c["spacing"]]
    ok("nifti_geometry", f"{len(subset)} cases, spacing (canonical axis order) e.g. {next(iter(geometry.values()))} mm")

    cfg_over = json.loads(json.dumps(overrides or {}))
    cfg_over.setdefault("training", {})["epochs"] = epochs
    run_dir = out / "run"
    trainer = run_training(
        data_dir, splits_path, str(run_dir), seed=0, overrides=cfg_over, mirror_dir=mirror_dir,
        allow_dirty=allow_dirty, max_train=max_train, max_val=max_val, smoke_test=True, device=device,
    )
    ok("training", f"{epochs} epoch(s)")

    best, last = run_dir / "checkpoints" / "best.pth", run_dir / "checkpoints" / "last.pth"
    for p in (best, last):
        ckpt = load_verified_checkpoint(str(p), device, expected_split_sha256=splits["sha256"])
    ok("checkpoint_reload", f"fingerprint verified, epoch {ckpt['epoch']}, commit {str(ckpt['git_commit'])[:8]}")
    manifest = json.loads((run_dir / "checkpoints" / "checkpoint_manifest.json").read_text())
    assert manifest["best"]["sha256"] == sha256_file(best), "manifest hash does not match best.pth"
    ok("checkpoint_manifest", "sha256 matches best.pth")
    if mirror_dir:
        assert sha256_file(Path(mirror_dir) / "best.pth") == sha256_file(best), "mirror copy differs"
        ok("checkpoint_mirror", f"best.pth verified in {mirror_dir}")

    from src.models.unet3d import build_model
    resumed = Trainer(build_model(ckpt["config"]), ckpt["config"], device, str(out / "resume_check"), resume_from=str(last))
    assert resumed.start_epoch == ckpt["epoch"] + 1
    ok("resume", f"would continue at epoch {resumed.start_epoch}")

    summary = run_evaluation(
        str(best), data_dir, splits_path, "val", str(out / "eval"), device,
        case_subset=list(splits["val"])[:max_val], smoke_test=True, verbose=False,
    )
    assert 0.0 <= summary["dice_all_cases"]["mean"] <= 1.0
    assert summary["hd95_unit"] == "mm" and summary["SMOKE_TEST"]
    rows = (out / "eval" / "val_per_case.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + max_val
    ok("full_volume_eval_and_serialisation", f"{max_val} val cases -> val_per_case.csv, val_summary.json (mm)")

    report = {"banner": BANNER, "steps": steps, "cases_used": subset, "spacing_mm": geometry,
              "note": "Metrics in eval/ come from a barely trained model on a few cases. Not a result."}
    atomic_write_text(out / "SMOKE_TEST_REPORT.json", json.dumps(report, indent=2))
    print("[smoke] all steps passed. " + BANNER)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--splits")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mirror-dir", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=4)
    parser.add_argument("--max-val", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use a generated tiny dataset (no MSD data needed)")
    args = parser.parse_args()

    if args.synthetic:
        out = Path(args.out_dir)
        data = make_synthetic_dataset(out / "synthetic" / "Task03_Liver")
        ids = list(list_cases(str(data)))
        splits = sp.make_splits(ids, ids, listing_source="synthetic dataset")
        path = out / "synthetic" / "splits.json"
        if not path.exists():
            sp.write_splits(splits, str(path))
        run_smoke_test(str(data), str(path), args.out_dir, args.max_train, args.max_val, args.epochs,
                       overrides=SYNTHETIC_OVERRIDES, mirror_dir=args.mirror_dir, allow_dirty=True)
    else:
        if not (args.data_dir and args.splits):
            parser.error("--data-dir and --splits are required unless --synthetic")
        run_smoke_test(args.data_dir, args.splits, args.out_dir, args.max_train, args.max_val, args.epochs,
                       mirror_dir=args.mirror_dir, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
