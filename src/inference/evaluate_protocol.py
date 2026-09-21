"""
Leakage-controlled, physical-unit evaluation of a checkpoint on NIfTI volumes.

What this measures
    Complete volumes: each case is loaded on its native grid, the whole volume
    is reconstructed by MONAI sliding-window inference (Gaussian blending), and
    the full-volume prediction is scored against the full-volume label.  This is
    NOT the 128^3 centre-crop score logged during training.

Units
    Voxel spacing (mm) is read from each NIfTI affine and passed to
    ``hausdorff_95``, so HD95 is a physical surface distance in millimetres.

Aggregation (per split)
    Per-case Dice and HD95 are computed first; the aggregate is the unweighted
    mean over cases (macro average, each patient counts once), reported with the
    sample standard deviation (ddof=1), median, inter-quartile range, min/max and
    a seeded 95% percentile-bootstrap confidence interval of the mean.  HD95 per
    case is max(directed 95th percentile pred->gt, directed 95th percentile
    gt->pred) over surface voxels.

Complete failures
    A case whose prediction is empty has no defined HD95 (stored as inf, status
    ``empty_prediction``).  It stays in Dice (Dice = 0) and is NEVER dropped from
    the report: the summary always states the total number of cases, how many
    have a finite HD95, how many are complete failures and which, and the HD95
    statistics are labelled ``finite_cases_only``.  A finite-case HD95 mean must
    not be quoted without the failure count beside it.

Leakage guards (see docs/EVALUATION_PROTOCOL.md)
    * the checkpoint must carry the fingerprint of the split file it was trained
      with, and it must match the split file given here;
    * the checkpoint must record that it was selected on the validation split;
    * the test split needs ``--confirm-final-test``, needs a prior validation
      evaluation of the SAME checkpoint (same SHA-256) in the output directory,
      and is locked to ONE checkpoint per split file: the lock lives beside the
      split file (``<splits>.test_lock.json``, override with ``--lock-file``), not
      in the output directory, so a fresh output directory does not bypass it.
      Every attempt is appended to the lock file.

Usage:
    python -m src.inference.evaluate_protocol --checkpoint best.pth \\
        --data-dir Task03_Liver --splits splits/msd_task03_v1.json --split val --out-dir results/protocol_v1
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.nifti import list_cases, load_case, preprocess_ct
from src.data.splits import load_splits
from src.inference.evaluate import evaluate_case, summarise
from src.inference.predict import predict_volume
from src.models.unet3d import build_model
from src.utils.device import get_device
from src.utils.provenance import git_state, sha256_file, verify_checkpoint_dict, utc_now

AGGREGATION_METHOD = (
    "Per-case Dice and HD95 (mm) on full reconstructed volumes; unweighted mean over cases "
    "(macro average) with sample std (ddof=1), median, IQR, min/max and a seeded 95% percentile "
    "bootstrap CI of the mean. HD95 = max of directed 95th percentiles of surface distances "
    "(pred->gt, gt->pred) using surface voxels and physical voxel spacing. Dice covers ALL cases. "
    "Complete failures (empty prediction, HD95 undefined) are counted and listed; HD95 statistics "
    "are over the finite-HD95 cases only and are labelled as such."
)


def predict_full_volume(model: nn.Module, volume: np.ndarray, config: Dict, device: torch.device) -> np.ndarray:
    """
    Sliding-window prediction over an entire preprocessed volume.

    Axes shorter than the inference patch are reflect-padded first (the same
    treatment training patches receive) and the prediction is cropped back, so
    the returned mask always has exactly ``volume.shape``.
    """
    patch = tuple(config["inference"]["patch_size"])
    pads = []
    for size, p in zip(volume.shape, patch):
        deficit = max(0, p - size)
        pads.append((deficit // 2, deficit - deficit // 2))
    padded = np.pad(volume, pads, mode="reflect") if any(a + b for a, b in pads) else volume
    pred = predict_volume(model, padded, config, device)
    return pred[tuple(slice(a, a + s) for (a, _), s in zip(pads, volume.shape))]


def evaluate_cases(
    model: nn.Module,
    config: Dict,
    cases: Dict[str, tuple],
    case_ids: List[str],
    device: torch.device,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Full-volume Dice and physical-mm HD95 for the named cases."""
    percentile = config["validation"]["hausdorff_percentile"]
    rows: List[Dict[str, object]] = []
    for i, cid in enumerate(case_ids, start=1):
        img_path, lbl_path = cases[cid]
        case = load_case(str(img_path), str(lbl_path))
        volume = preprocess_ct(case["image"], config["preprocessing"])
        pred = predict_full_volume(model, volume, config, device)
        spacing = case["spacing"]
        if not case["mask"].any():
            raise ValueError(f"{cid}: reference label is empty; this is a data error, not a model failure")
        scores = evaluate_case(pred, case["mask"], percentile, spacing)
        voxel_ml = float(np.prod(spacing)) / 1000.0
        status = "empty_prediction" if not pred.any() else "ok"
        rows.append({
            "case": cid,
            "status": status,
            "shape": "x".join(map(str, volume.shape)),
            "spacing_mm": "x".join(f"{s:.4g}" for s in spacing),
            "dice": scores["dice"],
            "hd95_mm": scores["hd95"] if status == "ok" else float("inf"),
            "gt_volume_ml": float(case["mask"].sum()) * voxel_ml,
            "pred_volume_ml": float(pred.sum()) * voxel_ml,
        })
        if verbose:
            hd = f"{scores['hd95']:.2f} mm" if status == "ok" else "undefined (EMPTY PREDICTION)"
            print(f"  [{i:02d}/{len(case_ids)}] {cid}: Dice {scores['dice']:.4f}  HD95 {hd}")
    return rows


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> List[float]:
    if values.size < 2:
        return [float(values.mean()), float(values.mean())] if values.size else [float("nan")] * 2
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _describe(a: np.ndarray, seed: int) -> Dict[str, object]:
    q = lambda p: float(np.percentile(a, p)) if a.size else float("nan")
    return {
        "mean": float(a.mean()) if a.size else float("nan"),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": q(50), "iqr": [q(25), q(75)],
        "min": float(a.min()) if a.size else float("nan"),
        "max": float(a.max()) if a.size else float("nan"),
        "mean_ci95": _bootstrap_ci(a, seed) if a.size else [float("nan")] * 2,
    }


def summarise_protocol(rows: List[Dict[str, object]], seed: int = 0) -> Dict[str, object]:
    """
    Aggregate per-case rows as described in ``AGGREGATION_METHOD``.

    Dice covers every case.  HD95 statistics cover only cases with a finite
    HD95 and are labelled as such; the number of cases excluded (complete
    failures) is reported next to them and the failed case ids are listed.
    """
    dice = np.array([r["dice"] for r in rows], dtype=float)
    hd_all = np.array([r["hd95_mm"] for r in rows], dtype=float)
    finite = np.isfinite(hd_all)
    failed = [r["case"] for r, ok in zip(rows, finite) if not ok]
    n, n_fin = len(rows), int(finite.sum())
    summary: Dict[str, object] = {
        "n_cases": n,
        "dice_all_cases": _describe(dice, seed),
        "hd95_mm": {
            "n_cases": n,
            "n_finite": n_fin,
            "n_complete_failures": n - n_fin,
            "complete_failure_cases": failed,
            "all_cases_finite": n_fin == n,
            "finite_cases_only": _describe(hd_all[finite], seed),
        },
        "aggregation": AGGREGATION_METHOD,
        "hd95_unit": "mm",
    }
    summary["report_text"] = format_report(summary)
    return summary


def format_report(summary: Dict[str, object]) -> str:
    """One-paragraph report that cannot show a finite-case HD95 without the failure count."""
    d, h = summary["dice_all_cases"], summary["hd95_mm"]
    n = summary["n_cases"]
    text = (f"Dice over all {n} cases: mean {d['mean']:.4f} (sd {d['std']:.4f}, median {d['median']:.4f}, "
            f"95% CI of mean {d['mean_ci95'][0]:.4f} to {d['mean_ci95'][1]:.4f}). ")
    f = h["finite_cases_only"]
    if h["n_complete_failures"]:
        text += (f"COMPLETE FAILURES: {h['n_complete_failures']} of {n} case(s) had an empty prediction "
                 f"({', '.join(h['complete_failure_cases'])}); HD95 is undefined for them. "
                 f"HD95 over the {h['n_finite']} of {n} finite cases only: ")
    else:
        text += f"HD95 over all {n} cases (no complete failures): "
    text += f"mean {f['mean']:.2f} mm (sd {f['std']:.2f}, median {f['median']:.2f}, max {f['max']:.2f})."
    return text


def check_protocol(ckpt: Dict, split_sha256: str, split: str) -> Dict:
    """Enforce that the checkpoint was trained/selected under this exact split file."""
    protocol = (ckpt.get("config") or {}).get("protocol")
    if not protocol:
        raise SystemExit("Refusing: checkpoint has no protocol metadata (legacy/historical model). "
                         "It was not trained under a recorded split, so it cannot be scored against these splits.")
    if protocol.get("splits_sha256") != split_sha256:
        raise SystemExit("Refusing: checkpoint was trained with a different split file "
                         f"({protocol.get('splits_sha256')} != {split_sha256}).")
    if protocol.get("selection_split") != "val":
        raise SystemExit("Refusing: checkpoint does not record selection on the validation split.")
    return protocol


def _default_lock(splits_path: str) -> Path:
    return Path(str(splits_path) + ".test_lock.json")


def run_evaluation(
    checkpoint: str,
    data_dir: str,
    splits_path: str,
    split: str,
    out_dir: str,
    device: torch.device,
    confirm_final_test: bool = False,
    lock_file: Optional[str] = None,
    bootstrap_seed: int = 0,
    case_subset: Optional[List[str]] = None,
    smoke_test: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Evaluate one checkpoint on one split under the protocol's guards and write
    ``<split>_per_case.csv`` / ``<split>_summary.json`` into ``out_dir``.

    ``case_subset`` (validation only) restricts to named cases; it exists for
    the smoke test, whose output is stamped as not a result.
    """
    if split == "test" and not confirm_final_test:
        raise SystemExit("Refusing to touch the test split without --confirm-final-test.")
    if case_subset is not None and split != "val":
        raise SystemExit("A case subset is only allowed on the validation split.")

    splits = load_splits(splits_path, include_test=(split == "test"))
    ckpt = torch.load(checkpoint, map_location=device)
    protocol = check_protocol(ckpt, splits["sha256"], split)  # type: ignore[arg-type]
    try:
        verify_checkpoint_dict(ckpt, checkpoint, splits["sha256"])  # type: ignore[arg-type]
    except ValueError as e:
        raise SystemExit(f"Refusing: {e}")
    ckpt_hash = sha256_file(checkpoint)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = Path(lock_file) if lock_file else _default_lock(splits_path)
    if split == "test":
        val_summary = out / "val_summary.json"
        ok = val_summary.exists() and json.loads(val_summary.read_text()).get("checkpoint_sha256") == ckpt_hash
        if not ok:
            raise SystemExit("Refusing: evaluate this exact checkpoint on the validation split first "
                             f"(no matching {val_summary.name} in {out}); the test set is for a frozen checkpoint only.")
        if lock.exists():
            prior = json.loads(lock.read_text())
            if prior["checkpoint_sha256"] != ckpt_hash:
                raise SystemExit("Refusing: the test split was already evaluated with a different checkpoint "
                                 f"({prior['checkpoint_sha256'][:12]}, lock file {lock}). Selecting among models on the "
                                 "test set invalidates it; start a new protocol version with a new split instead.")

    config = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cases = list_cases(data_dir)
    case_ids = list(case_subset) if case_subset is not None else list(splits[split])  # type: ignore[arg-type]
    missing = [c for c in case_ids if c not in cases or (case_subset is not None and c not in splits["val"])]  # type: ignore[operator]
    if missing:
        raise SystemExit(f"Cases not available for this split: {missing[:5]}")

    if verbose:
        print(f"Full-volume evaluation of {len(case_ids)} '{split}' cases (HD95 in mm)")
    rows = evaluate_cases(model, config, cases, case_ids, device, verbose=verbose)
    summary = summarise_protocol(rows, bootstrap_seed)
    git = git_state()
    summary.update({
        "split": split,
        "splits_file": str(splits_path),
        "splits_sha256": splits["sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection_val_dice_patch": ckpt.get("val_dice"),
        "training_seed": protocol.get("seed"),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "evaluated_utc": utc_now(),
    })
    if smoke_test:
        summary["SMOKE_TEST"] = "Infrastructure test only. These numbers are NOT a model result and must not be reported."

    with open(out / f"{split}_per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))

    if split == "test":
        prior = json.loads(lock.read_text()) if lock.exists() else {
            "splits_sha256": splits["sha256"], "checkpoint_sha256": ckpt_hash, "first_evaluated_utc": utc_now(), "attempts": []}
        prior["attempts"].append({"utc": utc_now(), "checkpoint_sha256": ckpt_hash, "out_dir": str(out)})
        lock.write_text(json.dumps(prior, indent=2))

    if verbose:
        print("\n" + ("[SMOKE TEST, NOT A RESULT] " if smoke_test else "") + str(summary["report_text"]))
        print(f"Wrote {out}/{split}_per_case.csv and {split}_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="MSD Task03_Liver root (imagesTr/, labelsTr/)")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Required for --split test. Test results must not influence any later modelling choice.")
    parser.add_argument("--lock-file", default=None, help="Default: <splits file>.test_lock.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.data_dir, args.splits, args.split, args.out_dir, get_device(),
                   confirm_final_test=args.confirm_final_test, lock_file=args.lock_file,
                   bootstrap_seed=args.bootstrap_seed)


if __name__ == "__main__":
    main()
