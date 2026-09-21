"""
Three-way case-level split: train / validation (model selection) / test (final).

Construction (protocol v1, seed 42, 85 / 20 / 26 of 131 cases)
    1. Identify the historical validation cases: the 26 cases that chose the
       historical checkpoint, obtained from the historical deterministic split
       logic (``historical_train_val_split``: seed 42, 20%), not guessed.
    2. Final-test eligibility: ``eligible = all_cases - historical_validation``
       (105 cases).  Historical validation cases were already used to evaluate and
       select the historical model, so they are excluded from the new final test.
    3. Select the 26 final-test cases from ``eligible`` with ``default_rng(42)``.
    4. Divide the remaining 105 cases (which include all 26 historical validation
       cases) into 85 train and 20 validation with the same generator.
    Historical validation cases may therefore appear in the new train or
    validation sets; they can never appear in the new test set.  This is an
    experimental-design constraint, not seed searching.

The split is generated ONCE from the real case listing, written to a JSON file that
is committed before any training, and identified everywhere by a SHA-256
fingerprint over the three case lists AND the construction (historical validation
ids, exclusion rule, seed).  Training code only receives the train and validation
lists (``load_splits(..., include_test=False)`` also drops the eligibility list);
the test list is exposed only to ``evaluate_protocol`` behind an explicit flag.

Frozen parameters: the CLI has no ``--seed`` option and ``write_splits`` never
overwrites, so a split cannot be regenerated until a favourable test set appears.
A different seed or fractions means a new protocol version (new file, new
fingerprint, new results) and must never be chosen by looking at performance.

No split is generated over a dataset with duplicate content: ``make`` refuses if two different
case ids share identical image or label voxel arrays (see ``validate_dataset``).

The historical listing is not guessed.  By default it is taken to be the MSD
listing itself, and generation is REFUSED unless that listing matches the recorded
evidence about the historical dataset (``check_historical_naming``).  If the real
naming differs, supply the historical listing explicitly with ``--historical-listing``.

CLI:
    python -m src.data.splits make --data-dir <Task03_Liver> --out splits/msd_task03_v1.json \
        [--historical-listing historical_case_names.txt] [--validation-report validation.json]
    python -m src.data.splits verify --splits splits/msd_task03_v1.json
    python -m src.data.splits describe --data-dir <Task03_Liver> --splits splits/msd_task03_v1.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.data.dataset import historical_train_val_split

PROTOCOL_VERSION = "v1"
DEFAULT_SEED = 42
EXPECTED_CASES = 131
HISTORICAL_VAL_FRACTION = 0.2          # config.yaml dataset.val_split of the historical run
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_VAL_FRACTION = 0.15
EXCLUSION_RULE = "test_excludes_historical_validation"
EXCLUSION_RULE_TEXT = ("Final-test cases must not be historical validation cases: those cases were already used "
                       "to evaluate and select the historical model. eligible_test = all_cases - historical_validation.")
HISTORICAL_DERIVATION = (
    "src.data.dataset.historical_train_val_split: np.random.default_rng(42).permutation(sorted case names); "
    "first max(1, int(n * 0.2)) names are the historical validation cases"
)

# What is known about the historical dataset listing without having the Kaggle mirror: its files were named
# liver_<id>_img.npy, and the historical run's overlay figures (kaggle/train_kaggle.py takes
# ``sorted(images_dir.glob("*.npy"))[:3]``, i.e. sorted by FULL FILENAME) were liver_0, liver_100, liver_101.
# This is a NECESSARY consistency check on a candidate listing, not proof that the listing is identical.
HISTORICAL_NAMING_PATTERN = r"liver_\d+"
HISTORICAL_FILE_SUFFIX = "_img.npy"
HISTORICAL_SORTED_FIRST_THREE = ["liver_0", "liver_100", "liver_101"]


def fingerprint(
    train: Sequence[str], val: Sequence[str], test: Sequence[str],
    historical_validation: Sequence[str], seed: int,
) -> str:
    """
    SHA-256 over the canonical content of the three case lists and of the
    construction (historical validation ids, exclusion rule, seed), so editing
    either the membership or the construction invalidates it.
    """
    payload = json.dumps(
        {"train": sorted(train), "val": sorted(val), "test": sorted(test),
         "historical_validation": sorted(historical_validation), "exclusion_rule": EXCLUSION_RULE, "seed": seed},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def check_historical_naming(case_ids: Sequence[str]) -> None:
    """
    Refuse to treat ``case_ids`` as the historical listing unless it is consistent
    with what is known about it (names ``liver_<id>``; sorted-first-three evidence).
    Raises instead of guessing.
    """
    bad = [c for c in case_ids if not re.fullmatch(HISTORICAL_NAMING_PATTERN, c)]
    if bad:
        raise ValueError(f"Case names such as {bad[:3]} do not follow the historical 'liver_<id>' convention; "
                         "the historical validation ids cannot be derived reliably. Provide the historical "
                         "listing explicitly (--historical-listing).")
    filenames = sorted(f"{c}{HISTORICAL_FILE_SUFFIX}" for c in case_ids)      # sorted like the historical glob
    first3 = [f[: -len(HISTORICAL_FILE_SUFFIX)] for f in filenames[:3]]
    if first3 != HISTORICAL_SORTED_FIRST_THREE:
        raise ValueError(f"Sorted filename listing starts with {first3}, but the historical run's first three sorted files "
                         f"were {HISTORICAL_SORTED_FIRST_THREE}. This is not the same listing; provide the "
                         "historical listing explicitly (--historical-listing).")


def historical_validation_cases(historical_listing: Sequence[str]) -> List[str]:
    """The historical validation ids, from the historical split logic applied to the historical listing."""
    _, val = historical_train_val_split(historical_listing, HISTORICAL_VAL_FRACTION)
    return sorted(val)


def make_splits(
    case_ids: Sequence[str],
    historical_listing: Sequence[str],
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    listing_source: str = "explicit",
) -> Dict[str, object]:
    """
    Build the split described in the module docstring.

    ``historical_listing`` is required (no default) so the historical ids are
    never silently assumed: pass the real MSD listing after ``check_historical_naming``
    or an explicitly supplied historical listing.  Every historical validation id
    must exist in ``case_ids``.
    """
    ids = sorted(case_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case ids")
    hist_val = historical_validation_cases(historical_listing)
    unknown = sorted(set(hist_val) - set(ids))
    if unknown:
        raise ValueError(f"Historical validation ids not present in the dataset: {unknown[:5]} "
                         "(naming mismatch; not guessing)")

    n = len(ids)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    eligible = sorted(set(ids) - set(hist_val))
    if n_test < 1 or n_val < 1 or n - n_test - n_val < 1:
        raise ValueError(f"Cannot split {n} cases with fractions {test_fraction}/{val_fraction}")
    if len(eligible) < n_test:
        raise ValueError(f"Only {len(eligible)} cases eligible for the final test, need {n_test}")

    rng = np.random.default_rng(seed)
    test = sorted(eligible[i] for i in rng.permutation(len(eligible))[:n_test])
    remaining = sorted(set(ids) - set(test))                   # includes every historical validation case
    val = sorted(remaining[i] for i in rng.permutation(len(remaining))[:n_val])
    train = sorted(set(remaining) - set(val))
    return {
        "dataset": "MSD Task03_Liver (labelled training cases)",
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "sha256": fingerprint(train, val, test, hist_val, seed),
        "historical_validation": {
            "case_ids": hist_val,
            "derivation": HISTORICAL_DERIVATION,
            "listing_source": listing_source,
            "n_in_new_train": len(set(hist_val) & set(train)),
            "n_in_new_val": len(set(hist_val) & set(val)),
            "n_in_new_test": 0,
        },
        "exclusion_rule": {"id": EXCLUSION_RULE, "text": EXCLUSION_RULE_TEXT},
        "test_eligible": {"n": len(eligible), "case_ids": eligible},
        "train": train,
        "val": val,
        "test": test,
    }


def validate_splits(splits: Dict[str, object]) -> str:
    """
    Check disjointness, exclusion rule, eligibility bookkeeping and that the
    stored fingerprint matches; return it.
    """
    parts: Dict[str, List[str]] = {k: list(splits[k]) for k in ("train", "val", "test")}  # type: ignore[arg-type]
    for name, cases in parts.items():
        if len(set(cases)) != len(cases):
            raise ValueError(f"Duplicate case in '{name}'")
        if not cases:
            raise ValueError(f"Split '{name}' is empty")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(parts[a]) & set(parts[b])
        if overlap:
            raise ValueError(f"Leakage: {sorted(overlap)[:5]} appear in both '{a}' and '{b}'")

    hist = list(splits["historical_validation"]["case_ids"])  # type: ignore[index]
    leaked = set(hist) & set(parts["test"])
    if leaked:
        raise ValueError(f"Leakage: historical validation cases {sorted(leaked)[:5]} are in the final test set "
                         "(violates the exclusion rule)")
    universe = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    eligible = list(splits["test_eligible"]["case_ids"])  # type: ignore[index]
    if sorted(eligible) != sorted(universe - set(hist)) or not set(parts["test"]) <= set(eligible):
        raise ValueError("Test-eligibility list is inconsistent with all_cases - historical_validation")
    if not set(hist) <= universe:
        raise ValueError("Historical validation ids are not all present in the split")

    actual = fingerprint(parts["train"], parts["val"], parts["test"], hist, splits["seed"])  # type: ignore[arg-type]
    if splits.get("sha256") != actual:
        raise ValueError("Split file fingerprint does not match its contents (edited after creation?)")
    return actual


def write_splits(splits: Dict[str, object], path: str) -> None:
    validate_splits(splits)
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists; splits are write-once. Choose a new file name for a new protocol version.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=1))


def load_splits(path: str, include_test: bool = False) -> Dict[str, object]:
    """
    Load and validate a split file.  With ``include_test=False`` (the default,
    used by every training path) the test case names and the test-eligibility
    list are removed from the returned dict, so training code cannot touch them
    by accident.  The fingerprint is still verified against the full file.
    """
    splits = json.loads(Path(path).read_text())
    validate_splits(splits)
    if not include_test:
        splits = {k: v for k, v in splits.items() if k not in ("test", "test_eligible")}
    return splits


def read_listing_file(path: str) -> List[str]:
    """One case name per line (``_img`` / ``.npy`` / ``.nii.gz`` suffixes tolerated); blank lines ignored."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        for suffix in (".nii.gz", ".npy", "_img"):
            if line.endswith(suffix):
                line = line[: -len(suffix)]
        names.append(line)
    return names


def describe_splits(cases: Dict[str, tuple], splits: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """
    Non-performance dataset characteristics per split, to spot an obviously
    pathological split BEFORE freezing it: case count, voxel spacing, volume
    shape and reference liver volume.  Reads headers and labels only; it never
    involves a model.  This is a sanity check, not a search over seeds.
    """
    import nibabel as nib

    out: Dict[str, Dict[str, float]] = {}
    for name in ("train", "val", "test"):
        rows = []
        for cid in splits[name]:  # type: ignore[union-attr]
            img_path, lbl_path = cases[cid]
            lbl = nib.load(str(lbl_path))
            zooms = np.linalg.norm(lbl.affine[:3, :3], axis=0)
            ml = float((np.asanyarray(lbl.dataobj) > 0).sum() * np.prod(zooms) / 1000.0)
            rows.append((*zooms, lbl.shape[2] if lbl.ndim == 3 else np.nan, ml))
        a = np.array(rows, dtype=float)
        out[name] = {
            "n": len(rows),
            "slice_thickness_mm_median": float(np.median(a[:, :3].max(axis=1))),
            "slice_thickness_mm_max": float(a[:, :3].max()),
            "inplane_mm_median": float(np.median(a[:, :3].min(axis=1))),
            "liver_ml_median": float(np.median(a[:, 4])),
            "liver_ml_min": float(a[:, 4].min()),
            "liver_ml_max": float(a[:, 4].max()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--data-dir", required=True)
    mk.add_argument("--out", required=True)
    mk.add_argument("--expect-n", type=int, default=EXPECTED_CASES,
                    help="Refuse to freeze a split if the dataset does not have exactly this many labelled cases")
    mk.add_argument("--validation-report", default=None,
                    help="JSON from `validate_dataset --report`; if given and passing for exactly this listing, the duplicate-"
                         "content check is not repeated. Otherwise it is computed here. There is no way to skip it.")
    mk.add_argument("--historical-listing", default=None,
                    help="Text file with the historical run's case names (one per line). Default: the MSD listing, "
                         "accepted only if it passes the historical-naming consistency check")
    vf = sub.add_parser("verify")
    vf.add_argument("--splits", required=True)
    ds = sub.add_parser("describe")
    ds.add_argument("--data-dir", required=True)
    ds.add_argument("--splits", required=True)
    args = parser.parse_args()

    if args.cmd == "make":
        from src.data.nifti import list_cases
        cases = list_cases(args.data_dir)
        case_ids = list(cases)
        if len(case_ids) != args.expect_n:
            raise SystemExit(f"Found {len(case_ids)} cases, expected {args.expect_n}: not freezing a split on an unexpected dataset.")
        from src.data.validate_dataset import check_content_before_split
        check_content_before_split(cases, args.validation_report)      # refuses duplicate images / labels
        if args.historical_listing:
            listing, source = read_listing_file(args.historical_listing), f"explicit file {args.historical_listing}"
        else:
            check_historical_naming(case_ids)                # raises instead of guessing
            listing, source = case_ids, "MSD listing (passed the historical-naming consistency check)"
        splits = make_splits(case_ids, listing, listing_source=source)
        write_splits(splits, args.out)
        print(f"Wrote {args.out}: {splits['counts']}  sha256={splits['sha256']}")
        h = splits["historical_validation"]
        print(f"Historical validation cases excluded from the test set: {len(h['case_ids'])} "
              f"(now in new train: {h['n_in_new_train']}, new val: {h['n_in_new_val']}, new test: 0); "
              f"test-eligible population: {splits['test_eligible']['n']}")
        print("Commit this file BEFORE training. Do not regenerate it.")
    elif args.cmd == "describe":
        from src.data.nifti import list_cases
        splits = load_splits(args.splits, include_test=True)
        for name, stats in describe_splits(list_cases(args.data_dir), splits).items():
            print(name, {k: round(v, 2) for k, v in stats.items()})
    else:
        s = json.loads(Path(args.splits).read_text())
        print(f"OK {s['counts']}  sha256={validate_splits(s)}")


if __name__ == "__main__":
    main()
