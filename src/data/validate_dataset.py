"""
Validate an MSD Task03_Liver directory before freezing a split or training.

Checks (errors stop the run, warnings are printed):
  * ``imagesTr/`` and ``labelsTr/`` exist; exactly ``--expect-cases`` volumes
    (default 131); every image has a label and vice versa (hidden ``._*`` files
    from macOS archives are ignored and counted);
  * per case: 3-D volumes, identical image/label shape and geometry, a
    non-sheared grid with positive spacing in a plausible range, labels only
    in {0, 1, 2} with a non-empty foreground;
  * with ``--deep``: image intensities look like Hounsfield units.

It reads geometry and labels only unless ``--deep`` is given; it never touches a model.

Usage:
    python -m src.data.validate_dataset --data-dir /content/data/Task03_Liver
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np

from src.data.nifti import _axis_spacing


def validate_dataset(data_dir: str, expect_cases: int = 131, deep: bool = False) -> Dict[str, object]:
    root = Path(data_dir)
    errors: List[str] = []
    warnings: List[str] = []
    images_dir, labels_dir = root / "imagesTr", root / "labelsTr"
    for d in (images_dir, labels_dir):
        if not d.is_dir():
            errors.append(f"Missing directory: {d}")
    if errors:
        return {"n_cases": 0, "errors": errors, "warnings": warnings}

    hidden = [p.name for d in (images_dir, labels_dir) for p in d.glob("._*")]
    if hidden:
        warnings.append(f"Ignored {len(hidden)} hidden '._*' file(s)")
    images = {p.name: p for p in images_dir.glob("*.nii.gz") if not p.name.startswith(".")}
    labels = {p.name: p for p in labels_dir.glob("*.nii.gz") if not p.name.startswith(".")}
    if len(images) != expect_cases:
        errors.append(f"Found {len(images)} images, expected {expect_cases}")
    for name in sorted(set(images) - set(labels)):
        errors.append(f"Image without label: {name}")
    for name in sorted(set(labels) - set(images)):
        errors.append(f"Label without image: {name}")

    spacings = []
    for name in sorted(set(images) & set(labels)):
        try:
            img, lbl = nib.load(str(images[name])), nib.load(str(labels[name]))
            if img.ndim != 3 or lbl.ndim != 3:
                errors.append(f"{name}: not 3-D (image {img.shape}, label {lbl.shape})")
                continue
            if img.shape != lbl.shape:
                errors.append(f"{name}: shape mismatch image {img.shape} vs label {lbl.shape}")
            if not np.allclose(img.affine, lbl.affine, rtol=1e-4, atol=1e-3):
                errors.append(f"{name}: image/label affines differ")
            sp = _axis_spacing(img.affine)          # raises on shear / non-positive spacing
            spacings.append(sp)
            if not all(0.2 <= s <= 12 for s in sp):
                warnings.append(f"{name}: unusual spacing {tuple(round(s, 3) for s in sp)} mm")
            vals = np.unique(np.rint(np.asanyarray(lbl.dataobj)).astype(np.int64))
            if not set(vals.tolist()) <= {0, 1, 2}:
                errors.append(f"{name}: unexpected label values {vals.tolist()}")
            if not (vals > 0).any():
                errors.append(f"{name}: label has no foreground")
            if deep:
                data = np.asanyarray(img.dataobj)
                lo, hi = float(data.min()), float(data.max())
                if not (-2500 < lo < 0 and hi > 100):
                    warnings.append(f"{name}: intensity range [{lo:.0f}, {hi:.0f}] does not look like HU")
        except Exception as e:  # unreadable or malformed file
            errors.append(f"{name}: {type(e).__name__}: {e}")

    summary: Dict[str, object] = {"n_cases": len(images), "errors": errors, "warnings": warnings}
    if spacings:
        a = np.array(spacings)
        summary["spacing_mm_min"] = [round(float(x), 3) for x in a.min(axis=0)]
        summary["spacing_mm_max"] = [round(float(x), 3) for x in a.max(axis=0)]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--expect-cases", type=int, default=131)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    s = validate_dataset(args.data_dir, args.expect_cases, args.deep)
    for w in s["warnings"]:
        print("WARNING:", w)
    for e in s["errors"]:
        print("ERROR:", e)
    print(f"{s['n_cases']} cases; spacing range (array order) {s.get('spacing_mm_min')} to {s.get('spacing_mm_max')} mm; "
          f"{len(s['errors'])} error(s), {len(s['warnings'])} warning(s)")
    sys.exit(1 if s["errors"] else 0)


if __name__ == "__main__":
    main()
