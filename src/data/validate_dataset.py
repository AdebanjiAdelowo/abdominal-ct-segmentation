"""
Validate an MSD Task03_Liver directory before freezing a split or training.

Structural checks (errors stop the run, warnings are printed):
  * ``imagesTr/`` and ``labelsTr/`` exist; exactly ``--expect-cases`` volumes
    (default 131); every image has a label and vice versa (hidden ``._*`` files
    from macOS archives are ignored and counted);
  * per case: 3-D volumes, identical image/label shape and geometry, a
    non-sheared grid with positive spacing in a plausible range, labels only
    in {0, 1, 2} with a non-empty foreground;
  * with ``--deep``: image intensities look like Hounsfield units.

Content checks (on by default; ``--no-content-check`` skips them and is only for
quick structural inspection, never before generating a split):
  * DUPLICATE IMAGES: two different case ids whose image voxel arrays are
    identical are an ERROR.
  * DUPLICATE LABELS: two different case ids whose label voxel arrays are
    identical are an ERROR.

What is hashed.  Content, never filenames.  Images: SHA-256 over the array
shape and its voxel values as C-order float32 (after NIfTI scaling), so files
that differ only in header, affine, dtype encoding or compression hash the
same.  Labels: SHA-256 over the shape and the rounded uint8 label values.  Both
are streamed in bounded chunks.  Only EXACT copies are detected: a flipped,
transposed, cropped or resampled copy, or an image differing in a single voxel,
has a different hash.  Geometry is validated separately (above).

Why duplicate labels fail rather than warn.  Labels are already required to be
non-empty, and two different patients cannot legitimately have voxel-identical
non-empty 3-D masks.  An identical mask under two case ids means a copied or
mislabelled file, which would silently corrupt training or evaluation, and
duplicated images carry the same risk of leakage between splits.  Either way
the dataset must be investigated before a split is generated.

It reads geometry, labels and (for content checks) images; it never touches a model.

Usage:
    python -m src.data.validate_dataset --data-dir /content/data/Task03_Liver --report validation.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from src.data.nifti import _axis_spacing


HASH_METHOD = ("SHA-256 over array shape and voxel values: images as C-order float32 after NIfTI scaling, "
               "labels as rounded uint8; headers/affines excluded; exact copies only")
_CHUNK_SLABS = 16


def _hash_array(arr: np.ndarray, dtype) -> str:
    """SHA-256 of ``arr``'s shape and its values as C-order ``dtype``, streamed in slabs along axis 0."""
    h = hashlib.sha256()
    h.update(repr(tuple(int(x) for x in arr.shape)).encode())
    for start in range(0, arr.shape[0], _CHUNK_SLABS):
        h.update(np.ascontiguousarray(arr[start:start + _CHUNK_SLABS], dtype=dtype))
    return h.hexdigest()


def image_content_hash(path) -> str:
    img = nib.load(str(path))
    return _hash_array(img.get_fdata(dtype=np.float32), np.float32)


def label_content_hash(path) -> str:
    img = nib.load(str(path))
    return _hash_array(np.rint(np.asanyarray(img.dataobj)), np.uint8)


def content_hashes(cases: Dict[str, Tuple[Path, Path]]) -> Dict[str, Dict[str, str]]:
    """Content hashes for every case: ``{"image": {case_id: hash}, "label": {case_id: hash}}``."""
    return {
        "image": {cid: image_content_hash(img) for cid, (img, _) in cases.items()},
        "label": {cid: label_content_hash(lbl) for cid, (_, lbl) in cases.items()},
    }


def find_content_duplicates(hashes: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    """Groups of two or more different case ids sharing one content hash, per category."""
    groups = []
    for category, per_case in hashes.items():
        by_hash: Dict[str, List[str]] = {}
        for cid, h in per_case.items():
            by_hash.setdefault(h, []).append(cid)
        for h, ids in sorted(by_hash.items()):
            if len(ids) > 1:
                groups.append({"category": category, "sha256": h, "cases": sorted(ids)})
    return groups


def duplicate_messages(groups: List[Dict[str, object]]) -> List[str]:
    return [f"Duplicate {g['category']} content across different case ids "
            f"(sha256 {str(g['sha256'])[:16]}...): {', '.join(g['cases'])}" for g in groups]  # type: ignore[arg-type]


def check_content_before_split(cases: Dict[str, Tuple[Path, Path]], report_path: Optional[str] = None) -> None:
    """
    Gate used by ``splits make``: raise ``SystemExit`` unless the dataset content
    is free of duplicate images and labels.  With ``report_path`` (from
    ``validate_dataset --report``) the recorded result is reused if it covers
    exactly these cases and passed; otherwise the hashes are computed here.
    """
    if report_path:
        rep = json.loads(Path(report_path).read_text())
        if not rep.get("content_check"):
            raise SystemExit(f"{report_path} was produced with --no-content-check; it cannot clear the dataset for splitting.")
        if sorted(rep.get("case_ids", [])) != sorted(cases):
            raise SystemExit(f"{report_path} does not cover exactly the current case listing.")
        if rep.get("errors") or rep.get("duplicates"):
            raise SystemExit(f"{report_path} records validation errors or duplicates; resolve them first.")
        return
    groups = find_content_duplicates(content_hashes(cases))
    if groups:
        raise SystemExit("Refusing to generate a split: duplicate dataset content found, investigate first.\n  "
                         + "\n  ".join(duplicate_messages(groups)))


def validate_dataset(data_dir: str, expect_cases: int = 131, deep: bool = False, content_check: bool = True) -> Dict[str, object]:
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

    duplicates: List[Dict[str, object]] = []
    hashes: Dict[str, Dict[str, str]] = {}
    paired = {n[: -len(".nii.gz")]: (images[n], labels[n]) for n in sorted(set(images) & set(labels))}
    if content_check and paired:
        try:
            hashes = content_hashes(paired)
            duplicates = find_content_duplicates(hashes)
            errors.extend(duplicate_messages(duplicates))
        except Exception as e:  # unreadable file: reported, structural loop already flags most causes
            errors.append(f"content check failed: {type(e).__name__}: {e}")

    summary: Dict[str, object] = {"n_cases": len(images), "errors": errors, "warnings": warnings,
                                  "content_check": content_check, "duplicates": duplicates,
                                  "case_ids": sorted(paired), "hash_method": HASH_METHOD, "hashes": hashes}
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
    parser.add_argument("--no-content-check", action="store_true",
                        help="Skip duplicate-content checks (structural inspection only; not sufficient before splitting)")
    parser.add_argument("--report", default=None, help="Write the full result (including per-case content hashes) as JSON")
    args = parser.parse_args()
    s = validate_dataset(args.data_dir, args.expect_cases, args.deep, content_check=not args.no_content_check)
    if args.report:
        Path(args.report).write_text(json.dumps(s, indent=1))
    for w in s["warnings"]:
        print("WARNING:", w)
    for e in s["errors"]:
        print("ERROR:", e)
    print(f"content check: {'ON' if s['content_check'] else 'SKIPPED'}; duplicate groups: {len(s['duplicates'])}")
    print(f"{s['n_cases']} cases; spacing range (array order) {s.get('spacing_mm_min')} to {s.get('spacing_mm_max')} mm; "
          f"{len(s['errors'])} error(s), {len(s['warnings'])} warning(s)")
    sys.exit(1 if s["errors"] else 0)


if __name__ == "__main__":
    main()
