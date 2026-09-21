"""
NIfTI loading that keeps the physical geometry of each CT volume.

The historical ``.npy`` pipeline discarded voxel spacing, so surface distances
could only be measured in voxels.  Everything here returns the array together
with its voxel spacing in millimetres, in the SAME axis order as the returned
array, so that ``hausdorff_95(..., voxel_spacing=spacing)`` yields millimetres.

Axis handling: volumes are reoriented to the closest canonical (RAS+) frame
with ``nibabel.as_closest_canonical``.  This may permute or flip array axes, so
spacing is recomputed from the reoriented affine (norm of each affine column)
rather than read from the header, which guarantees ``spacing[i]`` belongs to
array axis ``i`` whatever the on-disk orientation was.
"""

from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np

Spacing = Tuple[float, float, float]

_ORTHOGONALITY_TOL = 1e-3


def _axis_spacing(affine: np.ndarray) -> Spacing:
    """Voxel size (mm) along each array axis, from the affine's column norms."""
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    # Physical distance = index distance * spacing only holds for an
    # orthogonal (non-sheared) voxel grid.  MSD volumes are axis-aligned; any
    # other file must be handled with the full affine, so refuse it loudly.
    unit = linear / spacing
    off_diag = np.abs(unit.T @ unit - np.eye(3)).max()
    if off_diag > _ORTHOGONALITY_TOL:
        raise ValueError(
            "Sheared/oblique voxel grid: index*spacing is not a physical distance "
            f"(max off-diagonal {off_diag:.3g}). Resample to an orthogonal grid first."
        )
    if not np.all(spacing > 0):
        raise ValueError(f"Non-positive voxel spacing derived from affine: {spacing}")
    return tuple(float(s) for s in spacing)  # type: ignore[return-value]


def load_nifti(path: str, is_label: bool = False) -> Tuple[np.ndarray, Spacing, np.ndarray]:
    """
    Load a NIfTI file in canonical orientation.

    Returns:
        (array, spacing_mm, affine). ``array`` is float32 for images and uint8
        for labels (rounded, so interpolated header scaling cannot create
        fractional labels); ``spacing_mm[i]`` is the voxel size along array axis i.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    if img.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D volume, got shape {img.shape}")
    spacing = _axis_spacing(img.affine)
    if is_label:
        data = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
    else:
        data = img.get_fdata(dtype=np.float32)
    return data, spacing, np.asarray(img.affine)


def load_case(image_path: str, label_path: str) -> Dict[str, object]:
    """
    Load an image/label pair and check that they share one geometry.

    The label is binarised exactly as in the historical pipeline (every positive
    label, liver and tumour, is foreground).  A mismatch in shape or spacing
    raises instead of silently scoring misaligned volumes.
    """
    image, spacing, affine = load_nifti(image_path)
    label, label_spacing, label_affine = load_nifti(label_path, is_label=True)
    if image.shape != label.shape:
        raise ValueError(f"Shape mismatch: image {image.shape} vs label {label.shape} ({image_path})")
    if not np.allclose(spacing, label_spacing, rtol=1e-4) or not np.allclose(affine, label_affine, rtol=1e-4, atol=1e-3):
        raise ValueError(f"Image and label geometry differ for {image_path}")
    return {"image": image, "mask": (label > 0).astype(np.uint8), "spacing": spacing}


def preprocess_ct(volume: np.ndarray, cfg_pre: Dict) -> np.ndarray:
    """
    Intensity pre-processing shared by training and evaluation: HU clipping and
    per-volume z-score (identical to ``LiverCTDataset`` / ``predict_from_file``).
    """
    lo, hi = cfg_pre["intensity_clip"]
    volume = np.clip(volume.astype(np.float32), lo, hi)
    if cfg_pre["normalize"]:
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
    return volume


def list_cases(data_dir: str, images_subdir: str = "imagesTr", labels_subdir: str = "labelsTr") -> Dict[str, Tuple[Path, Path]]:
    """Map case id -> (image, label) for an MSD-layout directory (skips hidden ``._*`` files)."""
    images = Path(data_dir) / images_subdir
    labels = Path(data_dir) / labels_subdir
    cases = {}
    for p in sorted(images.glob("*.nii.gz")):
        if p.name.startswith("."):
            continue
        case_id = p.name[: -len(".nii.gz")]
        lbl = labels / p.name
        if not lbl.exists():
            raise FileNotFoundError(f"No label for {case_id}: {lbl}")
        cases[case_id] = (p, lbl)
    if not cases:
        raise FileNotFoundError(f"No .nii.gz volumes in {images}")
    return cases
