"""
Segmentation quality metrics used during validation.

Dice similarity coefficient and percentile Hausdorff distance are the two
primary metrics reported in medical image segmentation benchmarks.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> float:
    """
    Volumetric Dice similarity coefficient for binary tensors.

    Args:
        pred:    Thresholded binary prediction, any shape with values in {0, 1}.
        target:  Ground-truth binary mask, same shape as pred.
        smooth:  Laplace smoothing term to avoid 0/0 on empty masks.

    Returns:
        Dice score in [0, 1].
    """
    p = pred.view(-1).float()
    t = target.view(-1).float()
    intersection = (p * t).sum()
    return float((2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth))


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """
    Extract the surface (boundary) voxels of a binary mask.

    A voxel is on the surface if it is foreground but has at least one
    neighbour (6-connectivity, i.e. the default `scipy.ndimage` structuring
    element) that is background or lies outside the volume. Implemented as
    mask XOR erosion(mask); `binary_erosion` uses border_value=0, so
    foreground voxels touching the volume boundary are correctly kept as
    surface. If erosion removes the whole object (e.g. a single isolated
    voxel, or any object with no interior), every foreground voxel is
    itself surface, which this formulation returns automatically.
    """
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def hausdorff_95(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: int = 95,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Percentile Hausdorff distance between two binary segmentation masks.

    Standard definition: surface voxels are extracted from each mask via
    binary erosion, nearest-neighbour surface-to-surface distances are
    computed independently in each direction (pred -> target and
    target -> pred) using scipy's cKDTree, the requested percentile is
    taken of EACH direction separately, and the final HD95 is the MAX of
    the two directional percentiles (the standard "directed-then-max"
    convention; this is what makes the metric a true, asymmetric-robust
    Hausdorff distance rather than a pooled nearest-neighbour statistic).

    Args:
        pred:          Binary prediction, shape (..., D, H, W), values in {0, 1}.
        target:        Binary ground-truth, same shape as pred.
        percentile:    Percentile applied independently to each directional
                        surface-distance distribution.
        voxel_spacing: Physical voxel spacing in mm (d, h, w). Used to convert
                       voxel indices to metric distances.

    Returns:
        HD at the requested percentile in mm; float('inf') if either mask
        is empty (handles edge cases during early training).
    """
    pred_bin = (pred.squeeze() > 0.5).astype(bool)
    target_bin = (target.squeeze() > 0.5).astype(bool)

    if not pred_bin.any() or not target_bin.any():
        return float("inf")

    pred_surface = _surface_voxels(pred_bin)
    target_surface = _surface_voxels(target_bin)

    spacing = np.array(voxel_spacing)
    pred_pts = np.argwhere(pred_surface) * spacing
    target_pts = np.argwhere(target_surface) * spacing

    tree_pred = cKDTree(pred_pts)
    tree_target = cKDTree(target_pts)

    # Directed surface-to-surface distances, kept separate per direction.
    dist_p2t, _ = tree_target.query(pred_pts)
    dist_t2p, _ = tree_pred.query(target_pts)

    hd_p2t = np.percentile(dist_p2t, percentile)
    hd_t2p = np.percentile(dist_t2p, percentile)

    return float(max(hd_p2t, hd_t2p))
