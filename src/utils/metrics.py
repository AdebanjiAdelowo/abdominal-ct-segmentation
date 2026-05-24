"""
Segmentation quality metrics used during validation.

Dice similarity coefficient and percentile Hausdorff distance are the two
primary metrics reported in medical image segmentation benchmarks.
"""

from typing import Tuple

import numpy as np
import torch
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


def hausdorff_95(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: int = 95,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Percentile Hausdorff distance between two binary segmentation masks.

    Uses a bidirectional nearest-neighbour surface-distance approach via
    scipy's cKDTree, which scales well for large point clouds.

    Args:
        pred:          Binary prediction, shape (..., D, H, W), values in {0, 1}.
        target:        Binary ground-truth, same shape as pred.
        percentile:    Percentile of the symmetric surface-distance distribution.
        voxel_spacing: Physical voxel spacing in mm (d, h, w). Used to convert
                       voxel indices to metric distances.

    Returns:
        HD at the requested percentile in mm; float('inf') if either mask
        is empty (handles edge cases during early training).
    """
    pred_bin = (pred.squeeze() > 0.5).astype(bool)
    target_bin = (target.squeeze() > 0.5).astype(bool)

    pred_pts = np.argwhere(pred_bin) * np.array(voxel_spacing)
    target_pts = np.argwhere(target_bin) * np.array(voxel_spacing)

    if pred_pts.size == 0 or target_pts.size == 0:
        return float("inf")

    tree_pred = cKDTree(pred_pts)
    tree_target = cKDTree(target_pts)

    # Surface-to-surface distances in both directions
    dist_p2t, _ = tree_target.query(pred_pts)
    dist_t2p, _ = tree_pred.query(target_pts)

    all_distances = np.concatenate([dist_p2t, dist_t2p])
    return float(np.percentile(all_distances, percentile))
