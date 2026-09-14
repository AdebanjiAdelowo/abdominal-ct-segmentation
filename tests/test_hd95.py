"""
Regression tests for `src.utils.metrics.hausdorff_95`.

Context: an independent audit found the original implementation was wrong
in two ways:

  1. It computed nearest-neighbour distances over ALL foreground voxels
     instead of extracted SURFACE voxels.
  2. It pooled both directions (pred->gt and gt->pred) into one array and
     took a single percentile, instead of the standard convention: take
     the 95th percentile separately in each direction, then take the MAX
     of the two.

The audit's synthetic reproduction was two solid cubes of side 40 voxels,
offset along one axis by a known 3-voxel translation: the buggy function
returned 1.05 instead of the geometrically correct 3.0. This file
reproduces that exact case (`test_hd95_matches_audit_reproduction`) plus
a few supporting checks.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import hausdorff_95


def _buggy_hausdorff_95(pred: np.ndarray, target: np.ndarray, percentile: int = 95) -> float:
    """
    Re-creates the ORIGINAL (buggy) implementation exactly, for comparison
    purposes only: all-foreground voxels (no surface extraction), both
    directions pooled together before a single percentile.
    """
    pred_bin = (pred.squeeze() > 0.5).astype(bool)
    target_bin = (target.squeeze() > 0.5).astype(bool)

    pred_pts = np.argwhere(pred_bin)
    target_pts = np.argwhere(target_bin)

    tree_pred = cKDTree(pred_pts)
    tree_target = cKDTree(target_pts)

    dist_p2t, _ = tree_target.query(pred_pts)
    dist_t2p, _ = tree_pred.query(target_pts)

    all_distances = np.concatenate([dist_p2t, dist_t2p])
    return float(np.percentile(all_distances, percentile))


def _make_offset_cubes(side: int, shift: int) -> tuple:
    """Two identical solid cubes of edge `side`, offset by `shift` voxels along axis 0."""
    vol_shape = (side + shift + 2, side, side)
    pred = np.zeros(vol_shape, dtype=np.uint8)
    target = np.zeros(vol_shape, dtype=np.uint8)
    pred[0:side, 0:side, 0:side] = 1
    target[shift:side + shift, 0:side, 0:side] = 1
    return pred, target


def test_hd95_matches_audit_reproduction():
    """
    Exact reproduction of the audit's synthetic test: two 40-voxel-edge
    cubes offset by a known 3-voxel shift.

    Before the fix, `hausdorff_95` returned 1.05 for this input (verified
    below via `_buggy_hausdorff_95`, the archived original logic). The
    fixed function must return the geometrically correct 3.0.
    """
    pred, target = _make_offset_cubes(side=40, shift=3)

    buggy_value = _buggy_hausdorff_95(pred, target, percentile=95)
    assert buggy_value == pytest.approx(1.05, abs=1e-6), (
        "sanity check: the archived buggy logic should reproduce the "
        "audit's reported 1.05mm on this exact input"
    )

    fixed_value = hausdorff_95(pred, target, percentile=95)
    assert fixed_value == pytest.approx(3.0, abs=1e-6), (
        f"expected the corrected HD95 to equal the true 3-voxel offset, got {fixed_value}"
    )


def test_hd95_exact_on_axis_aligned_parallel_plates():
    """
    Two identical thin square plates on the same (x, y) footprint, offset
    along z by a known N-voxel gap. Every plate voxel is a surface voxel,
    and every voxel has a unique closest counterpart directly across the
    gap, so the correct HD95 is exactly N with no boundary-effect slack.
    """
    shape = (10, 10, 10)
    plate_a = np.zeros(shape, dtype=np.uint8)
    plate_b = np.zeros(shape, dtype=np.uint8)
    plate_a[2:8, 2:8, 0] = 1
    plate_b[2:8, 2:8, 3] = 1

    result = hausdorff_95(plate_a, plate_b, percentile=95)
    assert result == pytest.approx(3.0, abs=1e-9)


def test_hd95_respects_voxel_spacing():
    """Anisotropic spacing along the offset axis should scale the result linearly."""
    shape = (10, 10, 10)
    plate_a = np.zeros(shape, dtype=np.uint8)
    plate_b = np.zeros(shape, dtype=np.uint8)
    plate_a[2:8, 2:8, 0] = 1
    plate_b[2:8, 2:8, 3] = 1

    result = hausdorff_95(plate_a, plate_b, percentile=95, voxel_spacing=(1.0, 1.0, 2.0))
    assert result == pytest.approx(6.0, abs=1e-9)


def test_hd95_is_directed_max_not_pooled_average():
    """
    Isolates bug #2 (pooling both directions before one percentile) from
    bug #1 (missing surface extraction). Pred and target share an
    identical 6-voxel core, so the pred->target direction is ~entirely
    zero; target additionally has a thin 10-voxel spike absent from pred,
    which makes the target->pred 95th percentile 2.0 while contributing
    too small a fraction of the pooled, combined distribution to move a
    single pooled percentile off 0.0.

    A correct per-direction-then-max implementation must return 2.0
    (the worse direction); a pooled implementation returns 0.0 here even
    with correct surface extraction, which is exactly the failure mode
    the fix replaces.
    """
    shape = (20, 20, 20)
    pred = np.zeros(shape, dtype=np.uint8)
    target = np.zeros(shape, dtype=np.uint8)

    # Identical 6x6x6 core in both masks.
    pred[2:8, 2:8, 2:8] = 1
    target[2:8, 2:8, 2:8] = 1

    # A thin 10-voxel spike sticking out of target only, absent from pred.
    target[8:18, 4, 4] = 1

    directed_max = hausdorff_95(pred, target, percentile=95)
    assert directed_max == pytest.approx(2.0, abs=1e-9)


def test_hd95_empty_mask_returns_inf():
    shape = (8, 8, 8)
    empty = np.zeros(shape, dtype=np.uint8)
    non_empty = np.zeros(shape, dtype=np.uint8)
    non_empty[2:5, 2:5, 2:5] = 1

    assert hausdorff_95(empty, non_empty) == float("inf")
    assert hausdorff_95(non_empty, empty) == float("inf")


def test_hd95_identical_masks_is_zero():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[3:7, 3:7, 3:7] = 1

    assert hausdorff_95(mask, mask.copy()) == pytest.approx(0.0, abs=1e-9)
