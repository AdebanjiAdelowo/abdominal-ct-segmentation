"""
Segmentation visualisation: axial / coronal / sagittal slices with overlay.

Saves PNG figures suitable for papers and reports.  All rendering uses the
non-interactive Agg backend so the script runs in headless Kaggle notebooks
and SSH sessions without a display.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless — must be set before importing pyplot

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _render_slice(
    ax: plt.Axes,
    ct_slice: np.ndarray,
    mask_slice: Optional[np.ndarray],
    title: str,
) -> None:
    """
    Draw a single 2-D CT slice with a semi-transparent segmentation overlay.

    The CT intensities are min-max normalised to [0, 1] for display only;
    the underlying array values are not modified.

    Args:
        ax:         Matplotlib Axes to draw on.
        ct_slice:   2-D float array, shape (H, W).
        mask_slice: 2-D binary array, shape (H, W), or None.
        title:      Axes title string.
    """
    s = ct_slice.astype(float)
    lo, hi = s.min(), s.max()
    s = (s - lo) / (hi - lo + 1e-8)

    ax.imshow(s, cmap="gray", interpolation="none", origin="upper")

    if mask_slice is not None and mask_slice.any():
        # RGBA overlay: semi-transparent red for foreground (liver)
        rgba = np.zeros((*mask_slice.shape, 4), dtype=float)
        rgba[mask_slice > 0] = [1.0, 0.18, 0.18, 0.45]
        ax.imshow(rgba, interpolation="none", origin="upper")

    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_segmentation_figure(
    volume: np.ndarray,
    pred_mask: np.ndarray,
    out_path: str,
    gt_mask: Optional[np.ndarray] = None,
    case_name: str = "",
) -> None:
    """
    Save a figure of axial, coronal, and sagittal mid-slices with overlay.

    If ``gt_mask`` is provided a second row is added for side-by-side
    comparison between the prediction and the ground-truth.

    Args:
        volume:    CT volume, shape (D, H, W).  Pre-processed float values.
        pred_mask: Binary prediction mask, shape (D, H, W).
        out_path:  Full path to the output PNG file (parent dirs created).
        gt_mask:   Optional ground-truth binary mask, shape (D, H, W).
        case_name: Case identifier shown in the figure suptitle.
    """
    D, H, W = volume.shape
    d, h, w = D // 2, H // 2, W // 2

    # Extract mid-plane slices for each anatomical view
    slices_vol = {
        "Axial":    volume[d, :, :],
        "Coronal":  volume[:, h, :],
        "Sagittal": volume[:, :, w],
    }
    slices_pred = {
        "Axial":    pred_mask[d, :, :],
        "Coronal":  pred_mask[:, h, :],
        "Sagittal": pred_mask[:, :, w],
    }
    slices_gt = (
        {
            "Axial":    gt_mask[d, :, :],
            "Coronal":  gt_mask[:, h, :],
            "Sagittal": gt_mask[:, :, w],
        }
        if gt_mask is not None
        else None
    )

    n_rows = 2 if gt_mask is not None else 1
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 4.2 * n_rows),
        dpi=150,
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # unify indexing

    # Row 0 — prediction
    for col, view in enumerate(["Axial", "Coronal", "Sagittal"]):
        subtitle = f"{view} ({'z' if view == 'Axial' else 'y' if view == 'Coronal' else 'x'}={[d,h,w][col]})"
        _render_slice(axes[0, col], slices_vol[view], slices_pred[view], f"Pred — {subtitle}")

    # Row 1 — ground-truth (if provided)
    if slices_gt is not None:
        for col, view in enumerate(["Axial", "Coronal", "Sagittal"]):
            _render_slice(axes[1, col], slices_vol[view], slices_gt[view], f"GT — {view}")

    # Shared legend
    liver_patch = mpatches.Patch(facecolor=(1.0, 0.18, 0.18, 0.45), label="Liver")
    fig.legend(
        handles=[liver_patch],
        loc="lower right",
        fontsize=8,
        framealpha=0.8,
    )

    suptitle = f"Liver Segmentation — {case_name}" if case_name else "Liver Segmentation"
    fig.suptitle(suptitle, fontsize=10, y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[visualise] Saved: {out_path}")
