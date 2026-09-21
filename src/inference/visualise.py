"""
Segmentation visualisation: three orthogonal mid-slices with overlay.

Panels are labelled by ARRAY AXIS and slice index only.  This module does not
name anatomical planes: the historical ``.npy`` volumes carry no orientation
metadata, so their axis order cannot be tied to anatomy, and the plane names
this module used to print were unsupported.  A caller that has derived the
orientation from reliable geometry (for example NIfTI affines) may pass
``plane_names`` explicitly; nothing is inferred here.

Saves PNG figures suitable for papers and reports.  All rendering uses the
non-interactive Agg backend so the script runs in headless Kaggle notebooks
and SSH sessions without a display.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def view_titles(
    shape: Tuple[int, int, int],
    plane_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Titles for the three mid-slice panels, one per array axis.

    Panel ``k`` is the slice through array axis ``k`` at the middle index.  By
    default the title states only the axis and index; ``plane_names`` (three
    strings) is used verbatim if the caller has verified the orientation.
    """
    if plane_names is not None and len(plane_names) != 3:
        raise ValueError("plane_names must contain exactly three names")
    titles = []
    for axis, size in enumerate(shape):
        base = f"Slice through axis {axis} (index {size // 2} of {size})"
        titles.append(f"{plane_names[axis]}: {base}" if plane_names else base)
    return titles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_segmentation_figure(
    volume: np.ndarray,
    pred_mask: np.ndarray,
    out_path: str,
    gt_mask: Optional[np.ndarray] = None,
    case_name: str = "",
    plane_names: Optional[Sequence[str]] = None,
) -> None:
    """
    Save a figure of the three orthogonal mid-slices with overlay.

    Panel ``k`` is the slice through array axis ``k`` at its middle index (see
    :func:`view_titles`).  If ``gt_mask`` is provided a second row is added for
    side-by-side comparison between the prediction and the ground-truth.

    Args:
        volume:      CT volume, 3-D, array axis order as stored.  Pre-processed
                     float values.  No orientation is assumed.
        pred_mask:   Binary prediction mask, same shape as ``volume``.
        out_path:    Full path to the output PNG file (parent dirs created).
        gt_mask:     Optional ground-truth binary mask, same shape.
        case_name:   Case identifier shown in the figure suptitle.
        plane_names: Optional three names for the panels, used verbatim.  Pass
                     only names derived from verified orientation metadata.
    """
    shape = volume.shape
    i0, i1, i2 = shape[0] // 2, shape[1] // 2, shape[2] // 2
    titles = view_titles(shape, plane_names)

    def mid_slices(a: np.ndarray):
        return [a[i0, :, :], a[:, i1, :], a[:, :, i2]]

    slices_vol = mid_slices(volume)
    slices_pred = mid_slices(pred_mask)
    slices_gt = mid_slices(gt_mask) if gt_mask is not None else None

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
    for col in range(3):
        _render_slice(axes[0, col], slices_vol[col], slices_pred[col], f"Pred: {titles[col]}")

    # Row 1 — ground-truth (if provided)
    if slices_gt is not None:
        for col in range(3):
            _render_slice(axes[1, col], slices_vol[col], slices_gt[col], f"GT: {titles[col]}")

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
