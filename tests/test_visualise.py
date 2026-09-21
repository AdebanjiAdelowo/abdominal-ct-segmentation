"""
Tests for the neutral labelling in ``visualise.py``.

The historical ``.npy`` volumes carry no orientation metadata, so the module must
not print anatomical plane names unless a caller passes verified ones.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import visualise as vis

FORBIDDEN = re.compile(r"axial|coronal|sagittal", re.I)


def test_default_titles_name_only_axis_and_index():
    titles = vis.view_titles((512, 512, 74))
    assert titles == ["Slice through axis 0 (index 256 of 512)",
                      "Slice through axis 1 (index 256 of 512)",
                      "Slice through axis 2 (index 37 of 74)"]
    assert not any(FORBIDDEN.search(t) for t in titles)


def test_verified_plane_names_are_used_verbatim_and_validated():
    titles = vis.view_titles((20, 30, 40), plane_names=["A", "B", "C"])
    assert titles[1].startswith("B: ") and "axis 1" in titles[1]
    with pytest.raises(ValueError, match="exactly three"):
        vis.view_titles((20, 30, 40), plane_names=["A", "B"])


def test_module_source_makes_no_anatomical_plane_claims():
    assert not FORBIDDEN.search(Path(vis.__file__).read_text()), "visualise.py must not name anatomical planes"


def test_figure_is_written_and_slices_are_the_documented_axes(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(16, 18, 20)).astype(np.float32)
    pred = (vol > 0).astype(np.uint8)
    captured = {}
    real = vis._render_slice

    def spy(ax, ct, mask, title):
        captured.setdefault("shapes", []).append(ct.shape)
        captured.setdefault("titles", []).append(title)
        return real(ax, ct, mask, title)

    monkeypatch.setattr(vis, "_render_slice", spy)
    out = tmp_path / "o" / "fig.png"
    vis.save_segmentation_figure(vol, pred, str(out), gt_mask=pred, case_name="case")
    assert out.exists() and out.stat().st_size > 1000
    # panel k is the slice through array axis k: shapes (18,20), (16,20), (16,18); two rows (pred, GT)
    assert captured["shapes"] == [(18, 20), (16, 20), (16, 18)] * 2
    assert all(t.startswith(("Pred: ", "GT: ")) for t in captured["titles"])
    assert not any(FORBIDDEN.search(t) for t in captured["titles"])
