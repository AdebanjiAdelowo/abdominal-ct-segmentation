"""
Tests for the leakage-controlled protocol: split integrity, NIfTI geometry,
physical-unit (mm) HD95 under anisotropic voxel spacing, and the evaluation
guards.

All data here is synthetic.  These tests verify the machinery (units, axis
order, leakage guards); they say nothing about the accuracy of any trained model.

Unit-mix-up strategy: every physical test uses three DIFFERENT spacings
(3.0, 0.7, 0.9 mm).  A voxel-space result, an ignored spacing, or a spacing
whose axes were not permuted together with the array all give a different
number than the physical one and therefore fail.
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits as sp
from src.data.nifti import list_cases, load_case, load_nifti
from src.inference import evaluate_protocol as ep
from src.inference.evaluate import evaluate_case
from src.models.unet3d import build_model
from src.utils.metrics import hausdorff_95

SPACING = (3.0, 0.7, 0.9)  # mm along file axes 0, 1, 2 (all different on purpose)
SHIFT = 3                  # voxels
SIDE = 52


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _box(shape, lo, size=40):
    m = np.zeros(shape, dtype=np.uint8)
    m[lo[0]:lo[0] + size, lo[1]:lo[1] + size, lo[2]:lo[2] + size] = 1
    return m


def _shifted_pair(axis, shape=(SIDE,) * 3):
    lo = [3, 3, 3]
    lo_shifted = list(lo)
    lo_shifted[axis] += SHIFT
    return _box(shape, lo), _box(shape, lo_shifted)


def _permuted_affine(spacing=SPACING):
    """File axis 0 -> world z, axis 1 -> world x, axis 2 -> world y (non-canonical on purpose)."""
    a = np.zeros((4, 4))
    a[2, 0] = spacing[0]
    a[0, 1] = spacing[1]
    a[1, 2] = spacing[2]
    a[3, 3] = 1
    return a


def _save_nifti(path, array, affine):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(path))


def _surface(m):
    return m.astype(bool) & ~binary_erosion(m.astype(bool), border_value=0)


def _oracle_hd95(a, b, spacing, q=95):
    """Independent implementation: Euclidean distance transform with physical sampling."""
    sa, sb = _surface(a), _surface(b)
    d_to_b = distance_transform_edt(~sb, sampling=spacing)
    d_to_a = distance_transform_edt(~sa, sampling=spacing)
    return max(np.percentile(d_to_b[sa], q), np.percentile(d_to_a[sb], q))


class _ThresholdStub(nn.Module):
    def forward(self, x):
        return x * 10.0


def _config(patch=32, epochs=2):
    return {
        "dataset": {"num_workers": 0, "pin_memory": False},
        "preprocessing": {"patch_size": [patch] * 3, "intensity_clip": [-200, 250], "normalize": True},
        "model": {"in_channels": 1, "out_channels": 1, "base_features": 4, "depth": 2, "residual": False, "dropout": 0.0},
        "training": {"batch_size": 2, "epochs": epochs, "lr": 1e-3, "weight_decay": 1e-5, "grad_clip": 1.0,
                     "dice_weight": 0.5, "bce_weight": 0.5, "scheduler": {"type": "cosine", "eta_min": 1e-6}},
        "validation": {"val_interval": 1, "hausdorff_percentile": 95},
        "inference": {"sw_batch_size": 2, "overlap": 0.25, "patch_size": [patch] * 3},
        "checkpoint": {"save_dir": "unused", "save_best": True, "monitor": "val_dice"},
        "logging": {"metrics_csv": "metrics.csv"},
    }


# --------------------------------------------------------------------------
# A. split integrity and the historical-validation exclusion rule
# --------------------------------------------------------------------------

IDS = [f"liver_{i}" for i in range(131)]


def _historical_val_independent(ids, seed=42, frac=0.2):
    """Independent re-implementation of the historical split (no code shared with src)."""
    perm = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    return set(perm[: max(1, int(len(ids) * frac))])


@pytest.fixture(scope="module")
def s131():
    return sp.make_splits(IDS, IDS, listing_source="test")


def test_sizes_partition_and_disjointness(s131):
    s = s131
    assert s["counts"] == {"train": 85, "val": 20, "test": 26}
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (85, 20, 26)
    assert set(s["train"]).isdisjoint(s["val"]) and set(s["train"]).isdisjoint(s["test"]) and set(s["val"]).isdisjoint(s["test"])
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(IDS)             # union = all 131
    assert sorted(s["train"] + s["val"] + s["test"]) == sorted(IDS)                  # no duplicates
    assert sp.validate_splits(s) == s["sha256"] and s["seed"] == 42


def test_new_test_set_is_disjoint_from_the_historical_validation_set(s131):
    hist = set(s131["historical_validation"]["case_ids"])
    assert len(hist) == 26
    assert hist == _historical_val_independent(IDS), "historical ids must match the historical split logic"
    assert set(s131["test"]) & hist == set()                                          # the required property
    assert set(s131["test"]) <= set(s131["test_eligible"]["case_ids"])
    assert s131["test_eligible"]["n"] == 105 and set(s131["test_eligible"]["case_ids"]) == set(IDS) - hist
    assert s131["historical_validation"]["n_in_new_test"] == 0
    # historical validation cases are allowed in the new train / validation sets, and all 26 are accounted for
    assert (s131["historical_validation"]["n_in_new_train"] + s131["historical_validation"]["n_in_new_val"]) == 26
    assert hist <= set(s131["train"]) | set(s131["val"])


def test_metadata_records_the_construction(s131):
    assert s131["exclusion_rule"]["id"] == "test_excludes_historical_validation" and "eligible_test" in s131["exclusion_rule"]["text"]
    assert "default_rng(42)" in s131["historical_validation"]["derivation"]
    assert s131["protocol_version"] == "v1" and len(s131["sha256"]) == 64


def test_historical_logic_is_shared_with_the_historical_loader(tmp_path):
    """split_case_names (historical .npy loader) and the protocol's historical ids come from the same function."""
    from src.data.dataset import historical_train_val_split, split_case_names
    (tmp_path / "image").mkdir()
    for i in range(131):
        np.save(tmp_path / "image" / f"liver_{i}_img.npy", np.zeros(1, dtype=np.float32))
    cfg = {"dataset": {"images_subdir": "image", "images_suffix": "_img", "val_split": 0.2}}
    _, hist_val = split_case_names(cfg, str(tmp_path))
    assert set(hist_val) == set(sp.historical_validation_cases(IDS)) == _historical_val_independent(IDS)
    assert historical_train_val_split(IDS, 0.2)[1] == hist_val


def test_split_is_deterministic_and_test_independent_of_val_fraction(s131):
    assert sp.make_splits(IDS, IDS, listing_source="test") == s131                     # regeneration is identical
    c = sp.make_splits(IDS, IDS, val_fraction=0.25, listing_source="test")
    assert c["test"] == s131["test"], "test membership must not depend on the validation fraction"
    assert set(c["test"]) & set(c["historical_validation"]["case_ids"]) == set()


def _first_non_historical(s, split):
    hist = set(s["historical_validation"]["case_ids"])
    return next(c for c in s[split] if c not in hist)


def test_changing_membership_or_construction_invalidates_the_fingerprint(s131):
    s = s131
    moved = _first_non_historical(s, "train")
    tampered = dict(s, train=[c for c in s["train"] if c != moved], test=s["test"] + [moved])
    with pytest.raises(ValueError, match="fingerprint|eligib"):
        sp.validate_splits(tampered)
    hist_edit = dict(s, historical_validation=dict(s["historical_validation"], case_ids=s["historical_validation"]["case_ids"][:-1]))
    with pytest.raises(ValueError):                                                    # construction edits are caught too
        sp.validate_splits(hist_edit)
    with pytest.raises(ValueError, match="fingerprint"):
        sp.validate_splits(dict(s, seed=43))


def test_a_historical_case_in_the_test_set_is_rejected_even_with_a_recomputed_fingerprint(s131):
    s = s131
    h = s["historical_validation"]["case_ids"][0]
    where = "train" if h in s["train"] else "val"
    swap = s["test"][0]
    leaky = dict(s)
    leaky[where] = [swap if c == h else c for c in s[where]]
    leaky["test"] = [h if c == swap else c for c in s["test"]]
    leaky["sha256"] = sp.fingerprint(leaky["train"], leaky["val"], leaky["test"], s["historical_validation"]["case_ids"], s["seed"])
    with pytest.raises(ValueError, match="historical validation cases"):
        sp.validate_splits(leaky)


def test_overlap_between_splits_is_rejected(s131):
    leaky = dict(s131, val=s131["val"] + [s131["test"][0]])
    with pytest.raises(ValueError, match="Leakage"):
        sp.validate_splits(leaky)


def test_write_once_and_training_view_cannot_see_test_names(tmp_path, s131):
    path = tmp_path / "splits.json"
    sp.write_splits(s131, str(path))
    with pytest.raises(FileExistsError):
        sp.write_splits(s131, str(path))
    train_view = sp.load_splits(str(path), include_test=False)
    assert "test" not in train_view and "test_eligible" not in train_view and train_view["sha256"] == s131["sha256"]
    serialised = json.dumps(train_view)
    assert not any(f'"{c}"' in serialised for c in s131["test"]), "no test case id may appear anywhere in the training view"
    assert sp.load_splits(str(path), include_test=True)["test"] == s131["test"]


def test_edited_split_file_is_refused(tmp_path, s131):
    path = tmp_path / "splits.json"
    sp.write_splits(s131, str(path))
    d = json.loads(path.read_text())
    d["train"][0], d["test"][0] = d["test"][0], d["train"][0]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        sp.load_splits(str(path))


def test_historical_ids_are_never_guessed():
    with pytest.raises(ValueError, match="do not follow"):
        sp.check_historical_naming([f"case_{i:03d}" for i in range(131)])
    with pytest.raises(ValueError, match="not the same listing"):
        sp.check_historical_naming([f"liver_{i}" for i in range(1, 132)])                # right pattern, different listing
    sp.check_historical_naming(IDS)                                                     # the recorded evidence is satisfied
    with pytest.raises(ValueError, match="not present"):                                # historical ids absent from the dataset
        sp.make_splits([f"liver_{i}" for i in range(10)], IDS)
    # an explicit historical listing is honoured, and its 26 ids are excluded from the test set
    other = [f"liver_{i}" for i in range(131)]
    explicit = sp.make_splits(IDS, other, listing_source="explicit")
    assert explicit["historical_validation"]["listing_source"] == "explicit"


# --------------------------------------------------------------------------
# B. NIfTI geometry
# --------------------------------------------------------------------------

def test_spacing_follows_the_array_axes_after_canonical_reorientation(tmp_path):
    data = np.random.default_rng(0).random((8, 9, 10)).astype(np.float32)
    _save_nifti(tmp_path / "v.nii.gz", data, _permuted_affine())
    arr, spacing, _ = load_nifti(str(tmp_path / "v.nii.gz"))
    # File axes (0,1,2) had spacing (3.0, 0.7, 0.9) and map to world (z, x, y).
    # Canonical RAS+ array axes are (x, y, z) -> spacing must be (0.7, 0.9, 3.0)
    # and the array must have been permuted the same way.
    assert spacing == pytest.approx((0.7, 0.9, 3.0))
    assert arr.shape == (9, 10, 8)
    np.testing.assert_allclose(arr, np.transpose(data, (1, 2, 0)))


def test_flipped_axes_keep_positive_spacing(tmp_path):
    aff = np.diag([-1.5, 0.8, -2.5, 1.0])
    _save_nifti(tmp_path / "v.nii.gz", np.zeros((6, 7, 8), np.float32), aff)
    _, spacing, _ = load_nifti(str(tmp_path / "v.nii.gz"))
    assert spacing == pytest.approx((1.5, 0.8, 2.5))


def test_sheared_grid_is_refused(tmp_path):
    aff = np.eye(4)
    aff[0, 1] = 0.8  # shear
    _save_nifti(tmp_path / "v.nii.gz", np.zeros((5, 5, 5), np.float32), aff)
    with pytest.raises(ValueError, match="Sheared"):
        load_nifti(str(tmp_path / "v.nii.gz"))


def test_image_label_geometry_mismatch_is_refused(tmp_path):
    _save_nifti(tmp_path / "i.nii.gz", np.zeros((6, 6, 6), np.float32), np.diag([1.0, 1.0, 1.0, 1.0]))
    _save_nifti(tmp_path / "l_shape.nii.gz", np.zeros((6, 6, 7), np.uint8), np.diag([1.0, 1.0, 1.0, 1.0]))
    _save_nifti(tmp_path / "l_space.nii.gz", np.zeros((6, 6, 6), np.uint8), np.diag([1.0, 1.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="Shape"):
        load_case(str(tmp_path / "i.nii.gz"), str(tmp_path / "l_shape.nii.gz"))
    with pytest.raises(ValueError, match="geometry"):
        load_case(str(tmp_path / "i.nii.gz"), str(tmp_path / "l_space.nii.gz"))


# --------------------------------------------------------------------------
# C. physical-unit HD95 (the mm-vs-voxel guard)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_hd95_is_physical_mm_for_a_shift_along_each_axis(axis):
    pred, gt = _shifted_pair(axis)
    expected_mm = SHIFT * SPACING[axis]            # 9.0, 2.1, 2.7
    got = hausdorff_95(pred, gt, voxel_spacing=SPACING)
    assert got == pytest.approx(expected_mm, abs=1e-9)
    # ...and each way of getting units wrong would produce a different number:
    assert hausdorff_95(pred, gt) == pytest.approx(SHIFT)             # voxel space
    assert got != pytest.approx(SHIFT, abs=0.1)                       # a voxel-space result would fail here
    wrong_order = (SPACING[1], SPACING[2], SPACING[0])                # axes not permuted with the data
    assert abs(hausdorff_95(pred, gt, voxel_spacing=wrong_order) - expected_mm) > 0.3


def test_hd95_matches_independent_edt_oracle_on_random_anisotropic_blobs():
    rng = np.random.default_rng(7)
    zz, yy, xx = np.ogrid[:36, :40, :44]
    for spacing in [(3.0, 0.7, 0.9), (0.5, 2.5, 1.3), (5.0, 1.0, 0.6)]:
        def blob():
            m = np.zeros((36, 40, 44), bool)
            for _ in range(4):
                c = rng.integers(8, 28, 3)
                r = rng.integers(4, 9, 3)
                m |= (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1
            return m.astype(np.uint8)
        a, b = blob(), blob()
        assert hausdorff_95(a, b, voxel_spacing=spacing) == pytest.approx(_oracle_hd95(a, b, spacing), rel=1e-9)


def test_evaluate_case_forwards_spacing():
    pred, gt = _shifted_pair(0)
    assert evaluate_case(pred, gt, voxel_spacing=SPACING)["hd95"] == pytest.approx(9.0)
    assert evaluate_case(pred, gt)["hd95"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# D. end to end through NIfTI I/O + sliding window (full volumes, mm)
# --------------------------------------------------------------------------

def _make_case(root, cid, axis):
    """Image = bright box; label = the same box shifted SHIFT voxels along FILE axis `axis`."""
    pred_box, gt_box = _shifted_pair(axis)
    image = np.where(pred_box > 0, 100.0, -100.0).astype(np.float32)
    _save_nifti(root / "imagesTr" / f"{cid}.nii.gz", image, _permuted_affine())
    _save_nifti(root / "labelsTr" / f"{cid}.nii.gz", gt_box, _permuted_affine())


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_full_volume_pipeline_reports_physical_mm_despite_reorientation(tmp_path, axis):
    _make_case(tmp_path, "liver_0", axis)
    cases = list_cases(str(tmp_path))
    rows = ep.evaluate_cases(_ThresholdStub(), _config(), cases, ["liver_0"], torch.device("cpu"), verbose=False)
    r = rows[0]
    assert r["hd95_mm"] == pytest.approx(SHIFT * SPACING[axis], abs=1e-6)
    assert r["dice"] == pytest.approx(2 * 37 * 40 * 40 / (2 * 40 ** 3), abs=1e-4)
    assert r["spacing_mm"] == "0.7x0.9x3"           # canonical (x, y, z) order, not file order
    # volumes in mL come from spacing: 64,000 voxels * 3.0*0.7*0.9 mm^3
    assert r["gt_volume_ml"] == pytest.approx(64000 * 3.0 * 0.7 * 0.9 / 1000, rel=1e-6)


def test_volume_smaller_than_patch_is_padded_and_cropped_back():
    vol = np.random.default_rng(1).normal(size=(20, 24, 18)).astype(np.float32)
    out = ep.predict_full_volume(_ThresholdStub(), vol, _config(patch=32), torch.device("cpu"))
    assert out.shape == vol.shape
    np.testing.assert_array_equal(out, (vol > 0).astype(np.uint8))


class _ZeroStub(nn.Module):
    """A model that always predicts 'no liver': a complete segmentation failure."""

    def forward(self, x):
        return torch.full_like(x, -10.0)


def test_summary_states_finite_case_hd95_together_with_failure_counts():
    rows = [{"case": c, "dice": d, "hd95_mm": h} for c, d, h in
            [("a", 0.9, 4.0), ("b", 0.95, 2.0), ("c", 0.85, 6.0), ("d", 0.0, float("inf"))]]
    s = ep.summarise_protocol(rows)
    h = s["hd95_mm"]
    assert (h["n_cases"], h["n_finite"], h["n_complete_failures"]) == (4, 3, 1)
    assert h["complete_failure_cases"] == ["d"] and h["all_cases_finite"] is False
    assert h["finite_cases_only"]["mean"] == pytest.approx(4.0)          # finite cases only, and labelled so
    assert s["dice_all_cases"]["mean"] == pytest.approx((0.9 + 0.95 + 0.85 + 0.0) / 4)  # Dice keeps the failure
    assert "COMPLETE FAILURES: 1 of 4" in s["report_text"] and "d" in s["report_text"]
    assert "3 of 4 finite cases only" in s["report_text"]
    assert s["dice_all_cases"]["mean_ci95"][0] <= s["dice_all_cases"]["mean"] <= s["dice_all_cases"]["mean_ci95"][1]
    assert "macro" in s["aggregation"] and s["hd95_unit"] == "mm"
    assert ep.summarise_protocol(rows)["dice_all_cases"]["mean_ci95"] == s["dice_all_cases"]["mean_ci95"]  # seeded


def test_summary_without_failures_says_so():
    rows = [{"case": "a", "dice": 0.9, "hd95_mm": 4.0}, {"case": "b", "dice": 0.8, "hd95_mm": 6.0}]
    s = ep.summarise_protocol(rows)
    assert s["hd95_mm"]["all_cases_finite"] and s["hd95_mm"]["n_complete_failures"] == 0
    assert "no complete failures" in s["report_text"] and "COMPLETE FAILURES" not in s["report_text"]


def test_empty_prediction_is_a_visible_complete_failure_end_to_end(tmp_path):
    _make_case(tmp_path, "liver_0", 0)
    _make_case(tmp_path, "liver_1", 1)
    cases = list_cases(str(tmp_path))
    rows = ep.evaluate_cases(_ZeroStub(), _config(), cases, ["liver_0", "liver_1"], torch.device("cpu"), verbose=False)
    assert all(r["status"] == "empty_prediction" and r["hd95_mm"] == float("inf") for r in rows)
    assert all(r["dice"] < 1e-3 for r in rows)
    s = ep.summarise_protocol(rows)
    assert s["hd95_mm"]["n_complete_failures"] == 2 and s["hd95_mm"]["n_finite"] == 0
    assert "COMPLETE FAILURES: 2 of 2" in s["report_text"]


def test_empty_reference_label_is_a_data_error_not_a_model_failure(tmp_path):
    _save_nifti(tmp_path / "imagesTr" / "liver_0.nii.gz", np.zeros((20, 20, 20), np.float32), _permuted_affine())
    _save_nifti(tmp_path / "labelsTr" / "liver_0.nii.gz", np.zeros((20, 20, 20), np.uint8), _permuted_affine())
    with pytest.raises(ValueError, match="reference label is empty"):
        ep.evaluate_cases(_ThresholdStub(), _config(), list_cases(str(tmp_path)), ["liver_0"], torch.device("cpu"), verbose=False)


# --------------------------------------------------------------------------
# E. leakage guards, end to end (tiny CPU training run on synthetic NIfTI)
# --------------------------------------------------------------------------

@pytest.fixture()
def trained_protocol(tmp_path, monkeypatch):
    from src.training import protocol_train as tp

    data = tmp_path / "Task03_Liver"
    for i in range(10):
        _make_case(data, f"liver_{i}", axis=i % 3)
    ids10 = [f"liver_{i}" for i in range(10)]
    splits = sp.make_splits(ids10, ids10)
    split_path = tmp_path / "splits.json"
    sp.write_splits(splits, str(split_path))

    over = {"preprocessing": {"patch_size": [16] * 3}, "training": {"epochs": 2},
            "model": {"base_features": 4, "depth": 2}, "inference": {"patch_size": [16] * 3, "sw_batch_size": 2},
            "dataset": {"num_workers": 0, "pin_memory": False}}
    trainer = tp.run_training(str(data), str(split_path), str(tmp_path / "run"), seed=0, overrides=over,
                              allow_dirty=True, device=torch.device("cpu"))
    train_view = sp.load_splits(str(split_path), include_test=False)
    cfg = trainer.config
    train_loader, val_loader = tp.build_loaders(cfg, list_cases(str(data)), train_view)
    return dict(tp=tp, data=data, split_path=split_path, splits=splits, cfg=cfg, ckpt=tmp_path / "run" / "checkpoints" / "best.pth",
                train_loader=train_loader, val_loader=val_loader, tmp=tmp_path)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(ep, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sys, "argv", ["evaluate_protocol", *argv])
    ep.main()


def test_training_never_sees_test_cases(trained_protocol):
    t = trained_protocol
    seen = set(t["train_loader"].dataset.names) | set(t["val_loader"].dataset.names)
    assert not seen & set(t["splits"]["test"])
    assert set(t["train_loader"].dataset.names) == set(t["splits"]["train"])
    with pytest.raises(AssertionError):
        t["tp"].build_loaders(t["cfg"], list_cases(str(t["data"])), t["splits"])  # dict that still has 'test'
    import re
    assert not re.search(r"liver_\d+", json.dumps(t["cfg"])), "no case identifiers should be recorded in the training config"
    assert t["cfg"]["protocol"]["n_train"] + t["cfg"]["protocol"]["n_val"] == len(t["splits"]["train"]) + len(t["splits"]["val"])


def test_checkpoint_carries_split_fingerprint_and_provenance(trained_protocol):
    ckpt = torch.load(trained_protocol["ckpt"], map_location="cpu")
    assert ckpt["split_sha256"] == ckpt["config"]["protocol"]["splits_sha256"] == trained_protocol["splits"]["sha256"]
    assert ckpt["config"]["protocol"]["selection_split"] == "val" and ckpt["seed"] == 0


def test_val_then_test_evaluation_writes_records_and_locks(trained_protocol, monkeypatch):
    t, out = trained_protocol, trained_protocol["tmp"] / "out"
    base = ["--checkpoint", str(t["ckpt"]), "--data-dir", str(t["data"]), "--splits", str(t["split_path"]), "--out-dir", str(out)]

    with pytest.raises(SystemExit, match="confirm-final-test"):
        _run(monkeypatch, *base, "--split", "test")
    with pytest.raises(SystemExit, match="validation split first"):     # no val evaluation of this checkpoint yet
        _run(monkeypatch, *base, "--split", "test", "--confirm-final-test")

    _run(monkeypatch, *base, "--split", "val")
    val = json.loads((out / "val_summary.json").read_text())
    assert val["n_cases"] == len(t["splits"]["val"]) and val["hd95_unit"] == "mm"
    assert val["splits_sha256"] == t["splits"]["sha256"] and "report_text" in val

    _run(monkeypatch, *base, "--split", "test", "--confirm-final-test")
    test_summary = json.loads((out / "test_summary.json").read_text())
    assert test_summary["n_cases"] == len(t["splits"]["test"]) and "SMOKE_TEST" not in test_summary
    rows = (out / "test_per_case.csv").read_text().strip().splitlines()
    assert rows[0].startswith("case,status,shape,spacing_mm,dice,hd95_mm") and len(rows) == 1 + len(t["splits"]["test"])
    lock = Path(str(t["split_path"]) + ".test_lock.json")
    assert lock.exists() and not (out / "test_evaluation_lock.json").exists()   # lock is beside the split file

    _run(monkeypatch, *base, "--split", "test", "--confirm-final-test")          # same checkpoint again: allowed, recorded
    assert len(json.loads(lock.read_text())["attempts"]) == 2

    # a different checkpoint may not be scored on the test set, not even from a FRESH output directory
    other = torch.load(t["ckpt"], map_location="cpu")
    other["epoch"] = 999
    other_path = t["tmp"] / "other.pth"
    torch.save(other, other_path)
    fresh = t["tmp"] / "fresh_out"
    argv = ["--checkpoint", str(other_path), "--data-dir", str(t["data"]), "--splits", str(t["split_path"]), "--out-dir", str(fresh)]
    _run(monkeypatch, *argv, "--split", "val")                                    # val evaluation is allowed for any checkpoint
    with pytest.raises(SystemExit, match="different checkpoint"):
        _run(monkeypatch, *argv, "--split", "test", "--confirm-final-test")


def test_legacy_or_mismatched_checkpoints_are_refused(trained_protocol, monkeypatch):
    t = trained_protocol
    good = torch.load(t["ckpt"], map_location="cpu")

    legacy = dict(good, config={k: v for k, v in good["config"].items() if k != "protocol"})
    torch.save(legacy, t["tmp"] / "legacy.pth")
    mismatch = json.loads(json.dumps(good["config"]))
    mismatch["protocol"]["splits_sha256"] = "0" * 64
    torch.save(dict(good, config=mismatch), t["tmp"] / "mismatch.pth")
    unselected = json.loads(json.dumps(good["config"]))
    unselected["protocol"]["selection_split"] = "test"
    torch.save(dict(good, config=unselected), t["tmp"] / "unselected.pth")
    incomplete = {k: v for k, v in good.items() if k not in ("git_commit", "timestamp_utc")}
    torch.save(incomplete, t["tmp"] / "incomplete.pth")

    for name, msg in [("legacy.pth", "no protocol metadata"), ("mismatch.pth", "different split"),
                      ("unselected.pth", "validation split"), ("incomplete.pth", "missing")]:
        with pytest.raises(SystemExit, match=msg):
            _run(monkeypatch, "--checkpoint", str(t["tmp"] / name), "--data-dir", str(t["data"]),
                 "--splits", str(t["split_path"]), "--split", "val", "--out-dir", str(t["tmp"] / "o2"))
