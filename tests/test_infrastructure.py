"""
Tests for split freezing, durable checkpoints, resume, the dataset validator,
reproducibility guards and the smoke test.  Synthetic data only; nothing here
measures model quality.
"""

import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import smoke_test as st
from src.data import splits as sp
from src.data.nifti import list_cases
from src.data import validate_dataset as vd
from src.data.validate_dataset import validate_dataset
from src.training import protocol_train as tp
from src.training.trainer import Trainer
from src.models.unet3d import build_model
from src.utils import provenance as pv

CPU = torch.device("cpu")


@pytest.fixture()
def synthetic(tmp_path):
    data = st.make_synthetic_dataset(tmp_path / "Task03_Liver", n_cases=10)
    ids = list(list_cases(str(data)))
    splits = sp.make_splits(ids, ids)
    path = tmp_path / "splits.json"
    sp.write_splits(splits, str(path))
    return dict(data=data, splits=splits, path=path, tmp=tmp_path)


def _train(s, out, epochs=4, **kw):
    over = json.loads(json.dumps(st.SYNTHETIC_OVERRIDES))
    over["training"]["epochs"] = epochs
    return tp.run_training(str(s["data"]), str(s["path"]), str(out), seed=0, overrides=over,
                           allow_dirty=True, device=CPU, **kw)


# --------------------------------------------------------------------------
# frozen split parameters
# --------------------------------------------------------------------------

def test_protocol_v1_is_seed_42_with_the_historical_exclusion_recorded():
    ids = [f"liver_{i}" for i in range(131)]
    assert sp.DEFAULT_SEED == 42 and sp.PROTOCOL_VERSION == "v1"
    s = sp.make_splits(ids, ids)
    assert s["seed"] == 42 and s["counts"] == {"train": 85, "val": 20, "test": 26}
    assert s["historical_validation"]["n_in_new_test"] == 0 and s["test_eligible"]["n"] == 105


def test_cli_has_no_seed_option_and_refuses_unexpected_dataset_size(synthetic, monkeypatch):
    out = synthetic["tmp"] / "new_splits.json"
    argv = ["splits", "make", "--data-dir", str(synthetic["data"]), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv + ["--seed", "7"])
    with pytest.raises(SystemExit) as e:
        sp.main()
    assert e.value.code == 2                                         # argparse: unrecognised argument
    monkeypatch.setattr(sys, "argv", argv)                           # default --expect-n 131 vs 10 cases
    with pytest.raises(SystemExit, match="expected 131"):
        sp.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", argv + ["--expect-n", "10"])
    with pytest.raises(ValueError, match="not the same listing"):    # synthetic names fail the historical evidence: no guessing
        sp.main()
    assert not out.exists()
    listing = synthetic["tmp"] / "historical_listing.txt"
    listing.write_text("\n".join(f"{c}_img" for c in sorted(list_cases(str(synthetic["data"])))))
    monkeypatch.setattr(sys, "argv", argv + ["--expect-n", "10", "--historical-listing", str(listing)])
    sp.main()
    made = json.loads(out.read_text())
    assert made["seed"] == 42 and "explicit file" in made["historical_validation"]["listing_source"]
    assert not set(made["test"]) & set(made["historical_validation"]["case_ids"])
    with pytest.raises(FileExistsError):                             # write-once: no regenerating candidate splits
        sp.main()


def test_cli_default_path_uses_the_msd_listing_only_after_the_evidence_check(tmp_path, monkeypatch):
    import src.data.nifti as nifti
    ids = [f"liver_{i}" for i in range(131)]
    monkeypatch.setattr(nifti, "list_cases", lambda d: {c: (None, None) for c in ids})
    import src.data.validate_dataset as vd
    gate_calls = []
    monkeypatch.setattr(vd, "check_content_before_split", lambda cases, report=None: gate_calls.append(sorted(cases)))
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["splits", "make", "--data-dir", "ignored", "--out", str(out)])
    sp.main()
    made = json.loads(out.read_text())
    assert made["counts"] == {"train": 85, "val": 20, "test": 26} and "MSD listing" in made["historical_validation"]["listing_source"]
    assert not set(made["test"]) & set(made["historical_validation"]["case_ids"])
    assert made["historical_validation"]["case_ids"] == sorted(sp.historical_validation_cases(ids))
    assert gate_calls == [sorted(ids)], "the duplicate-content gate must run before a split is written"


def test_describe_uses_geometry_only(synthetic):
    d = sp.describe_splits(list_cases(str(synthetic["data"])), sp.load_splits(str(synthetic["path"]), include_test=True))
    assert set(d) == {"train", "val", "test"} and d["train"]["n"] == 6
    assert d["train"]["slice_thickness_mm_median"] == pytest.approx(3.0)
    assert d["train"]["inplane_mm_median"] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# durable writes
# --------------------------------------------------------------------------

def test_atomic_save_never_leaves_a_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "best.pth"
    pv.atomic_torch_save({"v": 1}, target)
    assert torch.load(target)["v"] == 1

    def boom(obj, f):
        f.write(b"partial")
        raise RuntimeError("disk error mid-write")
    monkeypatch.setattr(pv.torch, "save", boom)
    with pytest.raises(RuntimeError):
        pv.atomic_torch_save({"v": 2}, target)
    monkeypatch.undo()
    assert torch.load(target)["v"] == 1, "existing checkpoint must survive a failed write"


def test_mirror_detects_a_corrupt_copy_and_does_not_publish_it(tmp_path, monkeypatch):
    src = tmp_path / "best.pth"
    src.write_bytes(b"x" * 5000)
    monkeypatch.setattr(pv.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"x" * 10))
    with pytest.raises(IOError, match="hash verification"):
        pv.mirror_file(src, tmp_path / "drive")
    assert not (tmp_path / "drive" / "best.pth").exists()
    monkeypatch.undo()
    assert pv.mirror_file(src, tmp_path / "drive").read_bytes() == src.read_bytes()


def test_trainer_checkpoints_are_complete_atomic_and_mirrored(synthetic):
    mirror = synthetic["tmp"] / "drive"
    trainer = _train(synthetic, synthetic["tmp"] / "run", epochs=2, mirror_dir=str(mirror))
    ck = synthetic["tmp"] / "run" / "checkpoints"
    assert not list(ck.glob("*.tmp")) and not list(mirror.glob("*.tmp"))
    for name in ("best.pth", "last.pth"):
        c = pv.load_verified_checkpoint(str(ck / name), CPU, expected_split_sha256=synthetic["splits"]["sha256"])
        assert c["seed"] == 0 and c["model_config"]["depth"] == 2 and c["timestamp_utc"] and "git_commit" in c
        assert c["optimiser_state"] and c["scheduler_state"] and c["epoch"] in (1, 2)
    man = json.loads((ck / "checkpoint_manifest.json").read_text())
    assert man["best"]["sha256"] == pv.sha256_file(ck / "best.pth") and man["split_sha256"] == synthetic["splits"]["sha256"]
    for name in ("best.pth", "last.pth", "checkpoint_manifest.json", "metrics.csv"):
        assert (mirror / name).exists()
    assert pv.sha256_file(mirror / "last.pth") == pv.sha256_file(ck / "last.pth")   # final epoch always mirrored
    assert (synthetic["tmp"] / "run" / "run_config.yaml").exists() and (synthetic["tmp"] / "run" / "splits.json").exists()
    assert trainer.best_dice >= 0


def test_resume_continues_the_schedule_and_appends_the_log(synthetic):
    run = synthetic["tmp"] / "run"
    first = _train(synthetic, run, epochs=4, stop_after_epoch=2)
    lr_after_2 = first.optimiser.param_groups[0]["lr"]
    second = _train(synthetic, run, epochs=4, resume_from=str(run / "checkpoints" / "last.pth"))
    assert second.start_epoch == 3
    rows = (run / "checkpoints" / "metrics.csv").read_text().strip().splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == ["1", "2", "3", "4"], "log must continue, not restart"
    assert second.best_dice >= first.best_dice
    assert second.optimiser.param_groups[0]["lr"] < lr_after_2          # cosine schedule kept decaying


def test_resume_refuses_a_checkpoint_from_another_split(synthetic):
    run = synthetic["tmp"] / "run"
    t = _train(synthetic, run, epochs=2)
    other = json.loads(json.dumps(t.config))
    other["protocol"]["splits_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Refusing to resume"):
        Trainer(build_model(other), other, CPU, str(synthetic["tmp"] / "x"), resume_from=str(run / "checkpoints" / "last.pth"))


# --------------------------------------------------------------------------
# reproducibility guards
# --------------------------------------------------------------------------

def test_training_refuses_an_uncommitted_split_or_dirty_tree(tmp_path, monkeypatch):
    stray = tmp_path / "splits.json"
    stray.write_text("{}")
    with pytest.raises(SystemExit, match="not committed"):
        tp.check_reproducibility_state(str(stray), allow_dirty=False)
    tp.check_reproducibility_state(str(stray), allow_dirty=True)     # explicit experiment escape hatch

    monkeypatch.setattr(tp, "is_tracked_and_clean", lambda p: True)
    monkeypatch.setattr(tp, "git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(SystemExit, match="uncommitted changes"):
        tp.check_reproducibility_state(str(stray), allow_dirty=False)
    monkeypatch.setattr(tp, "git_state", lambda: {"commit": "abc", "dirty": False})
    assert tp.check_reproducibility_state(str(stray), allow_dirty=False)["commit"] == "abc"


def test_is_tracked_and_clean_on_real_repo_files(tmp_path):
    assert pv.is_tracked_and_clean(Path(__file__).resolve().parents[1] / "LICENSE")
    assert not pv.is_tracked_and_clean(tmp_path / "nope.json")


# --------------------------------------------------------------------------
# dataset validator
# --------------------------------------------------------------------------

def test_validator_accepts_a_good_dataset_and_reports_each_defect(tmp_path):
    root = st.make_synthetic_dataset(tmp_path / "ds", n_cases=5)
    good = validate_dataset(str(root), expect_cases=5)
    assert good["errors"] == [] and good["n_cases"] == 5

    assert any("expected 6" in e for e in validate_dataset(str(root), expect_cases=6)["errors"])

    (root / "labelsTr" / "liver_0.nii.gz").unlink()
    assert any("Image without label: liver_0" in e for e in validate_dataset(str(root), 5)["errors"])

    bad = np.zeros((40, 36, 32), np.uint8); bad[5:9, 5:9, 5:9] = 7
    nib.save(nib.Nifti1Image(bad, np.diag([1.0, 1.0, 1.0, 1.0])), str(root / "labelsTr" / "liver_0.nii.gz"))
    errs = validate_dataset(str(root), 5)["errors"]
    assert any("unexpected label values" in e for e in errs) and any("affines differ" in e for e in errs)

    sheared = np.eye(4); sheared[0, 1] = 0.9
    nib.save(nib.Nifti1Image(np.zeros((40, 36, 32), np.float32), sheared), str(root / "imagesTr" / "liver_1.nii.gz"))
    assert any("liver_1" in e and "Sheared" in e for e in validate_dataset(str(root), 5)["errors"])


# --------------------------------------------------------------------------
# smoke test
# --------------------------------------------------------------------------

def test_smoke_test_runs_end_to_end_and_is_stamped_as_not_a_result(synthetic):
    out = synthetic["tmp"] / "smoke"
    report = st.run_smoke_test(str(synthetic["data"]), str(synthetic["path"]), str(out), max_train=4, max_val=2, epochs=2,
                               overrides=st.SYNTHETIC_OVERRIDES, mirror_dir=str(synthetic["tmp"] / "drive"),
                               allow_dirty=True, device=CPU)
    assert set(report["steps"]) >= {"splits", "nifti_geometry", "training", "checkpoint_reload", "checkpoint_manifest",
                                    "checkpoint_mirror", "resume", "full_volume_eval_and_serialisation"}
    assert (out / "NOT_A_RESULT.txt").exists() and "NOT a model result" in json.loads((out / "eval" / "val_summary.json").read_text())["SMOKE_TEST"]
    splits = sp.load_splits(str(synthetic["path"]), include_test=True)
    assert not set(report["cases_used"]) & set(splits["test"]), "smoke test must never touch test cases"
    # the spacing seen through the whole chain is the physical spacing in canonical (x, y, z) order
    assert all(v == pytest.approx([0.7, 0.9, 3.0]) for v in report["spacing_mm"].values())


# --------------------------------------------------------------------------
# duplicate-content validation (synthetic data only)
# --------------------------------------------------------------------------

@pytest.fixture()
def ds(tmp_path):
    return st.make_synthetic_dataset(tmp_path / "ds", n_cases=6)


def _dup_image(root, src, dst):
    shutil.copy(root / "imagesTr" / f"{src}.nii.gz", root / "imagesTr" / f"{dst}.nii.gz")


def _dup_label(root, src, dst):
    shutil.copy(root / "labelsTr" / f"{src}.nii.gz", root / "labelsTr" / f"{dst}.nii.gz")


def test_distinct_volumes_are_not_flagged(ds):
    r = validate_dataset(str(ds), expect_cases=6)
    assert r["errors"] == [] and r["duplicates"] == [] and r["content_check"] is True
    assert len(set(r["hashes"]["image"].values())) == 6 and len(set(r["hashes"]["label"].values())) == 6


def test_duplicated_image_is_detected_and_fails_validation(ds):
    _dup_image(ds, "liver_1", "liver_4")                                   # same content, different case id
    r = validate_dataset(str(ds), expect_cases=6)
    groups = [g for g in r["duplicates"] if g["category"] == "image"]
    assert len(groups) == 1 and groups[0]["cases"] == ["liver_1", "liver_4"] and len(groups[0]["sha256"]) == 64
    assert not [g for g in r["duplicates"] if g["category"] == "label"]
    msg = next(e for e in r["errors"] if e.startswith("Duplicate image content"))
    assert "liver_1" in msg and "liver_4" in msg and groups[0]["sha256"][:16] in msg


def test_three_way_duplicate_is_reported_as_one_group(ds):
    _dup_image(ds, "liver_0", "liver_2")
    _dup_image(ds, "liver_0", "liver_3")
    groups = [g for g in validate_dataset(str(ds), 6)["duplicates"] if g["category"] == "image"]
    assert len(groups) == 1 and groups[0]["cases"] == ["liver_0", "liver_2", "liver_3"]


def test_duplicated_label_is_detected_and_fails_validation(ds):
    _dup_label(ds, "liver_1", "liver_4")                                   # different image, identical mask
    r = validate_dataset(str(ds), expect_cases=6)
    groups = [g for g in r["duplicates"] if g["category"] == "label"]
    assert len(groups) == 1 and groups[0]["cases"] == ["liver_1", "liver_4"]
    assert not [g for g in r["duplicates"] if g["category"] == "image"]
    assert any(e.startswith("Duplicate label content") for e in r["errors"])


def test_header_only_differences_do_not_hide_a_duplicate(tmp_path):
    arr = np.random.default_rng(3).normal(size=(12, 13, 14)).astype(np.float32)
    a, b, c = tmp_path / "a.nii.gz", tmp_path / "b.nii", tmp_path / "c.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.diag([1.0, 1.0, 1.0, 1.0])), str(a))
    nib.save(nib.Nifti1Image(arr.astype(np.float64), np.diag([0.7, 0.9, 3.0, 1.0])), str(b))      # other affine, dtype, no gzip
    changed = arr.copy(); changed[5, 5, 5] += 1.0
    nib.save(nib.Nifti1Image(changed, np.diag([1.0, 1.0, 1.0, 1.0])), str(c))
    assert vd.image_content_hash(a) == vd.image_content_hash(b)             # header/dtype/compression ignored
    assert vd.image_content_hash(a) != vd.image_content_hash(c)             # one voxel differs -> not a duplicate


def test_flipped_copy_is_not_claimed_as_an_exact_duplicate(tmp_path):
    arr = np.random.default_rng(4).normal(size=(10, 11, 12)).astype(np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tmp_path / "a.nii.gz"))
    nib.save(nib.Nifti1Image(arr[::-1].copy(), np.eye(4)), str(tmp_path / "flip.nii.gz"))
    assert vd.image_content_hash(tmp_path / "a.nii.gz") != vd.image_content_hash(tmp_path / "flip.nii.gz")


def test_no_content_check_skips_only_the_content_checks(ds, tmp_path):
    _dup_image(ds, "liver_1", "liver_4")
    r = validate_dataset(str(ds), expect_cases=6, content_check=False)
    assert r["content_check"] is False and r["duplicates"] == [] and not any("Duplicate" in e for e in r["errors"])
    (ds / "labelsTr" / "liver_0.nii.gz").unlink()                          # structural checks still run
    assert any("Image without label" in e for e in validate_dataset(str(ds), 6, content_check=False)["errors"])
    report = tmp_path / "skipped.json"
    report.write_text(json.dumps(validate_dataset(str(ds), 6, content_check=False)))
    with pytest.raises(SystemExit, match="no-content-check"):              # a skipped-check report cannot clear a dataset
        vd.check_content_before_split(list_cases(str(st.make_synthetic_dataset(tmp_path / "other", 6))), str(report))


def test_validator_cli_exit_codes_and_report(ds, tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["v", "--data-dir", str(ds), "--expect-cases", "6", "--report", str(report)])
    with pytest.raises(SystemExit) as ok:
        vd.main()
    assert ok.value.code == 0
    rep = json.loads(report.read_text())
    assert rep["content_check"] and rep["duplicates"] == [] and set(rep["hashes"]) == {"image", "label"} and "float32" in rep["hash_method"]
    _dup_image(ds, "liver_1", "liver_4")
    with pytest.raises(SystemExit) as bad:
        vd.main()
    assert bad.value.code == 1
    assert json.loads(report.read_text())["duplicates"][0]["cases"] == ["liver_1", "liver_4"]


def test_split_generation_refuses_duplicates_and_writes_nothing(ds, tmp_path, monkeypatch):
    listing = tmp_path / "hist.txt"
    listing.write_text("\n".join(sorted(list_cases(str(ds)))))
    out = tmp_path / "splits.json"
    argv = ["splits", "make", "--data-dir", str(ds), "--out", str(out), "--expect-n", "6", "--historical-listing", str(listing)]
    _dup_image(ds, "liver_2", "liver_5")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="duplicate dataset content"):
        sp.main()
    assert not out.exists()
    (ds / "imagesTr" / "liver_5.nii.gz").unlink()                          # restore a distinct volume for liver_5
    st_arr = np.random.default_rng(99).normal(size=(40, 36, 32)).astype(np.float32)
    nib.save(nib.Nifti1Image(st_arr, nib.load(str(ds / "imagesTr" / "liver_0.nii.gz")).affine), str(ds / "imagesTr" / "liver_5.nii.gz"))
    sp.main()                                                              # clean dataset: split written
    assert out.exists()


def test_split_generation_accepts_a_matching_passing_report_without_rehashing(ds, tmp_path, monkeypatch):
    report = tmp_path / "ok.json"
    report.write_text(json.dumps(validate_dataset(str(ds), 6)))
    cases = list_cases(str(ds))
    monkeypatch.setattr(vd, "content_hashes", lambda c: (_ for _ in ()).throw(AssertionError("must not rehash")))
    vd.check_content_before_split(cases, str(report))                      # passes, no hashing
    partial = {k: v for k, v in list(cases.items())[:-1]}
    with pytest.raises(SystemExit, match="exactly the current case listing"):
        vd.check_content_before_split(partial, str(report))
    bad = json.loads(report.read_text()); bad["duplicates"] = [{"category": "image", "sha256": "0" * 64, "cases": ["a", "b"]}]
    report.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="duplicates"):
        vd.check_content_before_split(cases, str(report))
