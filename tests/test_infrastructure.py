"""
Tests for split freezing, durable checkpoints, resume, the dataset validator,
reproducibility guards and the smoke test.  Synthetic data only; nothing here
measures model quality.
"""

import json
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
    out = tmp_path / "s.json"
    monkeypatch.setattr(sys, "argv", ["splits", "make", "--data-dir", "ignored", "--out", str(out)])
    sp.main()
    made = json.loads(out.read_text())
    assert made["counts"] == {"train": 85, "val": 20, "test": 26} and "MSD listing" in made["historical_validation"]["listing_source"]
    assert not set(made["test"]) & set(made["historical_validation"]["case_ids"])
    assert made["historical_validation"]["case_ids"] == sorted(sp.historical_validation_cases(ids))


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
