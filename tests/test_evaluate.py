"""
Tests for the full-volume evaluator and the shared train/val split.

These exercise the plumbing on tiny synthetic volumes (stub predictor, random
weights).  They say nothing about the accuracy of the trained model.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import split_case_names
from src.inference.evaluate import evaluate_case, evaluate_split, main, summarise
from src.models.unet3d import build_model

N_CASES = 6
SIDE = 24


def _config(data_dir: str) -> dict:
    return {
        "dataset": {
            "data_dir": data_dir,
            "images_subdir": "image",
            "labels_subdir": "liverMask",
            "images_suffix": "_img",
            "labels_suffix": "_liverMask",
            "val_split": 0.34,
        },
        "preprocessing": {"patch_size": [16, 16, 16], "intensity_clip": [-200, 250], "normalize": True},
        "model": {"in_channels": 1, "out_channels": 1, "base_features": 4, "depth": 2, "residual": False, "dropout": 0.0},
        "validation": {"hausdorff_percentile": 95},
        "inference": {"sw_batch_size": 2, "overlap": 0.25, "patch_size": [16, 16, 16]},
    }


@pytest.fixture()
def synthetic_dataset(tmp_path):
    """Kaggle-layout dataset: bright sphere (label) on a dark background."""
    (tmp_path / "image").mkdir()
    (tmp_path / "liverMask").mkdir()
    zz, yy, xx = np.ogrid[:SIDE, :SIDE, :SIDE]
    for i in range(N_CASES):
        c = SIDE // 2 + (i % 3) - 1
        sphere = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= 7 ** 2
        image = np.where(sphere, 100.0, -100.0).astype(np.float32)
        np.save(tmp_path / "image" / f"liver_{i:03d}_img.npy", image)
        np.save(tmp_path / "liverMask" / f"liver_{i:03d}_liverMask.npy", sphere.astype(np.uint8))
    return str(tmp_path)


class _ThresholdStub(nn.Module):
    """Predicts foreground wherever the z-scored input is positive."""

    def forward(self, x):
        return x * 10.0


def test_split_is_deterministic_and_disjoint(synthetic_dataset):
    cfg = _config(synthetic_dataset)
    train_a, val_a = split_case_names(cfg, synthetic_dataset)
    train_b, val_b = split_case_names(cfg, synthetic_dataset)
    assert (train_a, val_a) == (train_b, val_b)
    assert not set(train_a) & set(val_a)
    assert sorted(train_a + val_a) == [f"liver_{i:03d}" for i in range(N_CASES)]
    assert len(val_a) == 2


def test_split_matches_msd_task03_sizes(tmp_path):
    """131 labelled MSD volumes at val_split=0.2 must give the reported 105 / 26."""
    (tmp_path / "image").mkdir()
    for i in range(131):
        np.save(tmp_path / "image" / f"liver_{i}_img.npy", np.zeros(1, dtype=np.float32))
    train, val = split_case_names(_config(str(tmp_path)) | {"dataset": {**_config("")["dataset"], "val_split": 0.2}}, str(tmp_path))
    assert (len(train), len(val)) == (105, 26)


def test_evaluate_case_matches_known_geometry():
    a = np.zeros((46, 40, 40), dtype=np.uint8)
    b = np.zeros_like(a)
    a[0:40, :, :] = 1
    b[3:43, :, :] = 1
    scores = evaluate_case(a, b)
    assert scores["hd95"] == pytest.approx(3.0)
    assert scores["dice"] == pytest.approx(2 * 37 / 80, abs=1e-3)
    same = evaluate_case(a, a)
    assert same["hd95"] == 0.0 and same["dice"] == pytest.approx(1.0, abs=1e-4)


def test_evaluate_case_spacing_scales_hd95():
    a = np.zeros((46, 40, 40), dtype=np.uint8)
    b = np.zeros_like(a)
    a[0:40] = 1
    b[3:43] = 1
    assert evaluate_case(a, b, voxel_spacing=(2.0, 1.0, 1.0))["hd95"] == pytest.approx(6.0)


def test_summarise_counts_infinite_hd95_separately():
    rows = [{"dice": 0.9, "hd95": 2.0}, {"dice": 0.8, "hd95": 4.0}, {"dice": 0.0, "hd95": float("inf")}]
    s = summarise(rows)
    assert s["hd95_n_infinite"] == 1
    assert s["hd95_mean"] == pytest.approx(3.0)
    assert s["n_cases"] == 3


def test_evaluate_split_full_volume_pipeline(synthetic_dataset):
    cfg = _config(synthetic_dataset)
    rows = evaluate_split(_ThresholdStub(), cfg, synthetic_dataset, torch.device("cpu"))
    _, val = split_case_names(cfg, synthetic_dataset)
    assert [r["case"] for r in rows] == val
    assert all(r["shape"] == f"{SIDE}x{SIDE}x{SIDE}" for r in rows)
    assert all(r["dice"] > 0.99 and r["hd95"] <= 1.0 for r in rows)


def test_cli_writes_csv_and_summary_with_units(synthetic_dataset, tmp_path, monkeypatch):
    cfg = _config(synthetic_dataset)
    model = build_model(cfg)
    ckpt = tmp_path / "best.pth"
    torch.save({"epoch": 7, "model_state": model.state_dict(), "val_dice": 0.5, "config": cfg}, ckpt)
    out = tmp_path / "out" / "metrics.csv"

    monkeypatch.setattr(
        sys, "argv",
        ["evaluate", "--checkpoint", str(ckpt), "--data-dir", synthetic_dataset, "--out", str(out)],
    )
    main()

    lines = out.read_text().strip().splitlines()
    assert lines[0] == "case,shape,dice,hd95"
    assert len(lines) == 1 + 2
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["hd95_unit"] == "voxels"
    assert summary["checkpoint_epoch"] == 7
    assert summary["n_cases"] == 2
