"""
3D CT volume dataset for liver segmentation.

Loads (image, mask) pairs from .npy files, applies intensity pre-processing,
and extracts 3-D patches — random (foreground-biased) during training and
centre-crop during validation / inference.

Expected directory layout (configurable via configs/config.yaml):
    <data_dir>/
    ├── imagesTr/
    │   ├── liver_001.npy   # float32 array, shape (D, H, W), Hounsfield units
    │   └── ...
    └── labelsTr/
        ├── liver_001.npy   # int array, shape (D, H, W), values: 0=bg, 1=liver [, 2=tumour]
        └── ...
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LiverCTDataset(Dataset):
    """
    PyTorch Dataset for MSD Task03 liver segmentation.

    Each ``__getitem__`` call:
        1. Loads a volume / mask pair from disk (.npy).
        2. Clips intensity to the liver HU window and z-score normalises.
        3. Binarises the mask (all positive labels → 1).
        4. Extracts a 3-D patch (random during training, centre during val).
        5. Returns ``{"image": Tensor(1,D,H,W), "mask": Tensor(1,D,H,W),
                      "name": str}``.

    Args:
        data_dir:  Root directory containing images_subdir and labels_subdir.
        file_list: Base filenames (without .npy extension) to include.
                   If None, all .npy files found in images_subdir are used.
        mode:      One of ``'train'``, ``'val'``, or ``'test'``.
        config:    Full config dict loaded from config.yaml.
    """

    def __init__(
        self,
        data_dir: str,
        file_list: Optional[List[str]],
        mode: str,
        config: Dict,
    ) -> None:
        assert mode in {"train", "val", "test"}, f"Unknown mode: {mode}"
        self.mode = mode
        self.cfg_pre = config["preprocessing"]
        self.cfg_data = config["dataset"]
        self.patch_size: Tuple[int, int, int] = tuple(self.cfg_pre["patch_size"])  # type: ignore[assignment]

        images_dir = Path(data_dir) / self.cfg_data["images_subdir"]
        labels_dir = Path(data_dir) / self.cfg_data["labels_subdir"]

        img_suffix = self.cfg_data.get("images_suffix", "")
        lbl_suffix = self.cfg_data.get("labels_suffix", "")

        if file_list is None:
            file_list = sorted(
                p.stem[: -len(img_suffix)] if img_suffix else p.stem
                for p in images_dir.glob("*.npy")
            )

        self.samples: List[Tuple[Path, Optional[Path]]] = []
        for name in file_list:
            img_path = images_dir / f"{name}{img_suffix}.npy"
            lbl_path = labels_dir / f"{name}{lbl_suffix}.npy"
            if not img_path.exists():
                continue
            self.samples.append(
                (img_path, lbl_path if lbl_path.exists() else None)
            )

        if not self.samples:
            raise FileNotFoundError(
                f"No valid .npy files found in {images_dir}. "
                "Check dataset.images_subdir in config.yaml."
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img_path, lbl_path = self.samples[idx]

        volume = np.load(str(img_path)).astype(np.float32)
        mask = (
            np.load(str(lbl_path)).astype(np.float32)
            if lbl_path is not None
            else np.zeros_like(volume)
        )

        # Binarise: treat all positive labels as liver foreground.
        # The original MSD Task03 uses 1=liver and 2=tumour; both are
        # included in the binary liver segmentation target.
        mask = (mask > 0).astype(np.float32)

        # --- Intensity pre-processing ----------------------------------------
        lo, hi = self.cfg_pre["intensity_clip"]
        volume = np.clip(volume, lo, hi)

        if self.cfg_pre["normalize"]:
            # Per-volume z-score: zero mean, unit standard deviation.
            # Computed after clipping to avoid outlier-driven scale collapse.
            mean = volume.mean()
            std = volume.std() + 1e-8
            volume = (volume - mean) / std

        # --- Patch extraction ------------------------------------------------
        volume, mask = self._extract_patch(volume, mask)

        # Add channel dimension → (1, D, H, W)
        image_t = torch.from_numpy(volume).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return {"image": image_t, "mask": mask_t, "name": img_path.stem}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pad_to_patch(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reflect-pad volume / zero-pad mask so each dim >= patch_size."""
        pad_vol, pad_msk = [], []
        for dim_size, p_size in zip(volume.shape, self.patch_size):
            deficit = max(0, p_size - dim_size)
            pad_vol.append((deficit // 2, deficit - deficit // 2))
            pad_msk.append((deficit // 2, deficit - deficit // 2))

        if any(lo + hi > 0 for lo, hi in pad_vol):
            volume = np.pad(volume, pad_vol, mode="reflect")
            mask = np.pad(mask, pad_msk, mode="constant", constant_values=0)
        return volume, mask

    def _extract_patch(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a (patch_size) sub-volume from volume and the matching mask."""
        volume, mask = self._pad_to_patch(volume, mask)
        pd, ph, pw = self.patch_size
        D, H, W = volume.shape

        if self.mode == "train":
            # Foreground-biased sampling: with prob 0.5 place the patch centre
            # on a randomly chosen foreground voxel to avoid empty-liver crops.
            fg_voxels = np.argwhere(mask > 0)
            if len(fg_voxels) > 0 and random.random() < 0.5:
                cz, cy, cx = fg_voxels[random.randint(0, len(fg_voxels) - 1)]
                z = int(np.clip(cz - pd // 2, 0, D - pd))
                y = int(np.clip(cy - ph // 2, 0, H - ph))
                x = int(np.clip(cx - pw // 2, 0, W - pw))
            else:
                # Uniform random crop
                z = random.randint(0, D - pd)
                y = random.randint(0, H - ph)
                x = random.randint(0, W - pw)
        else:
            # Deterministic centre crop for validation / test
            z = (D - pd) // 2
            y = (H - ph) // 2
            x = (W - pw) // 2

        return (
            volume[z : z + pd, y : y + ph, x : x + pw],
            mask[z : z + pd, y : y + ph, x : x + pw],
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: Dict,
    data_dir: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    A fixed seed (42) is used for the train/val file split to ensure
    reproducible runs.  The val_split fraction from config is held out.

    Args:
        config:   Full config dict from config.yaml.
        data_dir: Root data directory (local or Kaggle path).

    Returns:
        (train_loader, val_loader)
    """
    images_dir = Path(data_dir) / config["dataset"]["images_subdir"]
    img_suffix = config["dataset"].get("images_suffix", "")
    all_files = sorted(
        p.stem[: -len(img_suffix)] if img_suffix else p.stem
        for p in images_dir.glob("*.npy")
    )

    if not all_files:
        raise FileNotFoundError(
            f"No .npy files found in {images_dir}. "
            "Verify the Kaggle dataset path and images_subdir."
        )

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(all_files).tolist()

    n_val = max(1, int(len(shuffled) * config["dataset"]["val_split"]))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    train_ds = LiverCTDataset(data_dir, train_files, "train", config)
    val_ds = LiverCTDataset(data_dir, val_files, "val", config)

    cfg_ds = config["dataset"]
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg_ds["num_workers"],
        pin_memory=cfg_ds["pin_memory"],
        drop_last=True,         # avoids BatchNorm edge-cases with size-1 batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # full-volume metrics require one sample at a time
        shuffle=False,
        num_workers=cfg_ds["num_workers"],
        pin_memory=cfg_ds["pin_memory"],
    )

    return train_loader, val_loader
