"""
Sliding-window inference over full CT volumes.

Full volumes are too large to forward through the model in one pass.
MONAI's sliding_window_inference tiles the volume into overlapping patches,
runs the model on each, and merges predictions using Gaussian weighting
(smooth boundaries, reduced tiling artefacts).

Only MONAI's inferer is used; all other logic is pure PyTorch / NumPy.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference


def predict_volume(
    model: nn.Module,
    volume: np.ndarray,
    config: Dict,
    device: torch.device,
) -> np.ndarray:
    """
    Run sliding-window inference on a single pre-processed volume.

    The volume should already be intensity-clipped and z-score normalised
    (the same pre-processing applied during training).

    Args:
        model:   Trained UNet3D, in eval mode.
        volume:  3-D float32 array of shape (D, H, W).
        config:  Full config dict from config.yaml.
        device:  Compute device.

    Returns:
        Binary segmentation mask, shape (D, H, W), dtype uint8.
    """
    cfg_inf = config["inference"]
    patch_size = tuple(cfg_inf["patch_size"])

    # Add batch and channel dimensions → (1, 1, D, H, W)
    tensor = (
        torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
    )

    model.eval()
    with torch.no_grad():
        # sliding_window_inference handles tiling, Gaussian blending, and
        # accumulation internally; sw_batch_size controls GPU memory usage.
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=patch_size,
            sw_batch_size=cfg_inf["sw_batch_size"],
            predictor=model,
            overlap=cfg_inf["overlap"],
            mode="gaussian",   # Gaussian weight window for smooth boundary merging
        )

    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def predict_from_file(
    model: nn.Module,
    image_path: str,
    config: Dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a .npy volume from disk, pre-process it, and run inference.

    Applies the same intensity clipping and per-volume normalisation that
    was applied during training (defined in config.preprocessing).

    Args:
        model:       Trained UNet3D.
        image_path:  Path to the .npy CT volume.
        config:      Full config dict from config.yaml.
        device:      Compute device.

    Returns:
        (volume, mask): normalised volume and binary prediction, both
        shape (D, H, W) as float32 / uint8 numpy arrays.
    """
    cfg_pre = config["preprocessing"]

    volume = np.load(image_path).astype(np.float32)

    lo, hi = cfg_pre["intensity_clip"]
    volume = np.clip(volume, lo, hi)

    if cfg_pre["normalize"]:
        mean = volume.mean()
        std = volume.std() + 1e-8
        volume = (volume - mean) / std

    mask = predict_volume(model, volume, config, device)
    return volume, mask
