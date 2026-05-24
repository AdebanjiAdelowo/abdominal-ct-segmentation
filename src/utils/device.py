"""Automatic compute-device selection: CUDA → MPS → CPU."""

import torch


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority:
        1. CUDA  — Nvidia GPU (Kaggle T4 / Colab / cluster)
        2. MPS   — Apple Silicon (local M-series development)
        3. CPU   — fallback

    Returns:
        torch.device pointing to the selected backend.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[device] Using: {device}")
    return device
