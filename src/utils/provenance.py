"""
Provenance and durable-write helpers shared by training, evaluation and the smoke test.

Checkpoints are expensive and, as this project learned, easy to lose.  Everything
that writes one goes through ``atomic_torch_save`` (never leaves a half-written
file) and, when a persistent mirror is configured (e.g. Google Drive on Colab),
``mirror_file`` (copy to a temp name, verify, then rename).
"""

import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def git_state() -> Dict[str, object]:
    """Commit hash and whether tracked files differ from it (a dirty tree makes the hash meaningless)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def is_tracked_and_clean(path) -> bool:
    """True if ``path`` is committed in this repository and unmodified since."""
    p = Path(path).resolve()
    try:
        rel = str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return False
    if _git("ls-files", "--error-unmatch", rel) is None:
        return False
    return _git("status", "--porcelain", "--", rel) == ""


def atomic_torch_save(obj, path) -> None:
    """torch.save to a temp file in the same directory, fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mirror_file(src, dst_dir, verify_hash: bool = True) -> Path:
    """
    Copy ``src`` into ``dst_dir`` (e.g. a Drive folder) via a temp name and rename.
    With ``verify_hash`` the copy is re-read and its SHA-256 compared before the
    rename, so a truncated network-drive write is never mistaken for a checkpoint.
    """
    src, dst_dir = Path(src), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    if verify_hash and sha256_file(tmp) != sha256_file(src):
        tmp.unlink(missing_ok=True)
        raise IOError(f"Mirror copy of {src.name} failed hash verification; not renamed into place")
    os.replace(tmp, dst)
    return dst


REQUIRED_CHECKPOINT_KEYS = (
    "format_version", "epoch", "model_state", "optimiser_state", "scheduler_state", "val_dice",
    "config", "model_config", "seed", "split_sha256", "git_commit", "timestamp_utc",
)


def verify_checkpoint_dict(ckpt: Dict, path="checkpoint", expected_split_sha256: Optional[str] = None) -> Dict:
    """
    Check that a loaded ``Trainer`` checkpoint is complete: every provenance key
    present, the top-level split fingerprint equal to the one embedded in its
    config, and (optionally) equal to an expected split file's fingerprint.
    """
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path}: checkpoint is missing {missing} (older or foreign format)")
    embedded = (ckpt["config"].get("protocol") or {}).get("splits_sha256")
    if ckpt["split_sha256"] != embedded:
        raise ValueError(f"{path}: split fingerprint fields disagree ({ckpt['split_sha256']} vs {embedded})")
    if expected_split_sha256 is not None and ckpt["split_sha256"] != expected_split_sha256:
        raise ValueError(f"{path}: trained on split {ckpt['split_sha256']}, expected {expected_split_sha256}")
    return ckpt


def load_verified_checkpoint(path, device, expected_split_sha256: Optional[str] = None) -> Dict:
    """torch.load followed by :func:`verify_checkpoint_dict`."""
    return verify_checkpoint_dict(torch.load(path, map_location=device), path, expected_split_sha256)
