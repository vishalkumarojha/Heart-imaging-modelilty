"""ACDC cardiac-MRI data plumbing (Phase 3).

    build_mri_frame()      -> DataFrame from data/processed/mri/split_index.csv
    load_volume()          -> np.float32 (X, Y, Z) via nibabel
    robust_normalize()     -> per-volume percentile-clip + z-score
    preprocess_pair()      -> [N_SLICES, 3, HW, HW]  channels = (ED, ES, ED-ES)
    MriVolumeDataset       -> (tensor [N,3,HW,HW], label int in 0..4)
    make_mri_dataloaders() -> train/val/test loaders for mri_train.py

Split (stratified by diagnosis, one patient per row, seed 42) is created by
mri_explore.py and frozen in split_index.csv. Normalisation is per-volume
(MRI intensity varies ~20x between patients) plus a small dataset-level
residual correction from norm_stats.json — NOT ImageNet / EchoNet stats.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import mri_config as MC

logger = logging.getLogger("capstone")


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def build_mri_frame() -> pd.DataFrame:
    if not MC.SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"{MC.SPLIT_CSV} missing - run `python -m src.mri_explore` first."
        )
    df = pd.read_csv(MC.SPLIT_CSV)
    df["split"] = df["split"].astype(str).str.lower()
    return df


def load_norm_stats() -> tuple:
    if MC.NORM_STATS_JSON.exists():
        d = json.loads(MC.NORM_STATS_JSON.read_text())
        return np.array(d["mean"], np.float32), np.array(d["std"], np.float32)
    logger.warning("norm_stats.json missing - using (0,1) residual stats")
    return np.zeros(3, np.float32), np.ones(3, np.float32)


# --------------------------------------------------------------------------- #
# Volume IO + per-volume normalisation
# --------------------------------------------------------------------------- #
def load_volume(path: str) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(path).dataobj, dtype=np.float32)


def robust_normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [MC.CLIP_PCT_LOW, MC.CLIP_PCT_HIGH])
    vol = np.clip(vol, lo, hi)
    return (vol - float(vol.mean())) / (float(vol.std()) + 1e-6)


def _resize_slice(sl: np.ndarray, hw: int):
    from scipy.ndimage import zoom

    return zoom(sl, (hw / sl.shape[0], hw / sl.shape[1]), order=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Augmentation (train only) — identical geometric params for ED & ES, all slices
# --------------------------------------------------------------------------- #
def _augment_pair(ed: np.ndarray, es: np.ndarray, rng: np.random.Generator):
    """ed, es: [N, HW, HW]. Rotation + scale (shared) + intensity jitter."""
    from scipy.ndimage import rotate, zoom

    angle = float(rng.uniform(-MC.AUG_ROTATION_DEG, MC.AUG_ROTATION_DEG))
    ed = rotate(ed, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
    es = rotate(es, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")

    s = 1.0 + float(rng.uniform(-MC.AUG_SCALE, MC.AUG_SCALE))
    if abs(s - 1.0) > 1e-3:
        ed, es = _zoom_center(ed, s), _zoom_center(es, s)

    a = 1.0 + float(rng.uniform(-MC.AUG_INTENSITY_JITTER, MC.AUG_INTENSITY_JITTER))
    b = float(rng.uniform(-MC.AUG_INTENSITY_JITTER, MC.AUG_INTENSITY_JITTER))
    return ed * a + b, es * a + b


def _zoom_center(vol: np.ndarray, s: float) -> np.ndarray:
    from scipy.ndimage import zoom

    n, h, w = vol.shape
    z = zoom(vol, (1, s, s), order=1)
    zh, zw = z.shape[1], z.shape[2]
    if s >= 1.0:  # crop center back to h,w
        top, left = (zh - h) // 2, (zw - w) // 2
        return z[:, top:top + h, left:left + w]
    out = np.zeros((n, h, w), np.float32)  # pad center
    top, left = (h - zh) // 2, (w - zw) // 2
    out[:, top:top + zh, left:left + zw] = z
    return out


# --------------------------------------------------------------------------- #
# Full preprocess: (ED path, ES path) -> [N_SLICES, 3, HW, HW]
# --------------------------------------------------------------------------- #
def preprocess_pair(
    ed_path: str,
    es_path: str,
    cfg: "MC.MriRunConfig",
    train: bool,
    norm: tuple,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    ed = robust_normalize(load_volume(ed_path))          # (X, Y, Z)
    es = robust_normalize(load_volume(es_path))
    z = ed.shape[2]
    idx = np.round(np.linspace(0, z - 1, cfg.n_slices)).astype(int)
    ed = np.stack([_resize_slice(ed[:, :, i], cfg.slice_hw) for i in idx])   # [N, HW, HW]
    es = np.stack([_resize_slice(es[:, :, i], cfg.slice_hw) for i in idx])

    if train:
        ed, es = _augment_pair(ed, es, rng or np.random.default_rng())

    diff = ed - es
    x = np.stack([ed, es, diff], axis=1).astype(np.float32)   # [N, 3, HW, HW]

    mean, std = norm
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return x


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MriVolumeDataset:
    """(tensor [N_SLICES, 3, HW, HW] float32, label int in 0..4). Skips + logs
    unreadable patients, returning the next valid sample."""

    def __init__(self, frame: pd.DataFrame, cfg: "MC.MriRunConfig", train: bool) -> None:
        import torch  # noqa: F401

        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.norm = load_norm_stats()
        self.ed = self.frame["ed_path"].tolist()
        self.es = self.frame["es_path"].tolist()
        self.labels = self.frame["label"].astype(int).tolist()
        self.pids = self.frame["patient_id"].tolist()
        self._bad: set[int] = set()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        import torch

        n = len(self.ed)
        for off in range(n):
            j = (idx + off) % n
            if j in self._bad:
                continue
            try:
                rng = np.random.default_rng(None) if self.train else None
                x = preprocess_pair(self.ed[j], self.es[j], self.cfg, self.train, self.norm, rng)
                return torch.from_numpy(x), int(self.labels[j])
            except Exception as e:
                if j not in self._bad:
                    logger.warning("skipping unreadable patient %s (%s)", self.pids[j], e)
                    self._bad.add(j)
        raise RuntimeError("no readable MRI volumes left in dataset")


# --------------------------------------------------------------------------- #
# Dataloaders
# --------------------------------------------------------------------------- #
def _quick_sample(sub: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(sub) <= n:
        return sub
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    per_class = max(1, n // (2 * MC.NUM_CLASSES))
    for c in range(MC.NUM_CLASSES):
        ci = sub.index[sub["label"] == c].tolist()
        rng.shuffle(ci)
        picks.extend(ci[:per_class])
    picks = list(dict.fromkeys(picks))
    rest = [i for i in sub.index if i not in set(picks)]
    rng.shuffle(rest)
    picks.extend(rest[: n - len(picks)])
    return sub.loc[picks[:n]].reset_index(drop=True)


def make_mri_dataloaders(cfg: "MC.MriRunConfig", frame: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    from torch.utils.data import DataLoader

    if frame is None:
        frame = build_mri_frame()
    subs = {s: frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if cfg.quick_test:
        subs["train"] = _quick_sample(subs["train"], MC.QUICK_TEST_TRAIN, cfg.seed)
        subs["val"] = _quick_sample(subs["val"], MC.QUICK_TEST_VAL, cfg.seed)
        logger.info("quick_test: train=%d val=%d", len(subs["train"]), len(subs["val"]))

    loaders: Dict[str, object] = {}
    for split, sub in subs.items():
        if len(sub) == 0:
            loaders[split] = None
            continue
        is_train = split == "train"
        ds = MriVolumeDataset(sub, cfg, train=is_train)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=is_train and len(sub) > cfg.batch_size,
            persistent_workers=cfg.num_workers > 0,
        )
    loaders["_subsets"] = subs
    loaders["_frame"] = frame
    return loaders
