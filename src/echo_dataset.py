"""EchoNet-Dynamic video data plumbing (Phase 2).

Layers, cleanest first:
    build_echo_frame()      -> DataFrame[FileName, path, ef_bucket, split, ...]
    sample_frame_indices()  -> N evenly-spaced frame indices
    read_clip()             -> [T, H, W, 3] uint8 RGB (cv2)
    build_transforms()      -> clip-consistent albumentations ReplayCompose
    EchoVideoDataset        -> (video_tensor [T,C,H,W], label int)
    make_echo_dataloaders() -> train/val/test loaders for echo_train.py

Uses the OFFICIAL FileList.csv `Split` column (one video per patient -> no
leakage risk, no custom split). Normalisation stats come from
data/processed/echo/norm_stats.json (computed by echo_explore.py) — NOT ImageNet.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import echo_config as EC

try:
    import cv2

    cv2.setNumThreads(0)  # DataLoader workers provide the parallelism
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

logger = logging.getLogger("capstone")


# --------------------------------------------------------------------------- #
# Index table
# --------------------------------------------------------------------------- #
def build_echo_frame(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and EC.INDEX_CACHE_CSV.exists():
        df = pd.read_csv(EC.INDEX_CACHE_CSV)
    else:
        df = pd.read_csv(EC.FILELIST_CSV)
        df.columns = [c.strip() for c in df.columns]
        df["path"] = df["FileName"].map(lambda n: str(EC.VIDEO_DIR / f"{n}.avi"))
        df["ef_bucket"] = df["EF"].astype(float).map(EC.bucket_ef)
    df["split"] = df["Split"].astype(str).str.strip().str.lower()
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    return df


def load_norm_stats() -> tuple:
    if EC.NORM_STATS_JSON.exists():
        d = json.loads(EC.NORM_STATS_JSON.read_text())
        return tuple(d["mean"]), tuple(d["std"])
    logger.warning("norm_stats.json missing - using fallback echo stats")
    return EC.FALLBACK_MEAN, EC.FALLBACK_STD


def compute_class_weights(train_frame: pd.DataFrame, num_classes: int = EC.NUM_CLASSES) -> np.ndarray:
    """Inverse-frequency CrossEntropy weights from the TRAIN split (~[5.25, 3.73, 0.96])."""
    counts = (
        train_frame["ef_bucket"].value_counts().reindex(range(num_classes)).fillna(0).to_numpy(float)
    )
    w = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# Frame sampling + decode
# --------------------------------------------------------------------------- #
def sample_frame_indices(n_total: int, n_want: int) -> np.ndarray:
    """N evenly-spaced integer indices in [0, n_total-1] (repeats if n_total < n_want)."""
    n_total = max(int(n_total), 1)
    return np.round(np.linspace(0, n_total - 1, n_want)).astype(int)


def read_clip(path: str, n_want: int) -> np.ndarray:
    """Decode `path` and return `n_want` evenly-spaced frames as [T, H, W, 3] uint8 RGB."""
    if cv2 is None:
        raise RuntimeError("opencv (cv2) not available")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"cannot open video: {path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sample_frame_indices(n_total if n_total > 0 else n_want, n_want)
    want_set = {int(i) for i in want}
    max_needed = int(want.max())

    grabbed: Dict[int, np.ndarray] = {}
    i = 0
    while i <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want_set:
            grabbed[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    if not grabbed:
        raise IOError(f"no frames decoded: {path}")

    # assemble in requested order; fill any gap (decode short) with nearest grabbed frame
    first_avail = grabbed[min(grabbed)]
    frames, last = [], first_avail
    for j in want:
        f = grabbed.get(int(j), last)
        frames.append(f)
        last = f
    return np.stack(frames)


# --------------------------------------------------------------------------- #
# Transforms (clip-consistent: one random parametrisation per clip)
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, frame_size: int, mean: Sequence[float], std: Sequence[float]):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ops = []
    if train:
        ops += [
            A.Rotate(limit=EC.AUG_ROTATION_DEG, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=EC.AUG_BRIGHTNESS, contrast_limit=EC.AUG_CONTRAST, p=0.7
            ),
        ]
        if EC.AUG_HFLIP:  # off by default - echo apical-4-chamber has a fixed orientation
            ops.append(A.HorizontalFlip(p=0.5))
    ops += [
        A.Resize(frame_size, frame_size),
        A.Normalize(mean=tuple(mean), std=tuple(std), max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.ReplayCompose(ops)


def apply_clip(transform, frames_np: np.ndarray):
    """Apply `transform` to frame 0, then replay the SAME random params on every
    other frame so augmentation is temporally coherent. -> tensor [T, C, H, W]."""
    import albumentations as A
    import torch

    first = transform(image=frames_np[0])
    replay = first["replay"]
    out = [first["image"]]
    for k in range(1, len(frames_np)):
        out.append(A.ReplayCompose.replay(replay, image=frames_np[k])["image"])
    return torch.stack(out)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EchoVideoDataset:
    """(video_tensor [T, C, H, W] float32, label int in {0,1,2}). Skips + logs
    unreadable videos, returning the next valid sample instead of crashing."""

    def __init__(self, frame: pd.DataFrame, cfg: "EC.EchoRunConfig", train: bool) -> None:
        import torch  # noqa: F401

        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        mean, std = load_norm_stats()
        self.transform = build_transforms(train, cfg.frame_size, mean, std)
        self.paths: List[str] = self.frame["path"].tolist()
        self.labels: List[int] = self.frame["ef_bucket"].astype(int).tolist()
        self._bad: set[int] = set()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        n = len(self.paths)
        for off in range(n):
            j = (idx + off) % n
            if j in self._bad:
                continue
            try:
                clip = read_clip(self.paths[j], self.cfg.frames_per_clip)
                video = apply_clip(self.transform, clip)          # [T, C, H, W]
                return video, int(self.labels[j])
            except Exception as e:
                if j not in self._bad:
                    logger.warning("skipping unreadable video %s (%s)", self.paths[j], e)
                    self._bad.add(j)
        raise RuntimeError("no readable videos left in dataset")


# --------------------------------------------------------------------------- #
# Dataloaders
# --------------------------------------------------------------------------- #
def _quick_sample(sub: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(sub) <= n:
        return sub
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    per_class = max(2, n // (2 * EC.NUM_CLASSES))
    for c in range(EC.NUM_CLASSES):
        ci = sub.index[sub["ef_bucket"] == c].tolist()
        rng.shuffle(ci)
        picks.extend(ci[:per_class])
    picks = list(dict.fromkeys(picks))
    rest = [i for i in sub.index if i not in set(picks)]
    rng.shuffle(rest)
    picks.extend(rest[: n - len(picks)])
    return sub.loc[picks[:n]].reset_index(drop=True)


def make_echo_dataloaders(cfg: "EC.EchoRunConfig", frame: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    from torch.utils.data import DataLoader

    if frame is None:
        frame = build_echo_frame()

    subs = {s: frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if cfg.quick_test:
        subs["train"] = _quick_sample(subs["train"], EC.QUICK_TEST_TRAIN, cfg.seed)
        subs["val"] = _quick_sample(subs["val"], EC.QUICK_TEST_VAL, cfg.seed)
        logger.info("quick_test: train=%d val=%d", len(subs["train"]), len(subs["val"]))

    loaders: Dict[str, object] = {}
    for split, sub in subs.items():
        if len(sub) == 0:
            loaders[split] = None
            continue
        is_train = split == "train"
        ds = EchoVideoDataset(sub, cfg, train=is_train)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=cfg.num_workers > 0,
        )
    loaders["_subsets"] = subs
    loaders["_frame"] = frame
    return loaders
