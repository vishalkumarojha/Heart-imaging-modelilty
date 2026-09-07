"""NIH ChestX-ray14 data plumbing for Phase 1 (Cardiomegaly / Effusion).

Layers, cleanest first:

  scan_image_paths()      -> {filename: absolute_path}          (X-ray specific)
  build_label_frame()     -> DataFrame[Image Index, Patient ID, <label cols>]
  assign_splits()         -> adds a 'split' column (official or patient-level)
  build_dataset_frame()   -> the joined, on-disk-verified table
  MultiLabelImageDataset  -> generic (image_path + label vector) -> tensors
  make_dataloaders()      -> the three loaders wired for train.py

Only `scan_image_paths` / `build_label_frame` know anything about NIH. The
Dataset class is modality-agnostic on purpose: an ECHO/MRI branch can feed it a
different frame + different transforms and reuse the rest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .utils import log_label_distribution

logger = logging.getLogger("capstone")

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- #
# 1. Locate images on disk  (handles flat AND nested "images/" subfolder)
# --------------------------------------------------------------------------- #
def scan_image_paths(
    raw_dir: Path = C.DATA_RAW_DIR, use_cache: bool = True
) -> Dict[str, str]:
    """Return {filename -> absolute path} for every image under images_*/ .

    Each `images_XXX/` may hold the PNGs directly or inside an extra `images/`
    subfolder; both are discovered with a recursive glob so we don't assume.
    """
    if use_cache and C.IMAGE_PATH_CACHE.exists():
        df = pd.read_csv(C.IMAGE_PATH_CACHE)
        logger.info("Loaded image path index from cache (%d files).", len(df))
        return dict(zip(df["filename"], df["path"]))

    mapping: Dict[str, str] = {}
    folders = sorted(raw_dir.glob("images_*"))
    if not folders:
        raise FileNotFoundError(f"No images_* folders under {raw_dir}")

    dupes = 0
    for folder in folders:
        found_here = 0
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                if p.name in mapping:
                    dupes += 1
                mapping[p.name] = str(p.resolve())
                found_here += 1
        logger.info("  %-14s -> %6d images", folder.name, found_here)

    if dupes:
        logger.warning("Encountered %d duplicate filenames across folders (kept last).", dupes)
    logger.info("Total unique image files on disk: %d", len(mapping))

    if use_cache:
        pd.DataFrame({"filename": list(mapping), "path": list(mapping.values())}).to_csv(
            C.IMAGE_PATH_CACHE, index=False
        )
    return mapping


# --------------------------------------------------------------------------- #
# 2. Parse Data_Entry_2017.csv -> binary label columns
# --------------------------------------------------------------------------- #
def build_label_frame(
    csv_path: Path = C.DATA_ENTRY_CSV,
    target_labels: Sequence[str] = C.TARGET_LABELS,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    findings = df["Finding Labels"].fillna("").str.split("|").apply(
        lambda parts: {s.strip() for s in parts}
    )
    for label in target_labels:
        df[label] = findings.apply(lambda s, l=label: int(l in s)).astype(np.int64)

    keep = ["Image Index", "Patient ID", "Finding Labels", *target_labels]
    return df[keep].copy()


# --------------------------------------------------------------------------- #
# 3. Splits
# --------------------------------------------------------------------------- #
def _find_official_split_files(raw_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    tv = next(iter(raw_dir.rglob(C.OFFICIAL_TRAINVAL_LIST)), None)
    te = next(iter(raw_dir.rglob(C.OFFICIAL_TEST_LIST)), None)
    return tv, te


def _read_list_file(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def assign_splits(
    frame: pd.DataFrame,
    raw_dir: Path = C.DATA_RAW_DIR,
    fractions: Tuple[float, float, float] = C.SPLIT_FRACTIONS,
    seed: int = C.SEED,
) -> Tuple[pd.DataFrame, str]:
    """Add a 'split' column ('train'|'val'|'test'). Returns (frame, method_str).

    Official NIH files (train_val_list.txt + test_list.txt) are used if found;
    since they only define train_val vs test, train_val is further divided into
    train/val at the PATIENT level. With no official files, all three splits are
    patient-level so no patient leaks across them.
    """
    frame = frame.copy()
    tv_file, te_file = _find_official_split_files(raw_dir)
    rng = np.random.default_rng(seed)

    if tv_file and te_file:
        method = f"official NIH files ({tv_file.name} + {te_file.name}); val carved from train_val at patient level"
        test_names = set(_read_list_file(te_file))
        trainval_names = set(_read_list_file(tv_file))
        frame["split"] = "unassigned"
        frame.loc[frame["Image Index"].isin(test_names), "split"] = "test"

        tv_mask = frame["Image Index"].isin(trainval_names)
        tv_patients = frame.loc[tv_mask, "Patient ID"].unique()
        rng.shuffle(tv_patients)
        # val fraction expressed relative to the train_val pool
        val_frac_within = fractions[1] / (fractions[0] + fractions[1])
        n_val = int(round(len(tv_patients) * val_frac_within))
        val_patients = set(tv_patients[:n_val])
        frame.loc[tv_mask & frame["Patient ID"].isin(val_patients), "split"] = "val"
        frame.loc[tv_mask & ~frame["Patient ID"].isin(val_patients), "split"] = "train"
    else:
        method = f"patient-level random split {tuple(fractions)} (no official split files found)"
        patients = frame["Patient ID"].unique().copy()
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        train_p = set(patients[:n_train])
        val_p = set(patients[n_train : n_train + n_val])
        frame["split"] = frame["Patient ID"].map(
            lambda pid: "train" if pid in train_p else ("val" if pid in val_p else "test")
        )

    # sanity: no patient across splits
    leak = (
        frame.groupby("Patient ID")["split"].nunique().gt(1).sum()
    )
    if leak:
        logger.warning("PATIENT LEAK: %d patients appear in >1 split!", leak)
    else:
        logger.info("Patient-level split OK: no patient shared across splits.")
    return frame, method


# --------------------------------------------------------------------------- #
# 4. Join everything + verify files exist on disk
# --------------------------------------------------------------------------- #
def build_dataset_frame(
    target_labels: Sequence[str] = C.TARGET_LABELS,
    use_cache: bool = True,
    write_cache: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """The master table used by training/eval.

    Columns: Image Index, Patient ID, Finding Labels, <labels>, path, split
    Only rows whose image file was actually located on disk are kept.
    Returns (frame, report_dict) where report_dict feeds the exploration print.
    """
    if use_cache and C.SPLIT_CACHE_CSV.exists():
        frame = pd.read_csv(C.SPLIT_CACHE_CSV)
        logger.info("Loaded dataset frame from cache: %s (%d rows).", C.SPLIT_CACHE_CSV.name, len(frame))
        report = {"from_cache": True, "rows": len(frame)}
        return frame, report

    path_map = scan_image_paths(use_cache=use_cache)
    labels = build_label_frame(target_labels=target_labels)
    labels, split_method = assign_splits(labels)

    n_csv = len(labels)
    labels["path"] = labels["Image Index"].map(path_map)
    missing_mask = labels["path"].isna()
    n_missing = int(missing_mask.sum())

    if n_missing:
        logger.warning(
            "%d / %d CSV entries have NO matching image file on disk - dropping them.",
            n_missing, n_csv,
        )
        sample_missing = labels.loc[missing_mask, "Image Index"].head(10).tolist()
        logger.warning("  e.g. %s", sample_missing)

    n_disk_only = len(set(path_map) - set(labels["Image Index"]))
    if n_disk_only:
        logger.warning("%d image files on disk are NOT referenced by the CSV.", n_disk_only)

    frame = labels.loc[~missing_mask].reset_index(drop=True)

    report = {
        "from_cache": False,
        "images_on_disk": len(path_map),
        "csv_entries": n_csv,
        "csv_without_file": n_missing,
        "disk_without_csv": n_disk_only,
        "usable_rows": len(frame),
        "split_method": split_method,
        "split_counts": frame["split"].value_counts().to_dict(),
        "label_counts": {
            lbl: {
                "total_pos": int(frame[lbl].sum()),
                "by_split": frame.groupby("split")[lbl].sum().to_dict(),
            }
            for lbl in target_labels
        },
    }

    if write_cache:
        frame.to_csv(C.SPLIT_CACHE_CSV, index=False)
        logger.info("Wrote dataset frame cache -> %s", C.SPLIT_CACHE_CSV)

    return frame, report


# --------------------------------------------------------------------------- #
# 5. Transforms  (albumentations; NO horizontal flip)
# --------------------------------------------------------------------------- #
def build_transforms(train: bool, image_size: int = C.IMAGE_SIZE):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    norm = A.Normalize(mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD)
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=C.AUG_ROTATION_DEG, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=C.AUG_BRIGHTNESS,
                contrast_limit=C.AUG_CONTRAST,
                p=0.7,
            ),
            norm,
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(image_size, image_size), norm, ToTensorV2()])


# --------------------------------------------------------------------------- #
# 6. Dataset  (modality-agnostic: path + label vector -> tensors)
# --------------------------------------------------------------------------- #
class MultiLabelImageDataset:
    """Generic multi-label image dataset.

    Not tied to NIH: give it a frame with a `path` column and `label_columns`,
    plus an albumentations transform. Corrupt/missing files are skipped by
    returning the next valid sample (and logged once).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        label_columns: Sequence[str],
        transform: Optional[Callable] = None,
        path_column: str = "path",
    ) -> None:
        # torch imported here so `import dataset` stays cheap for exploration
        import torch  # noqa: F401

        self.frame = frame.reset_index(drop=True)
        self.label_columns = list(label_columns)
        self.transform = transform
        self.path_column = path_column
        self._labels = self.frame[self.label_columns].to_numpy(dtype=np.float32)
        self._paths = self.frame[self.path_column].tolist()
        self._bad: set[int] = set()

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im.convert("RGB"))

    def __getitem__(self, idx: int):
        import torch

        n = len(self._paths)
        for offset in range(n):
            j = (idx + offset) % n
            if j in self._bad:
                continue
            try:
                img = self._load_image(self._paths[j])
                if self.transform is not None:
                    img = self.transform(image=img)["image"]
                else:
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                label = torch.from_numpy(self._labels[j]).float()
                return img, label
            except Exception as e:  # skip & log, don't crash
                if j not in self._bad:
                    logger.warning("Skipping unreadable image %s (%s)", self._paths[j], e)
                    self._bad.add(j)
        raise RuntimeError("No readable images left in dataset.")


# --------------------------------------------------------------------------- #
# 7. Dataloaders
# --------------------------------------------------------------------------- #
def make_dataloaders(
    cfg: "C.RunConfig",
    frame: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Build train/val/test DataLoaders from the master frame.

    Honours cfg.quick_test (500 train / 100 val, class-stratified-ish sample).
    """
    from torch.utils.data import DataLoader

    if frame is None:
        frame, _ = build_dataset_frame(target_labels=cfg.target_labels)

    subsets = {s: frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

    if cfg.quick_test:
        subsets["train"] = _quick_sample(subsets["train"], C.QUICK_TEST_TRAIN_SAMPLES, cfg.target_labels, cfg.seed)
        subsets["val"] = _quick_sample(subsets["val"], C.QUICK_TEST_VAL_SAMPLES, cfg.target_labels, cfg.seed)
        logger.info("quick_test: train=%d val=%d", len(subsets["train"]), len(subsets["val"]))

    loaders: Dict[str, object] = {}
    for split, sub in subsets.items():
        if len(sub) == 0:
            loaders[split] = None
            continue
        is_train = split == "train"
        ds = MultiLabelImageDataset(
            sub, cfg.target_labels, transform=build_transforms(is_train, cfg.image_size)
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=cfg.num_workers > 0,
        )
    loaders["_frame"] = frame
    loaders["_subsets"] = subsets
    return loaders


def _quick_sample(
    sub: pd.DataFrame, n: int, label_cols: Sequence[str], seed: int
) -> pd.DataFrame:
    """Grab ~n rows but make sure some positives for each label are included."""
    if len(sub) <= n:
        return sub
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    per_label = max(5, n // (2 * max(1, len(label_cols))))
    for lbl in label_cols:
        pos_idx = sub.index[sub[lbl] == 1].tolist()
        rng.shuffle(pos_idx)
        picks.extend(pos_idx[:per_label])
    picks = list(dict.fromkeys(picks))  # dedupe (an image can be positive for both)
    remaining = [i for i in sub.index if i not in set(picks)]
    rng.shuffle(remaining)
    picks.extend(remaining[: n - len(picks)])
    return sub.loc[picks[:n]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Convenience for exploration scripts
# --------------------------------------------------------------------------- #
def summarize(frame: pd.DataFrame, target_labels: Sequence[str] = C.TARGET_LABELS) -> None:
    for split in ("train", "val", "test"):
        sub = frame[frame["split"] == split]
        if len(sub):
            log_label_distribution(sub[list(target_labels)].to_numpy(), target_labels, split)
