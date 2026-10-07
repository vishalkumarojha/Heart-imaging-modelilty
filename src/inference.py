"""Deterministic logit / probability extraction for a checkpointed model.

Every downstream research step (calibration, thresholds, error analysis,
Grad-CAM case selection, external validation) consumes *the same* cached
prediction table instead of re-running the model, so all experiments are
comparable and auditable:

    outputs/predictions/raw/{split}__{ckpt_stem}__{sha8}.csv
    outputs/predictions/raw/{split}__{ckpt_stem}__{sha8}.meta.json

Columns:
    Image Index, Patient ID, split,
    true_<label>, logit_<label>, prob_<label>       (prob = sigmoid(logit))

Design rules
------------
* logits are stored, not just probabilities: temperature scaling is defined on
  logits (sigmoid(logit / T)), so the raw logit is the canonical artifact.
* The cache key contains the checkpoint SHA-256 prefix — a re-trained model can
  never silently reuse an old model's predictions.
* Evaluation transforms only (no augmentation), shuffle=False, so row order
  equals the frozen split frame order.
* AMP is used on CUDA exactly like `src.evaluate.py`, so cached scores are
  numerically comparable with the historical baseline run.

    python -m src.inference --split test          # warm the cache
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .reproducibility import capture_environment, sha256_file, short_hash, write_json
from .utils import set_seed, setup_logging

logger = setup_logging()

KEY_COLUMNS = ["Image Index", "Patient ID", "split"]


# --------------------------------------------------------------------------- #
# Model loading (shared with evaluate.py semantics)
# --------------------------------------------------------------------------- #
def load_model(checkpoint: Path, device):
    """Rebuild the classifier from a checkpoint. Returns (model, label_names)."""
    import torch

    from .model import build_model

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    labels: List[str] = list(ckpt.get("target_labels", list(C.TARGET_LABELS)))
    model = build_model(num_classes=len(labels), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "Loaded %s | epoch=%s | val AUROC mean=%s | labels=%s",
        checkpoint.name, ckpt.get("epoch", "?"),
        (ckpt.get("val_auroc") or {}).get("mean", "n/a"), labels,
    )
    return model, labels


# --------------------------------------------------------------------------- #
# Core prediction
# --------------------------------------------------------------------------- #
def predict_split(
    checkpoint: Path,
    split: str = "test",
    *,
    device=None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_amp: Optional[bool] = None,
    refresh: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """Run the model over one split and return the prediction table.

    Parameters
    ----------
    checkpoint : model artifact (baseline or research).
    split      : 'train' | 'val' | 'test'.
    refresh    : recompute even if a valid cache file exists.
    cache      : set False for a throwaway computation (tests).

    Raises
    ------
    FileNotFoundError  checkpoint missing
    RuntimeError       requested split has no data / row count mismatch
    """
    import torch
    from tqdm import tqdm

    from .dataset import make_dataloaders

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train first.")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")

    ckpt_sha = sha256_file(checkpoint)
    cache_path = _cache_path(checkpoint, split, ckpt_sha)
    meta_path = cache_path.with_suffix(".meta.json").with_name(
        cache_path.name.replace(".csv", ".meta.json")
    )
    if cache and not refresh and cache_path.exists():
        frame = pd.read_csv(cache_path)
        logger.info("Using cached predictions: %s (%d rows)", cache_path, len(frame))
        return frame

    set_seed(C.SEED)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(C.AMP and device.type == "cuda") if use_amp is None else use_amp

    model, label_names = load_model(checkpoint, device)

    cfg = C.RunConfig()
    if batch_size is not None:
        cfg.batch_size = batch_size
    if num_workers is not None:
        cfg.num_workers = num_workers
    loaders = make_dataloaders(cfg)
    loader = loaders.get(split)
    subset = loaders["_subsets"].get(split)
    if loader is None or subset is None or len(subset) == 0:
        raise RuntimeError(f"No data for split '{split}'.")
    if split == "train":
        logger.warning(
            "Predicting the TRAIN split: augmentation is active for train "
            "(see build_transforms) — scores are not deterministic."
        )

    t0 = time.time()
    logits: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"predict[{split}]", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(images)
            logits.append(out.float().cpu().numpy())
            targets.append(labels.numpy())

    logits_arr = np.concatenate(logits)
    targets_arr = np.concatenate(targets)
    if len(logits_arr) != len(subset):
        raise RuntimeError(
            f"Row-count mismatch: model produced {len(logits_arr)} rows for a "
            f"{len(subset)}-row split — refusing to write a misaligned table."
        )

    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    out = subset[KEY_COLUMNS].reset_index(drop=True).copy()
    for i, name in enumerate(label_names):
        out[f"true_{name}"] = targets_arr[:, i].astype(int)
        out[f"logit_{name}"] = logits_arr[:, i]
        out[f"prob_{name}"] = probs[:, i]

    dt = time.time() - t0
    logger.info(
        "Predicted %s/%s: %d rows in %.1fs (amp=%s, device=%s)",
        checkpoint.name, split, len(out), dt, amp, device,
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
        write_json(meta_path, {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": ckpt_sha,
            "split": split,
            "rows": int(len(out)),
            "labels": label_names,
            "image_size": cfg.image_size,
            "amp": amp,
            "device": str(device),
            "seconds": round(dt, 2),
            "environment": capture_environment(include_packages=False),
        })
        logger.info("Cached -> %s", cache_path)
    return out


def _cache_path(checkpoint: Path, split: str, ckpt_sha: str) -> Path:
    return C.RAW_PREDICTIONS_DIR / f"{split}__{checkpoint.stem}__{short_hash(ckpt_sha)}.csv"


def load_predictions(checkpoint: Path, split: str) -> Optional[pd.DataFrame]:
    """Return the cached prediction table if present, else None."""
    if not Path(checkpoint).exists():
        return None
    path = _cache_path(checkpoint, split, sha256_file(Path(checkpoint)))
    return pd.read_csv(path) if path.exists() else None


def ensure_predictions(
    checkpoint: Path, split: str, **kwargs
) -> pd.DataFrame:
    """Load from cache or compute — the entry point experiments should use."""
    cached = load_predictions(checkpoint, split)
    if cached is not None and not kwargs.get("refresh"):
        return cached
    return predict_split(checkpoint, split, **kwargs)


# --------------------------------------------------------------------------- #
# Column helpers
# --------------------------------------------------------------------------- #
def label_names_from(predictions: pd.DataFrame) -> List[str]:
    """['Cardiomegaly', 'Effusion'] inferred from `true_*` columns."""
    return [c[len("true_"):] for c in predictions.columns if c.startswith("true_")]


def as_arrays(
    predictions: pd.DataFrame, label_names: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
    """Split a prediction table into {y_true, logits, probs} (N, C) arrays."""
    labels = list(label_names) if label_names else label_names_from(predictions)
    return {
        "labels": labels,
        "y_true": predictions[[f"true_{l}" for l in labels]].to_numpy(dtype=np.int64),
        "logits": predictions[[f"logit_{l}" for l in labels]].to_numpy(dtype=np.float64),
        "probs": predictions[[f"prob_{l}" for l in labels]].to_numpy(dtype=np.float64),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache model logits/probabilities for a split")
    p.add_argument("--checkpoint", type=Path, default=C.BASELINE_CHECKPOINT_BEST)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--refresh", action="store_true", help="recompute even if cached")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_split(args.checkpoint, args.split, refresh=args.refresh)
