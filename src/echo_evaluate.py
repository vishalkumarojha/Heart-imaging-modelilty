"""Phase 2 evaluation: EF-category classifier on the official TEST split.

Reports per-class one-vs-rest AUROC, overall + balanced accuracy, macro-F1, and
the 3x3 confusion matrix. Writes per-video predictions to CSV.

    python -m src.echo_evaluate
    python -m src.echo_evaluate --checkpoint outputs/checkpoints/echo/echo_cnn_lstm_best.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import echo_config as EC
from .echo_dataset import make_echo_dataloaders
from .echo_train import compute_metrics
from .utils import set_seed, setup_logging

logger = setup_logging()


def load_model(checkpoint: Path, device):
    import torch

    from .echo_model import build_echo_model

    ckpt = torch.load(checkpoint, map_location=device)
    names = ckpt.get("class_names", list(EC.CLASS_NAMES))
    model = build_echo_model(num_classes=len(names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("Loaded %s | epoch %s | saved macro AUROC=%.4f | frames=%s",
                checkpoint.name, ckpt.get("epoch", "?"),
                (ckpt.get("val_metrics") or {}).get("macro_auroc", float("nan")),
                ckpt.get("frames_per_clip", "?"))
    return model, names, ckpt


def _predict(model, loader, device, amp):
    import torch
    from tqdm import tqdm

    probs, true = [], []
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="test", leave=False):
            videos = videos.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(videos)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            true.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(true)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def macro_f1(cm: np.ndarray) -> tuple:
    f1s = []
    for c in range(len(cm)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)), f1s


def evaluate(checkpoint: Path, split: str = "test") -> pd.DataFrame:
    import torch

    set_seed(EC.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, class_names, ckpt = load_model(checkpoint, device)

    cfg = EC.EchoRunConfig()
    cfg.frames_per_clip = int(ckpt.get("frames_per_clip", cfg.frames_per_clip))
    cfg.frame_size = int(ckpt.get("frame_size", cfg.frame_size))
    cfg.num_workers = min(cfg.num_workers, 4)

    loaders = make_echo_dataloaders(cfg)
    loader = loaders.get(split)
    if loader is None:
        raise RuntimeError(f"no data for split '{split}'")
    sub = loaders["_subsets"][split].reset_index(drop=True)

    amp = torch.cuda.is_available()
    probs, true = _predict(model, loader, device, amp)
    n = min(len(probs), len(sub))
    probs, true = probs[:n], true[:n]
    pred = probs.argmax(1)

    m = compute_metrics(true, probs, class_names)
    cm = confusion_matrix(true, pred, len(class_names))
    mf1, f1s = macro_f1(cm)

    logger.info("\n===== TEST RESULTS (%s, checkpoint=%s) =====", split, checkpoint.name)
    logger.info("%-16s %8s %8s", "class", "AUROC", "F1")
    logger.info("%s", "-" * 34)
    for i, c in enumerate(class_names):
        logger.info("%-16s %8.4f %8.4f", c, m["per_class_auroc"][c], f1s[i])
    logger.info("%s", "-" * 34)
    logger.info("macro AUROC        : %.4f", m["macro_auroc"])
    logger.info("macro F1           : %.4f", mf1)
    logger.info("overall accuracy   : %.4f", m["accuracy"])
    logger.info("balanced accuracy  : %.4f", m["balanced_accuracy"])
    logger.info("confusion matrix (rows=true, cols=pred) [%s]:", ", ".join(class_names))
    for i, c in enumerate(class_names):
        logger.info("  %-16s %s", c, cm[i].tolist())

    out = sub[["FileName", "split", "EF", "ef_bucket"]].iloc[:n].copy()
    out = out.rename(columns={"ef_bucket": "true_class"})
    out["pred_class"] = pred
    for i, c in enumerate(class_names):
        out[f"prob_{i}_{c.replace(' ', '')}"] = probs[:, i]
    out.to_csv(EC.PREDICTIONS_CSV, index=False)
    logger.info("Per-video predictions -> %s", EC.PREDICTIONS_CSV)

    rows = [{"class": c, "auroc": m["per_class_auroc"][c], "f1": f1s[i],
             "support": int(cm[i].sum())} for i, c in enumerate(class_names)]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EchoNet EF-category checkpoint")
    p.add_argument("--checkpoint", type=Path, default=EC.CKPT_BEST)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    evaluate(args.checkpoint, args.split)
