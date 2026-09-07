"""Evaluate the best checkpoint on the held-out test split.

Reports per-label AUROC, sensitivity, specificity, precision, F1 (threshold
0.5), prints a summary table, and writes per-image predictions + ground truth
to outputs/logs/test_predictions.csv.

    python -m src.evaluate
    python -m src.evaluate --checkpoint outputs/checkpoints/densenet121_best.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import config as C
from .dataset import make_dataloaders
from .utils import (
    binary_rates_from_confusion,
    format_metrics_table,
    multilabel_auroc,
    set_seed,
    setup_logging,
)

logger = setup_logging()


def load_model_from_checkpoint(checkpoint: Path, device):
    import torch

    from .model import build_model

    ckpt = torch.load(checkpoint, map_location=device)
    labels = ckpt.get("target_labels", list(C.TARGET_LABELS))
    model = build_model(num_classes=len(labels), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info(
        "Loaded %s | epoch %s | saved val AUROC mean=%.4f",
        checkpoint.name, ckpt.get("epoch", "?"),
        (ckpt.get("val_auroc") or {}).get("mean", float("nan")),
    )
    return model, labels


def _predict(model, loader, device, amp):
    import torch
    from tqdm import tqdm

    scores: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="test", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def evaluate(checkpoint: Path, split: str = "test", threshold: float = C.DECISION_THRESHOLD) -> pd.DataFrame:
    import torch

    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, label_names = load_model_from_checkpoint(checkpoint, device)

    cfg = C.RunConfig()
    loaders = make_dataloaders(cfg)
    loader = loaders.get(split)
    if loader is None:
        raise RuntimeError(f"No data for split '{split}'.")
    subset_frame = loaders["_subsets"][split].reset_index(drop=True)

    amp = torch.cuda.is_available()
    scores, targets = _predict(model, loader, device, amp)

    # DataLoader may drop nothing for eval (drop_last=False), lengths should match
    n = min(len(scores), len(subset_frame))
    scores, targets = scores[:n], targets[:n]

    preds = (scores >= threshold).astype(int)
    auroc = multilabel_auroc(targets, scores, label_names)

    rows = []
    for i, name in enumerate(label_names):
        rt = binary_rates_from_confusion(targets[:, i], preds[:, i])
        rows.append({
            "label": name,
            "AUROC": f"{auroc[name]:.4f}",
            "sensitivity": f"{rt['sensitivity']:.4f}",
            "specificity": f"{rt['specificity']:.4f}",
            "precision": f"{rt['precision']:.4f}",
            "F1": f"{rt['f1']:.4f}",
            "support_pos": rt["tp"] + rt["fn"],
            "support_neg": rt["tn"] + rt["fp"],
        })

    table = format_metrics_table(
        rows,
        ["label", "AUROC", "sensitivity", "specificity", "precision", "F1", "support_pos", "support_neg"],
    )
    logger.info("\n===== TEST RESULTS (%s split, threshold=%.2f) =====\n%s\n"
                "mean AUROC across labels: %.4f",
                split, threshold, table, auroc["mean"])

    # per-image predictions CSV
    out = subset_frame[["Image Index", "Patient ID", "split"]].iloc[:n].copy()
    for i, name in enumerate(label_names):
        out[f"true_{name}"] = targets[:, i].astype(int)
        out[f"score_{name}"] = scores[:, i]
        out[f"pred_{name}"] = preds[:, i]
    out.to_csv(C.PREDICTIONS_CSV, index=False)
    logger.info("Per-image predictions -> %s", C.PREDICTIONS_CSV)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate best checkpoint on test split")
    p.add_argument("--checkpoint", type=Path, default=C.CHECKPOINT_BEST)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=C.DECISION_THRESHOLD)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train first.")
    evaluate(args.checkpoint, args.split, args.threshold)
