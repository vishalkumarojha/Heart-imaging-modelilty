"""Phase 3 evaluation: ACDC 5-class diagnosis classifier on the held-out TEST split.

Per-class one-vs-rest AUROC, overall + balanced accuracy, macro-F1, and the 5x5
confusion matrix. Writes per-patient predictions to CSV.

NOTE: test = 15 patients (3 per class). Every number here carries a large error
bar — one misclassified patient moves a class recall by 33 pp. Read as ballpark.

    python -m src.mri_evaluate
    python -m src.mri_evaluate --checkpoint outputs/checkpoints/mri/mri_resnet18_bilstm_best.pt --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import mri_config as MC
from .mri_dataset import make_mri_dataloaders
from .mri_train import compute_metrics
from .utils import set_seed, setup_logging

logger = setup_logging()


def load_model(checkpoint: Path, device):
    import torch

    from .mri_model import build_mri_model

    ckpt = torch.load(checkpoint, map_location=device)
    names = ckpt.get("class_names", list(MC.CLASS_NAMES))
    model = build_mri_model(num_classes=len(names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    logger.info("Loaded %s | epoch %s | saved macro AUROC=%.4f | n_slices=%s hw=%s",
                checkpoint.name, ckpt.get("epoch", "?"),
                (ckpt.get("val_metrics") or {}).get("macro_auroc", float("nan")),
                ckpt.get("n_slices", "?"), ckpt.get("slice_hw", "?"))
    return model, names, ckpt


def _predict(model, loader, device, amp):
    import torch
    from tqdm import tqdm

    P, T = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="test", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            P.append(torch.softmax(logits.float(), 1).cpu().numpy())
            T.append(y.numpy())
    probs = np.concatenate(P)
    bad = ~np.isfinite(probs).all(1)
    if bad.any():
        logger.warning("%d/%d rows had non-finite probs; set to uniform", int(bad.sum()), len(probs))
        probs[bad] = 1.0 / probs.shape[1]
    return probs, np.concatenate(T)


def confusion_matrix(y_true, y_pred, k):
    cm = np.zeros((k, k), int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def macro_f1(cm):
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

    set_seed(MC.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, class_names, ckpt = load_model(checkpoint, device)
    cfg = MC.MriRunConfig()
    cfg.n_slices = int(ckpt.get("n_slices", cfg.n_slices))
    cfg.slice_hw = int(ckpt.get("slice_hw", cfg.slice_hw))
    cfg.num_workers = min(cfg.num_workers, 4)

    loaders = make_mri_dataloaders(cfg)
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

    logger.info("\n===== TEST RESULTS (%s, %s) =====", split, checkpoint.name)
    logger.info("%-6s %8s %8s", "class", "AUROC", "F1")
    logger.info("%s", "-" * 24)
    for i, c in enumerate(class_names):
        logger.info("%-6s %8.4f %8.4f", c, m["per_class_auroc"][c], f1s[i])
    logger.info("%s", "-" * 24)
    logger.info("macro AUROC       : %.4f", m["macro_auroc"])
    logger.info("macro F1          : %.4f", mf1)
    logger.info("overall accuracy  : %.4f", m["accuracy"])
    logger.info("balanced accuracy : %.4f", m["balanced_accuracy"])
    logger.info("confusion matrix (rows=true, cols=pred) [%s]:", ", ".join(class_names))
    for i, c in enumerate(class_names):
        logger.info("  %-6s %s", c, cm[i].tolist())

    out = sub[["patient_id", "group", "label", "split"]].iloc[:n].copy()
    out = out.rename(columns={"label": "true_class"})
    out["pred_class"] = pred
    for i, c in enumerate(class_names):
        out[f"prob_{i}_{c}"] = probs[:, i]
    out.to_csv(MC.PREDICTIONS_CSV, index=False)
    logger.info("Per-patient predictions -> %s", MC.PREDICTIONS_CSV)

    return pd.DataFrame([{"class": c, "auroc": m["per_class_auroc"][c], "f1": f1s[i],
                          "support": int(cm[i].sum())} for i, c in enumerate(class_names)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ACDC MRI diagnosis checkpoint")
    p.add_argument("--checkpoint", type=Path, default=MC.CKPT_BEST)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    evaluate(args.checkpoint, args.split)
