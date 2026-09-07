"""Modality-agnostic helpers: metrics, seeding, logging.

Nothing in here imports the X-ray dataset or model, so it can be reused by the
ECHO / MRI branches and the fusion trainer later.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, Iterable, Sequence

import numpy as np

logger = logging.getLogger("capstone")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python / numpy / torch RNGs.

    torch is imported lazily so `utils` stays light for scripts that only need
    the metric helpers.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ModuleNotFoundError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def multilabel_auroc(
    y_true: np.ndarray, y_score: np.ndarray, label_names: Sequence[str]
) -> Dict[str, float]:
    """Per-label AUROC + macro mean.

    Robust to a label having only one class present in `y_true` (can happen in
    --quick_test or tiny val sets): that label's AUROC is reported as NaN and
    excluded from the mean.
    """
    from sklearn.metrics import roc_auc_score

    y_true = _as_2d(y_true)
    y_score = _as_2d(y_score)
    out: Dict[str, float] = {}
    valid = []
    for i, name in enumerate(label_names):
        col = y_true[:, i]
        if col.min() == col.max():
            out[name] = float("nan")
            continue
        score = float(roc_auc_score(col, y_score[:, i]))
        out[name] = score
        valid.append(score)
    out["mean"] = float(np.mean(valid)) if valid else float("nan")
    return out


def binary_rates_from_confusion(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """sensitivity / specificity / precision / f1 / accuracy for one binary label."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall / TPR
    spec = tn / (tn + fp) if (tn + fp) else float("nan")   # TNR
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) not in (0, float("nan")) and (prec + sens) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "precision": prec, "f1": f1, "accuracy": acc,
    }


# --------------------------------------------------------------------------- #
# Label distribution reporting (expect heavy imbalance)
# --------------------------------------------------------------------------- #
def log_label_distribution(
    labels: np.ndarray, label_names: Sequence[str], split_name: str = ""
) -> Dict[str, Dict[str, float]]:
    """labels: (N, C) 0/1 array. Prints a positives / prevalence table."""
    labels = _as_2d(labels)
    n = len(labels)
    header = f"Label distribution{f' [{split_name}]' if split_name else ''}  (N={n})"
    logger.info(header)
    logger.info("  %-16s %10s %10s %14s", "label", "positives", "negatives", "prevalence")
    stats: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(label_names):
        pos = int(labels[:, i].sum())
        neg = n - pos
        prev = pos / n if n else float("nan")
        pos_weight = neg / pos if pos else float("inf")
        logger.info("  %-16s %10d %10d %13.4f", name, pos, neg, prev)
        stats[name] = {"positives": pos, "negatives": neg, "prevalence": prev, "pos_weight": pos_weight}
    # co-occurrence of all target labels at once (both conditions present)
    if labels.shape[1] >= 2:
        both = int(np.all(labels == 1, axis=1).sum())
        logger.info("  (rows with ALL target labels positive: %d)", both)
    return stats


def format_metrics_table(rows: Iterable[Dict[str, object]], columns: Sequence[str]) -> str:
    """Tiny fixed-width table renderer for the evaluate.py summary."""
    rows = list(rows)
    widths = {c: max(len(c), *(len(f"{r.get(c, ''):}") for r in rows)) if rows else len(c) for c in columns}
    line = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows)
    return f"{line}\n{sep}\n{body}"
