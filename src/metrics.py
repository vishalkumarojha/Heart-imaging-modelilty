"""Classification metrics for the calibration / threshold research.

Thin, explicit wrappers around scikit-learn plus the existing helpers in
`src.utils`, so every reported number can be traced to one function:

    auroc(y_true, y_prob)         discrimination (threshold-free)
    auprc(y_true, y_prob)         precision-recall summary (threshold-free,
                                  prevalence-sensitive — the honest companion
                                  to AUROC under class imbalance)
    binary_metrics(y_true, y_prob, threshold)   operating point: confusion
                                  counts, accuracy, precision, sensitivity,
                                  specificity, F1
    multilabel_report(...)        per-label + macro table for all of the above

Conventions
-----------
* Inputs are float probabilities in [0, 1] (post-sigmoid, optionally calibrated)
  and integer 0/1 labels — calibration never changes AUROC/AUPRC because
  temperature scaling is monotone, and this module is where that invariant is
  asserted by the tests.
* Undefined metrics (single-class y_true, no predicted positives, ...) return
  NaN instead of raising, so a full report can always be produced.
* Phase 4 adds `ece`, `brier` and `reliability_bins` here.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .utils import binary_rates_from_confusion, multilabel_auroc


def _as_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(np.float64).ravel()


def _guard(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")


# --------------------------------------------------------------------------- #
# Threshold-free metrics
# --------------------------------------------------------------------------- #
def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC AUC for one binary label. NaN if only one class is present."""
    from sklearn.metrics import roc_auc_score

    y_true, y_prob = _as_1d(y_true), _as_1d(y_prob)
    _guard(y_true, y_prob)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    NaN if the label has no positives (undefined) — reported as NaN, never as 0.
    """
    from sklearn.metrics import average_precision_score

    y_true, y_prob = _as_1d(y_true), _as_1d(y_prob)
    _guard(y_true, y_prob)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


# --------------------------------------------------------------------------- #
# Operating-point metrics
# --------------------------------------------------------------------------- #
def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Confusion counts + derived rates for one label at one threshold."""
    y_true, y_prob = _as_1d(y_true), _as_1d(y_prob)
    _guard(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    rates = binary_rates_from_confusion(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "support_pos": int(rates["tp"] + rates["fn"]),
        "support_neg": int(rates["tn"] + rates["fp"]),
        "tp": int(rates["tp"]), "tn": int(rates["tn"]),
        "fp": int(rates["fp"]), "fn": int(rates["fn"]),
        "accuracy": float(rates["accuracy"]),
        "precision": float(rates["precision"]),
        "recall": float(rates["sensitivity"]),
        "sensitivity": float(rates["sensitivity"]),
        "specificity": float(rates["specificity"]),
        "f1": float(rates["f1"]),
        "prevalence": float(y_true.mean()) if y_true.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Multi-label reporting
# --------------------------------------------------------------------------- #
def multilabel_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Per-label AUROC / AUPRC / operating metrics + macro means.

    Returns {"per_label": {name: {...}}, "macro": {...}}.  Macro means are
    computed over the *defined* values only (a NaN never silently becomes 0).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    _guard(y_true, y_prob)
    if y_true.shape[1] != len(label_names):
        raise ValueError(
            f"{y_true.shape[1]} label columns but {len(label_names)} names"
        )

    per_label: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(label_names):
        m = binary_metrics(y_true[:, i], y_prob[:, i], threshold)
        m["auroc"] = auroc(y_true[:, i], y_prob[:, i])
        m["auprc"] = auprc(y_true[:, i], y_prob[:, i])
        per_label[name] = m

    auroc_map = multilabel_auroc(y_true, y_prob, label_names)
    per_label["mean"] = {"auroc": auroc_map["mean"], "auprc": _nanmean(
        [per_label[n]["auprc"] for n in label_names]
    )}
    return {"threshold": float(threshold), "per_label": per_label, "macro": per_label["mean"]}


def _nanmean(values: Sequence[float]) -> float:
    vals = [v for v in values if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def report_to_rows(report: Dict[str, object], label_names: Sequence[str]) -> List[Dict[str, object]]:
    """Flatten `multilabel_report` into one row per label (+ mean) for CSV."""
    rows: List[Dict[str, object]] = []
    per_label = report["per_label"]  # type: ignore[index]
    for name in [*label_names, "mean"]:
        if name not in per_label:
            continue
        m = dict(per_label[name])
        m["label"] = name
        rows.append(m)
    return rows
