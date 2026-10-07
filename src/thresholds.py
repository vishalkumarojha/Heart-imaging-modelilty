"""Decision-threshold policies for the frozen classifier.

Five named, deterministic policies, each fitted **on validation predictions
only** and then frozen for test/external use:

    fixed                    τ = 0.50 (the baseline operating point)
    f1_optimal               τ maximising F1
    youden                   τ maximising Youden's J = sensitivity + specificity − 1
    sensitivity_constrained  largest τ with sensitivity ≥ target (default 0.90)
                             → maximises specificity subject to the constraint
    precision_constrained    τ with the highest recall among those achieving
                             precision ≥ target (default 0.50)
                             → if no threshold reaches the target, the policy
                               falls back to the achievable maximum precision
                               and reports constraint_met = False

Every return value carries the operating characteristics measured *at* the
chosen threshold, so a threshold record is self-contained.

Fitting on test/external data is refused by `fit_thresholds` (same rule as
calibration) — see `docs/PROJECT_SCOPE.md` §4.

    from src.thresholds import find_threshold, fit_thresholds
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config as C
from .metrics import binary_metrics

POLICIES: tuple = C.THRESHOLD_POLICIES
POLICY_ALIASES = {
    "fixed": "fixed", "0.5": "fixed", "default": "fixed",
    "f1": "f1_optimal", "f1_optimal": "f1_optimal",
    "youden": "youden", "youden_j": "youden",
    "sensitivity": "sensitivity_constrained",
    "sensitivity_constrained": "sensitivity_constrained",
    "recall_constrained": "sensitivity_constrained",
    "precision": "precision_constrained",
    "precision_constrained": "precision_constrained",
}


def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _candidates(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Distinct score values → sorted descending, plus +inf style sentinel.

    Using the observed scores as candidate thresholds makes every policy
    independent of an arbitrary grid and guarantees a threshold that reproduces
    the corresponding confusion matrix exactly.
    """
    scores = np.unique(y_prob)
    # a threshold just above the max predicts nothing (useful for edge cases);
    # thresholds are still capped at 1.0 because scores are probabilities.
    return scores


# --------------------------------------------------------------------------- #
# Individual policies
# --------------------------------------------------------------------------- #
def threshold_fixed(_: np.ndarray, __: np.ndarray, value: float = 0.5) -> Dict[str, object]:
    return {"threshold": float(value), "constraint_met": None, "target": float(value)}


def threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    """τ maximising F1; ties broken towards the **larger** τ (fewer false positives)."""
    y_true, y_prob = _as1d(y_true), _as1d(y_prob)
    best = {"threshold": 0.5, "f1": -1.0}
    for tau in _candidates(y_true, y_prob):
        m = binary_metrics(y_true, y_prob, float(tau))
        if m["f1"] > best["f1"] + 1e-12 or (
            abs(m["f1"] - best["f1"]) <= 1e-12 and tau > best["threshold"]
        ):
            best = {"threshold": float(tau), "f1": m["f1"]}
    return {"threshold": best["threshold"], "constraint_met": None, "target": None}


def threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    """τ maximising J = sensitivity + specificity − 1; ties → larger τ."""
    y_true, y_prob = _as1d(y_true), _as1d(y_prob)
    best = {"threshold": 0.5, "j": -np.inf}
    for tau in _candidates(y_true, y_prob):
        m = binary_metrics(y_true, y_prob, float(tau))
        if not np.isfinite(m["sensitivity"]) or not np.isfinite(m["specificity"]):
            continue
        j = m["sensitivity"] + m["specificity"] - 1.0
        if j > best["j"] + 1e-12 or (abs(j - best["j"]) <= 1e-12 and tau > best["threshold"]):
            best = {"threshold": float(tau), "j": float(j)}
    if best["j"] == -np.inf:  # single-class label → nothing to optimise
        return {"threshold": 0.5, "constraint_met": None, "target": None}
    return {"threshold": best["threshold"], "constraint_met": None, "target": None}


def threshold_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target: float = C.SENSITIVITY_TARGET
) -> Dict[str, object]:
    """Largest τ with sensitivity ≥ target (max specificity under the constraint)."""
    y_true, y_prob = _as1d(y_true), _as1d(y_prob)
    if y_true.sum() == 0:  # no positives → sensitivity undefined
        return {"threshold": 0.5, "constraint_met": False, "target": float(target),
                "reason": "no positive examples in the fitting split"}
    feasible = []
    for tau in _candidates(y_true, y_prob):
        m = binary_metrics(y_true, y_prob, float(tau))
        if np.isfinite(m["sensitivity"]) and m["sensitivity"] >= target - 1e-12:
            feasible.append((float(tau), m))
    if not feasible:
        # even predicting everything positive cannot reach the target (should not
        # happen when positives exist) → report honestly
        return {"threshold": 0.5, "constraint_met": False, "target": float(target),
                "reason": "target unreachable"}
    tau, m = max(feasible, key=lambda t: t[0])  # highest threshold = most specific
    return {"threshold": tau, "constraint_met": True, "target": float(target),
            "achieved_sensitivity": m["sensitivity"], "achieved_specificity": m["specificity"]}


def threshold_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target: float = C.PRECISION_TARGET
) -> Dict[str, object]:
    """Highest-recall τ with precision ≥ target; fallback = max achievable precision."""
    y_true, y_prob = _as1d(y_true), _as1d(y_prob)
    if y_true.sum() == 0:
        return {"threshold": 0.5, "constraint_met": False, "target": float(target),
                "reason": "no positive examples in the fitting split"}
    feasible, best_prec = [], (-1.0, 0.5)
    for tau in _candidates(y_true, y_prob):
        m = binary_metrics(y_true, y_prob, float(tau))
        if not np.isfinite(m["precision"]):
            continue
        if m["precision"] > best_prec[0]:
            best_prec = (m["precision"], float(tau))
        if m["precision"] >= target - 1e-12:
            feasible.append((float(tau), m))
    if feasible:
        tau, m = max(feasible, key=lambda t: t[1]["recall"])  # best recall ⇒ lowest τ
        return {"threshold": tau, "constraint_met": True, "target": float(target),
                "achieved_precision": m["precision"], "achieved_recall": m["recall"]}
    return {
        "threshold": best_prec[1], "constraint_met": False, "target": float(target),
        "achieved_precision": best_prec[0],
        "reason": f"no threshold reaches precision ≥ {target:g}; "
                  "fell back to the maximum achievable precision",
    }


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
_DISPATCH = {
    "fixed": lambda y, p, t, v: threshold_fixed(y, p, value=v["fixed"]),
    "f1_optimal": lambda y, p, t, v: threshold_f1(y, p),
    "youden": lambda y, p, t, v: threshold_youden(y, p),
    "sensitivity_constrained":
        lambda y, p, t, v: threshold_sensitivity(y, p, t["sensitivity_target"]),
    "precision_constrained":
        lambda y, p, t, v: threshold_precision(y, p, t["precision_target"]),
}


def find_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: str,
    *,
    sensitivity_target: float = C.SENSITIVITY_TARGET,
    precision_target: float = C.PRECISION_TARGET,
    fixed: float = C.DECISION_THRESHOLD,
) -> Dict[str, object]:
    """Fit one policy for one label and return a self-describing record."""
    key = POLICY_ALIASES.get(str(policy).lower())
    if key is None:
        raise ValueError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    targets = {"sensitivity_target": sensitivity_target,
               "precision_target": precision_target, "fixed": fixed}
    out = _DISPATCH[key](y_true, y_prob, targets, targets)  # type: ignore[operator]
    out.update({
        "policy": key,
        "fit_metrics": binary_metrics(y_true, y_prob, float(out["threshold"])),
        "fit_n": int(np.asarray(y_true).size),
        "fit_prevalence": float(_as1d(y_true).mean()) if np.asarray(y_true).size else float("nan"),
    })
    return out


def fit_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    policies: Sequence[str] = POLICIES,
    split: str = C.THRESHOLD_FIT_SPLIT,
    **targets,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Fit every policy for every label on one split.

    Returns {policy: {label: record}}. Refuses to fit on evaluation splits.
    """
    split_l = str(split).lower()
    if any(bad in split_l for bad in ("test", "external", "chexpert")):
        raise ValueError(
            f"refusing to fit thresholds on split '{split}': thresholds may only "
            "be derived from validation (or training) predictions"
        )
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    if y_true.shape[1] != len(label_names) or y_prob.shape[1] != len(label_names):
        raise ValueError("label columns do not match label_names")

    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for policy in policies:
        per_label: Dict[str, Dict[str, object]] = {}
        for i, name in enumerate(label_names):
            rec = find_threshold(y_true[:, i], y_prob[:, i], policy, **targets)
            rec["label"] = name
            rec["split"] = split
            per_label[name] = rec
        out[POLICY_ALIASES[str(policy).lower()]] = per_label
    return out


def apply_thresholds(
    y_prob: np.ndarray, thresholds: Dict[str, float], label_names: Sequence[str]
) -> np.ndarray:
    """Turn probabilities into 0/1 predictions with a per-label threshold map."""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    pred = np.zeros_like(y_prob, dtype=np.int64)
    for i, name in enumerate(label_names):
        pred[:, i] = (y_prob[:, i] >= float(thresholds[name])).astype(np.int64)
    return pred


def thresholds_to_json(fitted: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, object]:
    """Flatten `fit_thresholds` output for `write_json` (keeps fit metrics)."""
    return {"policies": fitted, "fit_split": {p: next(iter(v.values()))["split"]
                                              for p, v in fitted.items() if v}}
