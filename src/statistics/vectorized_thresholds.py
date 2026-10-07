"""Vectorized re-implementation of the *scalar* threshold policies.

`src.thresholds.find_threshold` loops over every distinct score and re-runs
`binary_metrics` per candidate, which is O(C) Python calls per fitted label.
That is fine once on validation but far too slow inside a 2000-rep patient
bootstrap.  This module computes every candidate's operating point in one
vectorized pass (sort + cumulative counts) and then reproduces the *same*
selection rule — including the tie-breaking — as the scalar dispatcher, so a
bootstrap refit is numerically identical to the frozen validation fit.

Contract
--------
For all five policies the returned record matches `find_threshold`'s schema
(policy, threshold, constraint_met, target, fit_metrics, fit_n,
fit_prevalence, label, split).  `assert_policy_equivalence` verifies on the
full validation split against the frozen `thresholds_val.json`,
`thresholds_calibrated_val.json` and `thresholds_logistic_val.json` within
1e-12, and the unit tests assert the scalar/vectorized agreement on synthetic
data with deliberately tied scores.

Score-handling notes that keep the two implementations aligned:
* candidate thresholds = the observed distinct score values (ascending),
  exactly `np.unique` in `src.thresholds._candidates`
* f1_optimal / youden ties → **larger** threshold  (fewer false positives)
* precision_constrained recall ties → **first in ascending order** (Python
  `max` returns the first maximal element)
* precision fallback = maximum achievable precision, ties → lowest threshold
* sensitivity_constrained = largest threshold satisfying the constraint

    from src.statistics.vectorized_thresholds import fit_policies_vectorized
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .. import config as C
from ..metrics import binary_metrics

_EPS = 1e-12


def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def candidate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """Operating characteristics at every distinct score (ascending candidate order).

    Returns parallel arrays aligned with `taus` (ascending unique scores):
    confusion counts, derived rates, and two helper masks.  O(n log n).
    """
    y = _as1d(y_true).astype(np.int64)
    p = _as1d(y_prob)
    taus = np.unique(p)                                  # ascending
    if taus.size == 0:
        return {"threshold": taus, "tp": np.array([], dtype=np.int64), "fp": np.array([], dtype=np.int64),
                "tn": np.array([], dtype=np.int64), "fn": np.array([], dtype=np.int64),
                "accuracy": np.array([]), "precision": np.array([]), "recall": np.array([]),
                "specificity": np.array([]), "f1": np.array([]), "sensitivity": np.array([])}
    # ascending score order; run k = the (tied) rows with score == taus[k]
    order = np.argsort(p, kind="stable")
    ys = y[order]
    run_start = np.concatenate(([0], np.cumsum(np.unique(p[order], return_counts=True)[1])[:-1]))
    P = int(ys.sum())
    N = int(ys.size) - P

    cum_y = np.cumsum(ys)
    pos_below = np.where(run_start > 0, cum_y[run_start - 1], 0)   # positives with score < tau
    tp = P - pos_below                                            # score >= tau
    fn = pos_below
    neg_below = run_start - pos_below                             # negatives with score < tau
    tn = neg_below
    fp = N - tn

    denom_tp_fp = tp + fp
    denom_tp_fn = tp + fn
    denom_tn_fp = tn + fp
    denom_f1 = 2 * tp + fp + fn

    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = (tp + tn) / np.maximum(tp + tn + fp + fn, 1)
        precision = np.where(denom_tp_fp > 0, tp / np.maximum(denom_tp_fp, 1), np.nan)
        recall = np.where(denom_tp_fn > 0, tp / np.maximum(denom_tp_fn, 1), np.nan)
        specificity = np.where(denom_tn_fp > 0, tn / np.maximum(denom_tn_fp, 1), np.nan)
        f1 = np.where(denom_f1 > 0, 2 * tp / np.maximum(denom_f1, 1), np.nan)

    return {
        "threshold": taus,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "sensitivity": recall, "specificity": specificity,
        "f1": f1,
    }


def _largest_within(metric: np.ndarray, band: float = _EPS) -> Optional[int]:
    """Index of the largest threshold whose metric is within `band` of the max."""
    finite = np.flatnonzero(np.isfinite(metric))
    if finite.size == 0:
        return None
    best = np.nanmax(metric)
    keep = np.flatnonzero(metric >= best - band)
    return int(keep[-1]) if keep.size else None


def select_policy(
    m: Dict[str, np.ndarray],
    policy: str,
    *,
    sensitivity_target: float = C.SENSITIVITY_TARGET,
    precision_target: float = C.PRECISION_TARGET,
    fixed: float = C.DECISION_THRESHOLD,
) -> Dict[str, object]:
    """Pick the operating point for one policy; returns {threshold, constraint_met, target, ...}."""
    if policy not in C.THRESHOLD_POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if policy == "fixed":
        return {"threshold": float(fixed), "constraint_met": None, "target": float(fixed)}
    if m["threshold"].size == 0:
        return {"threshold": 0.5, "constraint_met": None, "target": None}
    if policy == "f1_optimal":
        i = _largest_within(m["f1"])
        return {"threshold": float(m["threshold"][i] if i is not None else 0.5),
                "constraint_met": None, "target": None}
    if policy == "youden":
        j = m["sensitivity"] + m["specificity"] - 1.0
        i = _largest_within(j)
        return {"threshold": float(m["threshold"][i] if i is not None else 0.5),
                "constraint_met": None, "target": None}

    if policy == "sensitivity_constrained":
        sens = m["sensitivity"]
        feasible = np.flatnonzero(np.isfinite(sens) & (sens >= sensitivity_target - _EPS))
        if feasible.size == 0:
            return {"threshold": 0.5, "constraint_met": False,
                    "target": float(sensitivity_target), "reason": "target unreachable"}
        i = int(feasible[-1])           # largest threshold among feasible = most specific
        return {"threshold": float(m["threshold"][i]), "constraint_met": True,
                "target": float(sensitivity_target),
                "achieved_sensitivity": float(sens[i]),
                "achieved_specificity": float(m["specificity"][i])}

    if policy == "precision_constrained":
        prec = m["precision"]
        feasible = np.flatnonzero(np.isfinite(prec) & (prec >= precision_target - _EPS))
        if feasible.size:
            rec = m["recall"][feasible]
            best = int(np.argmax(rec))                  # first max in ascending order
            i = int(feasible[best])
            return {"threshold": float(m["threshold"][i]), "constraint_met": True,
                    "target": float(precision_target),
                    "achieved_precision": float(prec[i]),
                    "achieved_recall": float(m["recall"][i])}
        finite = np.flatnonzero(np.isfinite(prec))
        if finite.size == 0:
            return {"threshold": 0.5, "constraint_met": False,
                    "target": float(precision_target),
                    "reason": "no threshold yields a defined precision"}
        best = int(finite[np.argmax(prec[finite])])     # fallback: max achievable precision
        return {"threshold": float(m["threshold"][best]), "constraint_met": False,
                "target": float(precision_target),
                "achieved_precision": float(prec[best]),
                "reason": "no threshold reaches the precision target; "
                          "fell back to the maximum achievable precision"}
    raise AssertionError(f"unhandled policy {policy!r}")


def find_threshold_vectorized(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: str,
    **targets,
) -> Dict[str, object]:
    """One policy / one label; record schema matches `src.thresholds.find_threshold`."""
    if policy == "fixed":
        tau = float(targets.get("fixed", C.DECISION_THRESHOLD))
        return {"threshold": tau, "constraint_met": None,
                "target": tau, "policy": policy,
                "fit_metrics": binary_metrics(y_true, y_prob, tau),
                "fit_n": int(np.asarray(y_true).size),
                "fit_prevalence": float(_as1d(y_true).mean()) if np.asarray(y_true).size else float("nan")}
    rec = select_policy(candidate_metrics(y_true, y_prob), policy, **targets)
    rec["policy"] = policy
    rec["fit_metrics"] = binary_metrics(y_true, y_prob, float(rec["threshold"]))
    rec["fit_n"] = int(np.asarray(y_true).size)
    rec["fit_prevalence"] = float(_as1d(y_true).mean()) if np.asarray(y_true).size else float("nan")
    return rec


def fit_policies_vectorized(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    policies: Sequence[str] = C.THRESHOLD_POLICIES,
    split: str = C.THRESHOLD_FIT_SPLIT,
    **targets,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Same shape as `src.thresholds.fit_thresholds` (policy -> label -> record)."""
    split_l = str(split).lower()
    if any(bad in split_l for bad in ("test", "external", "chexpert")):
        raise ValueError(
            f"refusing to fit thresholds on split '{split}': thresholds may only "
            "be derived from validation (or training) predictions"
        )
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if y.shape[1] != len(label_names) or p.shape[1] != len(label_names):
        raise ValueError("label columns do not match label_names")
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for policy in policies:
        per_label: Dict[str, Dict[str, object]] = {}
        for i, name in enumerate(label_names):
            rec = find_threshold_vectorized(y[:, i], p[:, i], policy, **targets)
            rec["label"] = name
            rec["split"] = split
            per_label[name] = rec
        out[policy] = per_label
    return out


def assert_policy_equivalence(
    y_val: np.ndarray,
    prob_val: Dict[str, np.ndarray],
    label_names: Sequence[str],
    atol: float = 1e-9,
) -> Dict[str, object]:
    """Verify vectorized refit == frozen val fits for every variant/policy/label.

    `prob_val` maps variant -> (N, C) probability matrix.  Returns a summary
    dict and raises ValueError on any mismatch beyond `atol` — the vectorized
    fitter must reproduce the frozen validation parameters bit-for-bit.
    """
    from .data import frozen_threshold_values

    checks: Dict[str, object] = {}
    for variant, p in prob_val.items():
        for policy in C.THRESHOLD_POLICIES:
            refit = fit_policies_vectorized(y_val, np.asarray(p), label_names,
                                            policies=(policy,), split="val")
            frozen = frozen_threshold_values(variant)[policy]
            for i, name in enumerate(label_names):
                got = float(refit[policy][name]["threshold"])
                want = float(frozen[name])
                if abs(got - want) > atol:
                    raise ValueError(
                        f"vectorized threshold mismatch: variant={variant} "
                        f"policy={policy} label={name}: got {got:.12g} "
                        f"vs frozen {want:.12g} (Δ={abs(got - want):.2e})"
                    )
                checks.setdefault(variant, set()).add(policy)
                checks[f"{variant}/{policy}/{name}"] = float(got)
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in checks.items()}