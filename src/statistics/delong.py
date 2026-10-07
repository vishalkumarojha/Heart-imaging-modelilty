"""DeLong variance / CI / two-curve test for ROC AUC (Hanley–McNeil → DeLong).

Where DeLong is used in this upgrade and why (stated in the paper):
* single-curve SE + CI per label  -> `outputs/metrics/patient_stats/delong_auroc.json`
  (DeLong's placement-value covariance estimator is the standard, non-parametric
  choice; it avoids the Hanley–McNeil distributional assumptions).
* a two-curve DeLong test is *demonstrated* between the raw and calibrated
  scores of the same label — the honest point being that every calibration map
  we use (temperature, logistic with positive slope) is monotone, so the two
  curves coincide bit-for-bit:  ΔAUC = 0, the difference SE collapses, and the
  test is degenerate (z undefined / p = NaN) **by design**.  This is reported
  rather than hidden: no "calibration beats calibration on AUROC" claim is made.
* comparing AUCs *across* labels (Cardiomegaly vs Effusion) would be a
  different-outcome comparison and is deliberately not attempted with DeLong.

Placement values
----------------
V10(i)   = (# negative scores < D_i + ½·# negative scores == D_i) / n0   (i positive)
V01(j)   = (# positive scores > D_j + ½·# positive scores == D_j) / n1   (j negative)

θ = mean(V10) = mean(V01) = Mann-Whitney AUC (ties averaged = sklearn's AUC).

Single-curve variance (DeLong):
    var(θ) = S10 / n1 + S01 / n0,   S10 = Σ(V10−θ)²/(n1−1),  S01 = Σ(V01−θ)²/(n0−1)

Two correlated curves measured on the same patients add the co-variance term:
    cov(θa, θb) = S10_ab / n1 + S01_ab / n0
    z = (θa − θb) / √(var_a + var_b − 2·cov)

    from src.statistics.delong import delong_single_curve, delong_two_curves
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from scipy import stats

from .. import config as C
from ..metrics import auroc
from ..reproducibility import write_json
from ..utils import setup_logging
from .data import load_test_predictions, variant_probs

logger = setup_logging()


def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _placements(score: np.ndarray, y: np.ndarray) -> tuple:
    """(V10 for positives, V01 for negatives, theta). O(n log n), ties averaged."""
    score = _as1d(score)
    y = _as1d(y)
    pos = y == 1
    neg = y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    U, inv = np.unique(score, return_inverse=True)
    neg_eq = np.bincount(inv[neg], minlength=len(U)).astype(np.float64)
    pos_eq = np.bincount(inv[pos], minlength=len(U)).astype(np.float64)
    neg_lt = np.concatenate(([0.0], np.cumsum(neg_eq)))[:-1]          # #neg <  u
    pos_gt = n1 - np.cumsum(pos_eq)                                   # #pos >  u

    with np.errstate(invalid="ignore", divide="ignore"):
        v10 = (neg_lt[inv] + 0.5 * neg_eq[inv]) / n0
        v01 = (pos_gt[inv] + 0.5 * pos_eq[inv]) / n1
    theta = float(np.mean(v10[pos])) if n1 else float("nan")
    return v10[pos], v01[neg], theta


def delong_se(y: np.ndarray, v10: np.ndarray, v01: np.ndarray, theta: float) -> float:
    n1, n0 = int(v10.size), int(v01.size)
    if n1 < 2 or n0 < 2:
        return float("nan")
    s10 = float(np.sum((v10 - theta) ** 2) / (n1 - 1))
    s01 = float(np.sum((v01 - theta) ** 2) / (n0 - 1))
    var = s10 / n1 + s01 / n0
    return float(np.sqrt(var)) if var > 0 else float("nan")


def _ci(auc: float, se: float, level: float = C.PATIENT_BOOTSTRAP_CI_LEVEL) -> tuple:
    if np.isnan(auc) or np.isnan(se):
        return float("nan"), float("nan")
    z = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)
    lo = min(1.0, max(0.0, auc - z * se))
    hi = min(1.0, max(0.0, auc + z * se))
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
def delong_single_curve(score: np.ndarray, y: np.ndarray,
                        level: float = C.PATIENT_BOOTSTRAP_CI_LEVEL) -> Dict[str, float]:
    y = _as1d(y)
    if np.unique(y).size < 2:
        return {"auc": float("nan"), "se": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_pos": 0, "n_neg": 0}
    v10, v01, theta = _placements(score, y)
    se = delong_se(y, v10, v01, theta)
    lo, hi = _ci(theta, se, level)
    return {"auc": theta, "se": se, "ci_lower": lo, "ci_upper": hi,
            "n_pos": int(v10.size), "n_neg": int(v01.size)}


def delong_two_curves(score_a: np.ndarray, score_b: np.ndarray, y: np.ndarray,
                      level: float = C.PATIENT_BOOTSTRAP_CI_LEVEL) -> Dict[str, float]:
    """DeLong Z-test between two correlated ROC curves (same subjects)."""
    y = _as1d(y)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    out: Dict[str, float] = {"auc_a": float("nan"), "auc_b": float("nan"),
                             "se_a": float("nan"), "se_b": float("nan"),
                             "se_diff": float("nan"), "cov": float("nan"),
                             "z": float("nan"), "p": float("nan"),
                             "n_pos": n1, "n_neg": n0}
    if n1 < 2 or n0 < 2 or np.unique(y).size < 2:
        return out
    va, oa, ta = _placements(score_a, y)      # (v10_pos, v01_neg, auc)
    vb, ob, tb = _placements(score_b, y)
    out["auc_a"], out["auc_b"] = ta, tb
    out["se_a"] = delong_se(y, va, oa, ta)
    out["se_b"] = delong_se(y, vb, ob, tb)
    s10_ab = float(np.sum((va - ta) * (vb - tb)) / (n1 - 1))
    s01_ab = float(np.sum((oa - ta) * (ob - tb)) / (n0 - 1))
    cov = float(s10_ab / n1 + s01_ab / n0)
    out["cov"] = cov
    var_diff = out["se_a"] ** 2 + out["se_b"] ** 2 - 2.0 * cov
    if not np.isfinite(var_diff) or var_diff <= 0:
        # the two curves coincide (ΔAUC = 0, covariance ≈ variance) → degenerate
        return out
    se_diff = float(np.sqrt(var_diff))
    out["se_diff"] = se_diff
    z = (ta - tb) / se_diff
    out["z"] = float(z)
    out["p"] = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return out


# --------------------------------------------------------------------------- #
def run_delong() -> dict:
    df = load_test_predictions()
    per_label: Dict[str, Dict[str, float]] = {}
    demo: List[dict] = []
    for label in C.TARGET_LABELS:
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        raw = df[f"prob_{label}"].to_numpy(dtype=np.float64)
        cal = variant_probs(df, label, "calibrated")
        log = variant_probs(df, label, "logistic")
        per_label[label] = delong_single_curve(raw, y)
        # consistency: DeLong theta must equal sklearn's AUROC (defensive)
        skl = auroc(y, raw)
        if not np.isnan(per_label[label]["auc"]) and abs(per_label[label]["auc"] - skl) > 1e-12:
            logger.warning("DeLong AUC diverge for %s: %r vs sklearn %r", label,
                           per_label[label]["auc"], skl)
        for name, sc in (("raw", raw), ("calibrated", cal), ("logistic", log)):
            demo.append(dict(label=label, arm=name, **delong_single_curve(sc, y)))
        demo.append(dict(label=label, test="raw_vs_calibrated",
                         **delong_two_curves(raw, cal, y)))
        demo.append(dict(label=label, test="raw_vs_logistic",
                         **delong_two_curves(raw, log, y)))

    payload = {
        "generated_utc": None,
        "split": "test",
        "method": ("DeLong placement-value covariance: single-curve SE/CI per "
                   "label; two-curve Z-test shown only to document the monotone "
                   "degeneracy of AUROC under calibration"),
        "confidence_level": C.PATIENT_BOOTSTRAP_CI_LEVEL,
        "per_label": per_label,
        "single_curve_by_variant": demo,
        "monotone_invariance_note": (
            "Temperature and logistic (slope>0) calibration are strictly monotone, "
            "so AUC(raw) == AUC(calibrated) == AUC(logistic) exactly.  The paired "
            "DeLong ΔAUC therefore has variance 0 and no p-value: calibration "
            "cannot and does not claim discrimination gains."),
        "informed_consent": (
            "no cross-label AUROC comparisons are performed with DeLong, since "
            "the two labels are different outcomes."),
    }
    path = C.PATIENT_STATS_DIR / "delong_auroc.json"
    write_json(path, payload)
    logger.info("delong_auroc.json -> per-label AUROC SE/CI (%d rows in demo)", len(demo))
    return payload


if __name__ == "__main__":
    run_delong()