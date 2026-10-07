"""Patient-level (cluster) bootstrap for the primary effect-size statistics.

Replaces the legacy *image-level* resampling (`src.uncertainty`) with the
statistically defensible unit: **the patient**.  Images from one patient are
correlated (a session, not independent draws), so resampling rows violates the
exchangeability assumption behind percentile CIs.  Resampling patients (a block
bootstrap over each patient's full image bundle) preserves that dependence, and
every calibration arm is scored on the *same* resample, which makes the
confidence intervals *paired* across arms.

Artifacts
---------
outputs/final_results/confidence_intervals_patient.csv     percentile CIs
outputs/final_results/arm_differences.csv                  paired ΔA/B, ΔF1/… etc.
outputs/final_results/confidence_intervals_patient.meta.json

Method notes (reported in the paper):
* percentile bootstrap, resampling unit = patient, 5000 resamples (>= PATIENT_
  BOOTSTRAP_MIN_ITERATIONS), fixed seed 42 -> bit-reproducible
* within a resample every patient's images enter the cohort *in full*
* AUROC / AUPRC are invariant under every monotone calibration map we use
  (temperature, logistic-with-positive-slope), so a single pair of CI rows is
  emitted per label ('both (monotone-invariant)')
* threshold metrics use the FROZEN validation thresholds per (variant, policy)
* calibration metrics (Brier / NLL / ECE) use FROZEN reliability binning made on
  the full test split
* a resample that happens to lose a whole outcome class has an undefined AUROC /
  AUPRC -> NaN for that resample (nanpercentile below)
* precision with zero predicted positives is 0.0 (division-by-zero underflow)

Run:  python -m src.statistics.bootstrap
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import _bin_edges
from ..metrics import auprc, auroc
from ..reproducibility import write_json
from ..utils import setup_logging
from .data import (ARMS, PAIRED_COMPARISONS, POLICIES, VARIANTS, arm_thresholds,
                   frozen_threshold_values, load_test_predictions,
                   variant_probs)

logger = setup_logging()

METHOD = "patient-level cluster bootstrap (resampling unit = patient; percentile CI)"
THIRD_ORDER_METRICS = ("accuracy", "precision", "recall", "specificity", "f1")


class PatientResampler:
    """Block resampler over patient clusters.  Each patient's images stay whole."""

    def __init__(self, df: pd.DataFrame, id_col: str = "Patient ID"):
        clusters: List[np.ndarray] = []
        for _, sub in df.groupby(id_col, sort=False):
            clusters.append(sub.index.to_numpy(dtype=np.int64))
        self.sizes = np.array([c.size for c in clusters], dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.sizes)))[:-1]
        self.expanded = np.concatenate(clusters) if clusters else np.empty(0, np.int64)
        self.n_patients = int(len(clusters))

    def sample_rows(self, rng: np.random.Generator) -> np.ndarray:
        """All rows of `n_patients` bootstrap patients (with replacement), vectorized.

        sel         patients (with replacement)
        starts      start of each selected patient's block in `expanded`
        within      0..size-1 offset inside each block
        pos         final expanded-row position per resampled row
        """
        sel = rng.integers(0, self.n_patients, size=self.n_patients)
        sizes = self.sizes[sel]
        total = int(sizes.sum())
        starts = self.starts[sel]
        prefix = np.cumsum(sizes) - sizes
        within = np.arange(total, dtype=np.int64) - np.repeat(prefix, sizes)
        pos = np.repeat(starts, sizes) + within
        return self.expanded[pos]


# --------------------------------------------------------------------------- #
# Point-metric helpers
# --------------------------------------------------------------------------- #
def _confusion(ry: np.ndarray, p: np.ndarray, tau: float):
    y = ry.astype(np.int64)
    pred = (p >= tau)
    tp = int((y & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((y == 1) & ~pred).sum())
    return tp, fp, fn, tn


def _from_confusion(tp: int, fp: int, fn: int, tn: int, metric: str) -> float:
    if metric == "accuracy":
        return float((tp + tn) / max(tp + tn + fp + fn, 1))
    if metric == "precision":
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    if metric == "recall" or metric == "sensitivity":
        return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    if metric == "specificity":
        return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    if metric == "f1":
        den = 2 * tp + fp + fn
        return float(2 * tp / den) if den > 0 else float("nan")
    raise ValueError(f"unknown confusion-derived metric {metric!r}")


def _unique2(y: np.ndarray) -> bool:
    return bool(np.unique(y).size >= 2)


# --------------------------------------------------------------------------- #
# Rows for the two CSV artifacts
# --------------------------------------------------------------------------- #
def _ci_row(label: str, variant: str, policy: str, metric: str,
            point: float, lo: float, hi: float) -> dict:
    return {
        "split": "test",
        "class": label,
        "calibration": variant,
        "threshold_policy": policy,
        "metric": metric,
        "point_estimate": _r(point),
        "ci_lower": _r(lo),
        "ci_upper": _r(hi),
        "n_bootstraps": int(C.PATIENT_BOOTSTRAP_ITERATIONS),
        "confidence_level": C.PATIENT_BOOTSTRAP_CI_LEVEL,
        "method": METHOD,
        "rng_seed": C.PATIENT_BOOTSTRAP_SEED,
    }


def _diff_row(label: str, arm_a: str, arm_b: str, metric: str,
              point: float, lo: float, hi: float) -> dict:
    return {
        "split": "test",
        "class": label,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": metric,
        "delta_point_estimate": _r(point),
        "ci_lower": _r(lo),
        "ci_upper": _r(hi),
        "n_bootstraps": int(C.PATIENT_BOOTSTRAP_ITERATIONS),
        "confidence_level": C.PATIENT_BOOTSTRAP_CI_LEVEL,
        "method": METHOD,
        "rng_seed": C.PATIENT_BOOTSTRAP_SEED,
    }


def _r(x: float) -> float:
    return round(float(x), 6)


def percentile_ci(samples: np.ndarray, alpha: float) -> tuple:
    q = 100.0 * (1.0 - alpha)
    lo = float(np.nanpercentile(samples, 100.0 - q))
    hi = float(np.nanpercentile(samples, q))
    return lo, hi


# --------------------------------------------------------------------------- #
def run_patient_bootstrap(
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple:
    if iterations is None:
        iterations = int(C.PATIENT_BOOTSTRAP_ITERATIONS)
    if iterations < int(C.PATIENT_BOOTSTRAP_MIN_ITERATIONS):
        raise ValueError(
            f"{iterations} < PATIENT_BOOTSTRAP_MIN_ITERATIONS "
            f"({C.PATIENT_BOOTSTRAP_MIN_ITERATIONS}) — below the documented minimum"
        )
    if seed is None:
        seed = int(C.PATIENT_BOOTSTRAP_SEED)

    df = load_test_predictions()
    for label in C.TARGET_LABELS:
        for variant in VARIANTS:
            frozen_threshold_values(variant)      # fail fast if missing
    resampler = PatientResampler(df)
    rng = np.random.default_rng(seed)
    alpha = 1.0 - C.PATIENT_BOOTSTRAP_CI_LEVEL

    thresholds = {variant: frozen_threshold_values(variant) for variant in VARIANTS}

    ci_rows: List[dict] = []
    diff_rows: List[dict] = []

    for label in C.TARGET_LABELS:
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        ps = {variant: variant_probs(df, label, variant) for variant in VARIANTS}
        edges = {variant: _bin_edges(C.CALIBRATION_BINS, C.CALIBRATION_STRATEGY, ps[variant])
                 for variant in VARIANTS}

        # per-arm metric series that feed the paired deltas
        arm_samples: Dict[tuple, np.ndarray] = {}

        for b in range(iterations):
            rows = resampler.sample_rows(rng)
            ry = y[rows]

            # (1) rank metrics — monotone-invariant, computed once on raw probs
            if _unique2(ry):
                ra, rp_ = float(auroc(ry, ps["raw"][rows])), float(auprc(ry, ps["raw"][rows]))
            else:
                ra, rp_ = float("nan"), float("nan")

            for variant in VARIANTS:
                p_row = ps[variant][rows]
                # (2) calibration metrics — frozen binning
                brier = float(np.nanmean((p_row - ry) ** 2))
                nll = float(-np.nanmean(ry * np.log(_clip(p_row)) + (1.0 - ry) * np.log(_clip(1.0 - p_row))))
                ece = ece_from_edges(ry, p_row, edges[variant])
                for metric, val in (("brier_score", brier),
                                    ("negative_log_likelihood", nll),
                                    ("ece", ece)):
                    key = (variant, "calibration", metric)
                    if key not in arm_samples:
                        arm_samples[key] = np.full(iterations, np.nan)
                    arm_samples[key][b] = val

                # (3) threshold metrics — frozen validation thresholds
                for policy in POLICIES:
                    tau = float(thresholds[variant][policy][label])
                    tp, fp, fn, tn = _confusion(ry, p_row, tau)
                    for metric in THIRD_ORDER_METRICS:
                        val = _from_confusion(tp, fp, fn, tn, metric)
                        key = (variant, policy, metric)
                        if key not in arm_samples:
                            arm_samples[key] = np.full(iterations, np.nan)
                        arm_samples[key][b] = val

            # fold rank metrics into bootstrapped series under synthetic keys
            for metric, val in (("auroc", ra), ("auprc", rp_)):
                key = ("invariant", "n/a", metric)
                if key not in arm_samples:
                    arm_samples[key] = np.full(iterations, np.nan)
                arm_samples[key][b] = val

        # ---- CI rows -------------------------------------------------------
        for (variant, policy, metric), samples in arm_samples.items():
            if metric in ("auroc", "auprc"):
                point = float(auroc(y, ps["raw"])) if metric == "auroc" else float(auprc(y, ps["raw"]))
                lo, hi = percentile_ci(samples, alpha)
                ci_rows.append(_ci_row(label, "both (monotone-invariant)", policy, metric,
                                       point, lo, hi))
                continue
            if metric in ("brier_score", "negative_log_likelihood", "ece"):
                point = {
                    "brier_score": float(np.mean((ps[variant] - y) ** 2)),
                    "negative_log_likelihood": float(-np.mean(
                        y * np.log(_clip(ps[variant])) + (1.0 - y) * np.log(_clip(1.0 - ps[variant])))),
                    "ece": ece_from_edges(y, ps[variant], edges[variant]),
                }[metric]
                _lo, _hi = percentile_ci(samples, alpha)
                ci_rows.append(_ci_row(label, variant, "n/a", metric, point, _lo, _hi))
                continue
            tau = float(thresholds[variant][policy][label])
            tp, fp, fn, tn = _confusion(y, ps[variant], tau)
            point = _from_confusion(tp, fp, fn, tn, metric)
            lo, hi = percentile_ci(samples, alpha)
            ci_rows.append(_ci_row(label, variant, policy, metric, point, lo, hi))

        # ---- paired deltas between arms -----------------------------------
        arm_point: Dict[str, Dict[str, float]] = {}
        for arm, (variant, policy) in ARMS.items():
            arm_point[arm] = {}
            for metric in THIRD_ORDER_METRICS:
                tau = float(thresholds[variant][policy][label])
                tp, fp, fn, tn = _confusion(y, ps[variant], tau)
                arm_point[arm][metric] = _from_confusion(tp, fp, fn, tn, metric)

        for arm_a, arm_b in PAIRED_COMPARISONS:
            for metric in THIRD_ORDER_METRICS:
                a_key = (ARMS[arm_a][0], ARMS[arm_a][1], metric)
                b_key = (ARMS[arm_b][0], ARMS[arm_b][1], metric)
                if b_key not in arm_samples or a_key not in arm_samples:
                    continue
                delta = arm_samples[b_key] - arm_samples[a_key]
                point = arm_point[arm_b][metric] - arm_point[arm_a][metric]
                lo, hi = percentile_ci(delta, alpha)
                diff_rows.append(_diff_row(label, arm_a, arm_b, metric, point, lo, hi))

    out = pd.DataFrame(ci_rows)
    out.to_csv(C.CONFIDENCE_INTERVALS_PATIENT_CSV, index=False)
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(C.ARM_DIFFERENCES_CSV, index=False)
    logger.info("confidence_intervals_patient.csv -> %d rows", len(out))
    logger.info("arm_differences.csv               -> %d rows", len(diffs))

    meta = {
        "split": "test",
        "n_images": int(len(df)),
        "n_patients": int(resampler.n_patients),
        "n_bootstraps": iterations,
        "confidence_level": C.PATIENT_BOOTSTRAP_CI_LEVEL,
        "rng_seed": seed,
        "resampling_unit": "patient (cluster / block bootstrap)",
        "method": METHOD,
        "variants": list(VARIANTS),
        "policies": list(POLICIES),
        "arms": {k: {"variant": v, "policy": p} for k, (v, p) in ARMS.items()},
        "paired_comparisons": list(PAIRED_COMPARISONS),
        "note": ("AUROC/AUPRC are monotone-invariant so one pair of rows per label; "
                 "precision := 0 when a resample has no predicted positives; a "
                 "resample that loses an entire outcome class contributes NaN to "
                 "the AUROC/AUPRC distribution.  Paired differences are computed "
                 "on the identical patient resamples."),
    }
    write_json(C.FINAL_RESULTS_DIR / "confidence_intervals_patient.meta.json", meta)
    return C.CONFIDENCE_INTERVALS_PATIENT_CSV, C.ARM_DIFFERENCES_CSV


def _clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def ece_from_edges(y_true: np.ndarray, probs: np.ndarray, edges: np.ndarray) -> float:
    idx = np.digitize(probs, edges[1:-1], right=False)
    counts = np.bincount(idx, minlength=C.CALIBRATION_BINS).astype(np.float64)
    acc = np.bincount(idx, weights=y_true, minlength=C.CALIBRATION_BINS)
    conf = np.bincount(idx, weights=probs, minlength=C.CALIBRATION_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        coeff = counts / counts.sum()
        gap = np.abs(acc / np.maximum(counts, 1) - conf / np.maximum(counts, 1))
    return float(np.sum(coeff * np.where(counts > 0, gap, 0.0)))


if __name__ == "__main__":
    run_patient_bootstrap()