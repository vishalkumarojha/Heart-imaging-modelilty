"""Bootstrap confidence intervals for the primary effect-size statistics.

Produces `outputs/final_results/confidence_intervals.csv`. Every point estimate
re-derives the metrics from the FROZEN per-image test predictions
(`outputs/predictions/raw/test__*.csv`) and the FROZEN validation thresholds
(`outputs/metrics/thresholds/thresholds_val.json`), so the CI row for a metric
is guaranteed to be bracketing the exact number reported in master_results.csv.

Method (documented for the paper):
    * percentile bootstrap, resampling unit = test image (row), 2000 resamples
    * fixed RNG seed 42 -> bit-reproducible
    * AUROC / AUPRC are temperature-invariant, so a single pair of CI rows is
      written per label (variant = 'both (temperature-invariant)')
    * threshold metrics (accuracy/precision/recall/specificity/F1) are computed
      with the frozen validation thresholds per (variant, policy)
    * ECE / Brier are bootstrapped with the *frozen* reliability binning (made
      on the full test split), i.e. binning is a fixed transform
    * precision in a bootstrap sample with zero predicted positives is 0 for
      that sample (division by zero underflow), which the percentile CI reflects

Run:  python -m src.uncertainty
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .calibration import _bin_edges
from .metrics import auprc, auroc, binary_metrics
from .reproducibility import write_json
from .utils import setup_logging

logger = setup_logging()

POLICIES = ("fixed", "f1_optimal", "youden",
            "sensitivity_constrained", "precision_constrained")


class Rng:
    """Deterministic resampling: one fixed bitstream for the whole analysis."""

    def __init__(self, seed: int = C.BOOTSTRAP_SEED):
        self._r = np.random.default_rng(seed)

    def resample(self, n: int) -> np.ndarray:
        return self._r.integers(0, n, size=n, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Point metric helpers (mirror src.metrics.binary_metrics)
# --------------------------------------------------------------------------- #
def _binary_metric(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                   name: str) -> float:
    if name == "auroc":
        return auroc(y_true, y_prob)
    if name == "auprc":
        return auprc(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(np.int64)
    m = binary_metrics(y_true, y_pred)
    val = m[name]
    if name == "precision" and np.isnan(val):  # bootstrap sample: no pos preds
        return 0.0
    return float(val)


# --------------------------------------------------------------------------- #
def classify_row(label: str, variant: str, policy: str, metric: str,
                 point: float, lo: float, hi: float) -> dict:
    return {
        "split": "test",
        "class": label,
        "calibration": variant,
        "threshold_policy": policy,
        "metric": metric,
        "point_estimate": round(point, 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
        "n_bootstraps": int(C.BOOTSTRAP_ITERATIONS),
        "confidence_level": C.BOOTSTRAP_CI_LEVEL,
        "method": "percentile bootstrap (resampling unit = test image)",
        "rng_seed": C.BOOTSTRAP_SEED,
    }


def percentile_ci(samples: np.ndarray, alpha: float) -> tuple:
    q = 100.0 * (1.0 - alpha)
    lo = float(np.percentile(samples, 100.0 - q))
    hi = float(np.percentile(samples, q))
    return lo, hi


# --------------------------------------------------------------------------- #
def load_test_predictions() -> pd.DataFrame:
    raw_dir = C.RAW_PREDICTIONS_DIR
    matches = sorted(Path(raw_dir).glob("test__*.csv"))
    if not matches:
        raise FileNotFoundError(f"no frozen test predictions under {raw_dir}")
    df = pd.read_csv(matches[0])
    assert (df.split == "test").all(), "expected test-only predictions"
    return df


def frozen_thresholds(variant: str) -> Dict[str, Dict[str, float]]:
    file = C.THRESHOLD_METRICS_DIR / {
        "raw": "thresholds_val.json",
        "calibrated": "thresholds_calibrated_val.json",
    }[variant]
    raw = json.loads(file.read_text())
    return {policy: {lbl: float(entry["threshold"])
                     for lbl, entry in per_lbl.items()}
            for policy, per_lbl in raw["policies"].items()}


def frozen_bin_edges(label: str, probs: np.ndarray) -> np.ndarray:
    """Recreate the artifact binning (n_bins, strategy from config) on test."""
    return _bin_edges(C.CALIBRATION_BINS, C.CALIBRATION_STRATEGY, probs)


def ece_from_edges(y_true: np.ndarray, probs: np.ndarray, edges: np.ndarray) -> float:
    idx = np.digitize(probs, edges[1:-1], right=False)
    counts = np.bincount(idx, minlength=C.CALIBRATION_BINS).astype(np.float64)
    acc = np.bincount(idx, weights=y_true, minlength=C.CALIBRATION_BINS)
    conf = np.bincount(idx, weights=probs, minlength=C.CALIBRATION_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        coeff = counts / counts.sum()
        gap = np.abs(acc / np.maximum(counts, 1) - conf / np.maximum(counts, 1))
    return float(np.sum(coeff * np.where(counts > 0, gap, 0.0)))


def calibrated_probs(logits: np.ndarray, label: str) -> np.ndarray:
    """Apply the frozen temperature scalers recorded for this label."""
    temp = json.loads(C.TEMPERATURE_FILE.read_text())["metadata"]["temperatures"][label]
    from .calibration import sigmoid

    return sigmoid(logits / float(temp))


# --------------------------------------------------------------------------- #
def run_boot() -> Path:
    df = load_test_predictions()
    rng = Rng()
    alpha = 1.0 - C.BOOTSTRAP_CI_LEVEL
    rows: List[dict] = []

    ranks = ("auroc", "auprc")
    binary_thr = ("accuracy", "precision", "recall", "specificity", "f1")
    n = len(df)
    n_boot = int(C.BOOTSTRAP_ITERATIONS)

    for label in C.TARGET_LABELS:
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        logits = df[f"logit_{label}"].to_numpy(dtype=np.float64)
        probs = {
            "raw": df[f"prob_{label}"].to_numpy(dtype=np.float64),
            "calibrated": calibrated_probs(logits, label),
        }
        edges = {
            "raw": frozen_bin_edges(label, probs["raw"]),
            "calibrated": frozen_bin_edges(label, probs["calibrated"]),
        }

        for metric in ranks:
            point = _binary_metric(y, probs["raw"], 0.0, metric)
            samples = np.empty(n_boot)
            for b in range(n_boot):
                ix = rng.resample(n)
                samples[b] = _binary_metric(y[ix], probs["raw"][ix], 0.0, metric)
            lo, hi = percentile_ci(samples, alpha)
            rows.append(classify_row(
                label, "both (temperature-invariant)", "n/a", metric,
                point, lo, hi))

        for variant in ("raw", "calibrated"):
            p = probs[variant]
            th = frozen_thresholds(variant)
            for metric, fn in (("brier_score",
                                lambda yy, pp: float(np.mean((pp - yy) ** 2))),
                               ("ece",
                                lambda yy, pp: ece_from_edges(yy, pp, edges[variant]))):
                point = fn(y, p)
                samples = np.array([fn(y[ix := rng.resample(n)], p[ix])
                                    for _ in range(n_boot)])
                lo, hi = percentile_ci(samples, alpha)
                rows.append(classify_row(label, variant, "n/a", metric,
                                         point, lo, hi))

            for policy in POLICIES:
                threshold = float(th[policy][label])
                for metric in binary_thr:
                    point = _binary_metric(y, p, threshold, metric)
                    samples = np.empty(n_boot)
                    for b in range(n_boot):
                        ix = rng.resample(n)
                        samples[b] = _binary_metric(y[ix], p[ix], threshold, metric)
                    lo, hi = percentile_ci(samples, alpha)
                    rows.append(classify_row(label, variant, policy, metric,
                                             point, lo, hi))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(C.CONFIDENCE_INTERVALS_CSV, index=False)
    logger.info("confidence_intervals.csv -> %d rows", len(df_out))

    meta = {
        "split": "test",
        "n_images": int(n),
        "n_bootstraps": n_boot,
        "confidence_level": C.BOOTSTRAP_CI_LEVEL,
        "rng_seed": C.BOOTSTRAP_SEED,
        "resampling_unit": "test image (row)",
        "method": "percentile bootstrap (percentiles 2.5 / 97.5)",
        "note": ("AUROC/AUPRC are temperature-invariant so a single pair of "
                 "rows is emitted per label; calibration metrics use frozen "
                 "binning; precision=0 for bootstrap samples with no "
                 "predicted positives."),
    }
    meta_path = C.FINAL_RESULTS_DIR / "confidence_intervals.meta.json"
    write_json(meta_path, meta)
    return C.CONFIDENCE_INTERVALS_CSV


if __name__ == "__main__":
    run_boot()