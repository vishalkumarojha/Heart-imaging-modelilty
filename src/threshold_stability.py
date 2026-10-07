"""Decision-threshold stability on validation patient-bootstraps.

"Will the fitted operating point change if the validation sample were slightly
different?"  The threshold for every policy is refit on each of 2000 *patient*
resamples of the validation split (patients, not rows, are resampled), in both
the raw and the calibrated probability spaces.  Calibrated-space refits apply the
FROZEN temperature / logistic parameters to the resampled logits BEFORE fitting,
so the stability analysis measures threshold variability given the fit
calibrator — not a full re-estimation of the calibration (that is out of scope).

Outputs
-------
outputs/metrics/threshold_stability/threshold_stability.json   summary + raw CIs
outputs/metrics/threshold_stability/threshold_stability.csv    long table
outputs/plots/threshold/threshold_stability.png                CI bands per policy

The module first verifies (equality test) that the vectorized fitter reproduces
the frozen validation thresholds exactly; a mismatch raises before any bootstrap
runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .reproducibility import write_json
from .utils import setup_logging
from .statistics.bootstrap import PatientResampler
from .statistics.data import (VARIANTS, frozen_threshold_values,
                              load_val_predictions, variant_probs)
from .statistics.vectorized_thresholds import assert_policy_equivalence, find_threshold_vectorized

logger = setup_logging()

EXCLUDED_STATICS = {"fixed"}   # fixed τ = 0.5 has zero variance by construction


def run_threshold_stability(
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    if iterations is None:
        iterations = int(C.THRESHOLD_STABILITY_ITERATIONS)
    seed = int(C.THRESHOLD_STABILITY_SEED) if seed is None else int(seed)

    df = load_val_predictions()
    for label in C.TARGET_LABELS:
        for variant in VARIANTS:
            frozen_threshold_values(variant)

    # ---- vectorized == scalar equality check on the FULL val split ---------
    labels = C.TARGET_LABELS
    prob_stack = {variant: np.column_stack([variant_probs(df, l, variant) for l in labels])
                  for variant in VARIANTS}
    y_val = df[[f"true_{l}" for l in labels]].to_numpy(dtype=np.int64)
    logger.info("verifying vectorized fitter reproduces frozen thresholds…")
    eq = assert_policy_equivalence(y_val, prob_stack, labels)
    logger.info("equivalence OK for %d variant/projects", len(eq))

    resampler = PatientResampler(df)
    rng = np.random.default_rng(seed)

    summaries: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    rows: List[dict] = []
    for label in labels:
        y_1d = df[f"true_{label}"].to_numpy(dtype=np.float64)
        for variant in VARIANTS:
            p_1d = variant_probs(df, label, variant)
            samples = {policy: np.zeros(iterations, dtype=np.float64) for policy in C.THRESHOLD_POLICIES}
            for b in range(iterations):
                rows_sel = resampler.sample_rows(rng)
                ry = y_1d[rows_sel]
                rp = p_1d[rows_sel]
                for policy in C.THRESHOLD_POLICIES:
                    if policy == "fixed":
                        samples[policy][b] = C.DECISION_THRESHOLD
                    else:
                        rec = find_threshold_vectorized(ry, rp, policy)
                        samples[policy][b] = float(rec["threshold"])
            for policy in C.THRESHOLD_POLICIES:
                first = float(frozen_threshold_values(variant)[policy][label])
                s = samples[policy]
                lo, hi = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
                band5 = _within_band(first, lo, hi, 0.05)
                band10 = _within_band(first, lo, hi, 0.10)
                summary = {
                    "first_fit": first,
                    "mean": float(np.mean(s)),
                    "sd": float(np.std(s)),
                    "median": float(np.median(s)),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci_width": float(hi - lo),
                    "min": float(np.min(s)),
                    "max": float(np.max(s)),
                    "within_pm05pct_of_first_fit": bool(band5),
                    "within_pm10pct_of_first_fit": bool(band10),
                    "n_bootstraps": int(iterations),
                    "policy_trivially_static": policy in EXCLUDED_STATICS,
                }
                summaries.setdefault(label, {}).setdefault(variant, {})[policy] = summary
                rows.append({"label": label, "variant": variant, "policy": policy, **summary})
                logger.info("%s / %-11s / %-22s τ=%.4f [%.4f, %.4f] %s",
                            label, variant, policy, first, lo, hi,
                            "±5% ok" if band5 else "±5% FAIL")

    out_csv = C.THRESHOLD_STABILITY_DIR / "threshold_stability.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    payload = {
        "split": "val",
        "resampling_unit": "patient (cluster bootstrap)",
        "n_bootstraps": int(iterations),
        "rng_seed": seed,
        "method": ("thresholds refit per patient-bootstrap; calibrated space uses "
                   "FROZEN temperature/logistic params applied to resampled logits "
                   "before fitting; percentile CI = 2.5/97.5"),
        "equivalence_check": {
            "passed": True,
            "detail": f"vectorized refit == frozen val fits for all variants/policies/labels "
                      f"({len(eq)} checks), atol=1e-9",
            "atol": 1e-9,
        },
        "per_label": summaries,
    }
    write_json(C.THRESHOLD_STABILITY_JSON, payload)
    logger.info("threshold_stability.json + .csv -> %s", out_csv)
    return payload


def _within_band(first: float, lo: float, hi: float, frac: float) -> bool:
    if not np.isfinite(first) or first == 0.0:
        return False
    return bool((lo >= first * (1.0 - frac)) and (hi <= first * (1.0 + frac)))


if __name__ == "__main__":
    run_threshold_stability()