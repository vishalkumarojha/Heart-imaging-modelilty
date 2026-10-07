"""Prevalence-shift sensitivity on frozen test predictions (SIMULATED).

The canonical test operating points (arm D: calibrated + F1-optimal thresholds)
are reported at the *observed* test prevalence (~2.6% Cardiomegaly / low
single-digit Effusion).  Clinical question: how do F1 / precision / recall /
AUPRC behave if the deployed population had a different prevalence?

This module answers it by deterministically SUBSAMPLING NEGATIVES of the frozen
test predictions to reach each target prevalence — positives are kept in full and
no patient/feature covariates change.  That is a *prevalence-shift simulation*,
NOT external validation: covariate shift beyond prevalence is explicitly not
modelled (stated in the artifact and in the paper limitations).

Rules
-----
* AUROC is invariant under negative subsampling (rank sequence of scores among
  retained rows is unchanged in distribution) — reported as a check, not a claim.
* Positives are kept in full and only the negative pool is subsampled, so the
  simulated prevalence can only be RAISED relative to the observed cohort
  (enriched / referred populations).  Targets above the observed prevalence are
  simulated; targets below it are unreachable in this design and are flagged
  `target_not_feasible` (dropping positives to reduce prevalence is out of the
  negative-subsampling scope and is stated in the limitations).
* Reproducible subsampling: rng seeded per (label, target) with PREVALENCE_SEED.

Artifacts
---------
outputs/metrics/prevalence/prevalence_shift.json
outputs/metrics/prevalence/prevalence_shift.csv
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .metrics import auprc, auroc, binary_metrics
from .reproducibility import write_json
from .utils import setup_logging
from .statistics.data import load_test_predictions, variant_probs

logger = setup_logging()


def _seed_for(seed: int, label: str, target_pct: float) -> int:
    return int(hashlib.sha256(f"{seed}|{label}|{target_pct:.4g}".encode()).hexdigest()[:12], 16)


def run_prevalence_shift() -> dict:
    df = load_test_predictions()
    grid = [float(p) for p in C.PREVALENCE_GRID]
    rows: List[dict] = []
    per_label: Dict[str, object] = {}

    for label in C.TARGET_LABELS:
        arm_d_tau = _arm_d_threshold(label)
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        p_cal = variant_probs(df, label, "calibrated")
        p_raw = variant_probs(df, label, "raw")
        P = int((y == 1).sum())
        N = int((y == 0).sum())
        natural = P / (P + N)

        cohort_rows: List[dict] = []
        for target_pct in grid:
            target_prev = target_pct / 100.0
            if target_prev < natural - 1e-12:
                cohort_rows.append({
                    "target_prevalence_pct": target_pct,
                    "target_prevalence": target_prev,
                    "observed_prevalence": round(natural, 6),
                    "target_not_feasible": True,
                    "note": "target below observed prevalence is unreachable: "
                            "positives are kept in full and only negatives are "
                            "subsampled, which can only raise prevalence — "
                            "cell skipped",
                })
                continue
            want_neg = int(np.floor(P * (1.0 - target_prev) / target_prev))
            if want_neg >= N:
                cohort_rows.append(dict(target_prevalence_pct=target_pct,
                                        target_prevalence=target_prev,
                                        observed_prevalence=round(natural, 6),
                                        target_not_feasible=True,
                                        note="need more negatives than available"))
                continue
            rng = np.random.default_rng(_seed_for(C.PREVALENCE_SEED, label, target_pct))
            neg_idx = np.flatnonzero(y == 0)
            keep = rng.choice(neg_idx, size=want_neg, replace=False)
            cohort = np.concatenate((np.flatnonzero(y == 1), keep))
            yy, pp_cal, pp_raw = y[cohort], p_cal[cohort], p_raw[cohort]
            m = binary_metrics(yy, pp_cal, arm_d_tau)
            m["auroc"] = auroc(yy, pp_raw)          # invariant — recorded as a check
            m["auprc"] = auprc(yy, pp_cal)
            m["simulated"] = True
            m["target_not_feasible"] = False
            m["target_prevalence_pct"] = target_pct
            m["observed_prevalence"] = round(float(yy.mean()), 6)
            m["n_positives_kept"] = int(P)
            m["n_negatives_kept"] = int(want_neg)
            m["class"] = label
            cohort_rows.append(m)
            logger.info("prevalence %s %5.2f%%: F1 %.4f / prec %.4f / AUPRC %.4f",
                        label, target_pct, m["f1"], m["precision"], m["auprc"])
        per_label[label] = {"observed_prevalence": round(natural, 6),
                            "n_positives": P, "n_negatives": N,
                            "threshold_calibrated_f1_optimal": arm_d_tau,
                            "cohorts": cohort_rows}
        rows.extend(cohort_rows)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(C.PREVALENCE_DIR / "prevalence_shift.csv", index=False)

    payload = {
        "split": "test",
        "method": ("deterministic negative subsampling of frozen test predictions — "
                   "prevalence-shift SIMULATION, not external validation; "
                   "covariate shift not modelled"),
        "seed": C.PREVALENCE_SEED,
        "grid_pct": grid,
        "operating_point": "arm D (calibrated + f1_optimal)", "per_label": per_label,
    }
    write_json(C.PREVALENCE_SHIFT_JSON, payload)
    logger.info("prevalence_shift.json/.csv -> %d cohort cells", len(rows))
    return payload


def _arm_d_threshold(label: str) -> float:
    from .statistics.data import frozen_threshold_values
    return float(frozen_threshold_values("calibrated")["f1_optimal"][label])


if __name__ == "__main__":
    run_prevalence_shift()