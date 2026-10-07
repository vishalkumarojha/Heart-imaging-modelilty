"""Paired patient-level permutation tests + Holm-Bonferroni adjustment.

Permutation (resampling unit = patient) for *operating-point* metrics only.
AUROC/AUPRC are excluded because they are invariant under every monotone
calibration map we use: ΔAUROC between arms is identically 0 and the test would
be meaningless (see `src/statistics/delong.py` for the documented degeneracy).

Mechanism
---------
For each patient and each of the two arms, the frozen validation threshold
produces a per-patient confusion vector (tp, fp, tn, fn).  Under the null
"the arm assignment does not matter", the (arm_a, arm_b) contribution of every
patient is exchangeable, so a permutation flips the assignment of each patient
with probability ½ and re-aggregates the two pseudo-arms.  The test statistic is
the arm difference Δ = f1_D(cohort) − f1_A(cohort) re-evaluated at the *frozen*
thresholds — thresholds are never refit inside the permutation, matching the
decision-policy interpretation ("given the fitted policy, does arm B change the
outcome metric beyond resampling noise?").

p-value: two-sided Monte-Carlo, (1 + #{|Δ_perm| ≥ |Δ_obs|}) / (1 + n_perm).

Primary hypotheses (pre-registered in the paper):
    per label: ΔF1(arm D = calibrated + F1-optimal vs arm A = raw + fixed 0.50)
Two labels -> Holm-Bonferroni with n=2 on the primary p-values.  All other
comparisons are reported as exploratory (raw p, no adjustment, labelled as such).

Artifact: outputs/metrics/patient_stats/paired_tests.json
Run:      python -m src.statistics.tests
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import binary_metrics
from ..reproducibility import write_json
from ..utils import setup_logging
from .data import (ARMS, PAIRED_COMPARISONS, load_test_predictions,
                   arm_thresholds, variant_probs)

logger = setup_logging()

PRIMARY = {("A", "D", "f1")}   # (arm_a, arm_b, metric) *per label*


def _from_counts(tp: float, fp: float, fn: float, tn: float, metric: str) -> float:
    if metric == "accuracy":
        return float((tp + tn) / max(tp + tn + fp + fn, 1.0))
    if metric == "precision":
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    if metric in ("recall", "sensitivity"):
        return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    if metric == "specificity":
        return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    if metric == "f1":
        den = 2 * tp + fp + fn
        return float(2 * tp / den) if den > 0 else float("nan")
    raise ValueError(f"unknown metric {metric!r}")


def per_patient_confusion(df: pd.DataFrame, label: str, variant: str, policy: str,
                          id_col: str = "Patient ID") -> np.ndarray:
    """(n_patients, 4) TP / FP / FN / TN rows, rows aligned to df's patient order."""
    p = variant_probs(df, label, variant)
    tau = arm_thresholds()[{   # arm->(variant, policy) is in ARMS; invert here
        (v, pol): arm for arm, (v, pol) in ARMS.items()}[(variant, policy)]][label]
    pred = (p >= tau).astype(np.int64)
    y = df[f"true_{label}"].to_numpy(dtype=np.int64)
    counts: List[np.ndarray] = []
    for _, sub in df.groupby(id_col, sort=False):
        yy = y[sub.index.to_numpy()]
        pp = pred[sub.index.to_numpy()]
        tp = int(((yy == 1) & (pp == 1)).sum())
        fp = int(((yy == 0) & (pp == 1)).sum())
        fn = int(((yy == 1) & (pp == 0)).sum())
        tn = int(((yy == 0) & (pp == 0)).sum())
        counts.append(np.array([tp, fp, fn, tn], dtype=np.float64))
    return np.vstack(counts) if counts else np.empty((0, 4))


def permutation_test_paired(
    contrib_a: np.ndarray,
    contrib_b: np.ndarray,
    metric: str,
    rng: np.random.Generator,
    n_perm: int,
) -> Dict[str, float]:
    """Two-sided paired permutation test on per-patient confusion contributions."""
    n_pat = contrib_a.shape[0]
    obs_a = _from_counts(*contrib_a.sum(axis=0), metric)
    obs_b = _from_counts(*contrib_b.sum(axis=0), metric)
    obs = obs_b - obs_a
    count = 0
    for _ in range(n_perm):
        mask = rng.random(n_pat) < 0.5
        a_mix = np.where(mask[:, None] == 1, contrib_b, contrib_a)
        b_mix = np.where(mask[:, None] == 1, contrib_a, contrib_b)
        ma = _from_counts(*a_mix.sum(axis=0), metric)
        mb = _from_counts(*b_mix.sum(axis=0), metric)
        if not (np.isnan(ma) or np.isnan(mb)):
            d = mb - ma
            if abs(d) >= abs(obs):
                count += 1
    return {
        "delta_observed": float(obs),
        "delta": float(obs),
        "p_value": (1.0 + count) / (1.0 + n_perm),
        "n_permutations": int(n_perm),
        "p_two_sided": (1.0 + count) / (1.0 + n_perm),
    }


def holm_bonferroni(pvalues: Sequence[float]) -> List[Tuple[int, float]]:
    """Holm-Bonferroni adjusted p-values, returns [(original_index, adjusted_p)]."""
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvalues[idx]))
        adjusted[idx] = running
    return list(zip(order, adjusted))


def run_paired_tests(n_perm: int = 5000, seed: int = 42) -> dict:
    df = load_test_predictions()
    rng = np.random.default_rng(seed)
    combos = (("A", "D"), ("A", "C"), ("B", "D"), ("C", "D"))
    metrics = ("f1", "precision", "recall", "specificity")

    results: List[dict] = []
    primary_p: List[Tuple[int, float]] = []   # (original index into results, p)
    for label in C.TARGET_LABELS:
        contribs: Dict[tuple, np.ndarray] = {}
        for arm, (variant, policy) in ARMS.items():
            if arm not in ({"A", "B", "C", "D"}) or (variant, policy) in contribs:
                continue
            contribs[(variant, policy)] = per_patient_confusion(df, label, variant, policy)
        for arm_a, arm_b in combos:
            va, pa = ARMS[arm_a]
            vb, pb = ARMS[arm_b]
            ca, cb = contribs[(va, pa)], contribs[(vb, pb)]
            for metric in metrics:
                res = permutation_test_paired(ca, cb, metric, rng, n_perm)
                res.update({
                    "label": label, "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
                    "variant_a": va, "variant_b": vb,
                    "policy_a": pa, "policy_b": pb,
                })
                if (arm_a, arm_b, metric) in PRIMARY:
                    primary_p.append((len(results), res["p_value"]))
                    res["primary_hypothesis"] = True
                results.append(res)

    adjusted_pairs = holm_bonferroni([p for _, p in primary_p])
    for (res_idx, _), (_, adj) in zip(primary_p, adjusted_pairs):
        results[res_idx]["holm_adjusted_p"] = round(adj, 6)

    payload = {
        "split": "test",
        "resampling_unit": "patient (paired permutation; thresholds frozen on val)",
        "n_permutations": int(n_perm),
        "rng_seed": seed,
        "primary_hypotheses": sorted(f"Δ{metric} {b} vs {a} ({label})" for a, b, metric in PRIMARY
                                    for label in C.TARGET_LABELS),
        "holm_note": ("Holm-Bonferroni across the 2 primary per-label ΔF1 (D vs A) "
                      "tests; all other p-values are exploratory (unadjusted)."),
        "threshold_policy_note": ("Permutation holds the fitted validation thresholds "
                                  "frozen; it tests the operating-point delta of the "
                                  "decision policy, not threshold refitting."),
        "per_label_results": results,
    }
    path = C.PATIENT_STATS_DIR / "paired_tests.json"
    write_json(path, payload)
    logger.info("paired_tests.json -> %d permutation results (%d primary)",
                len(results), len(primary_p))
    return payload


if __name__ == "__main__":
    run_paired_tests()