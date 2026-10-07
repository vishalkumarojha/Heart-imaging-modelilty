"""Extension ablation arms E/F + decision-policy matrix.

The published ablation (arms A–D) is frozen and untouched.  This module adds the
two EXTENSION arms that exist because the IEEE upgrade introduces the second
calibration family (Platt-style logistic, `src.alternative_calibration`):

    A  raw probabilities        + fixed 0.50
    B  temperature-calibrated   + fixed 0.50
    C  raw probabilities        + F1-optimal (val)
    D  temperature-calibrated   + F1-optimal (val)          ← published primary
    E  logistic-calibrated      + fixed 0.50                (extension)
    F  logistic-calibrated      + F1-optimal (val)          (extension)

E/F are written as `outputs/metrics/experiments/ext_extension_arms.json` +
`outputs/final_results/extension_arms.csv`, and are ALWAYS labelled extension —
they never enter master_results.csv (A–D stay the frozen published comparison).

The second artifact, `outputs/final_results/decision_policy_analysis.csv`, is the
full operating-point picture: every variant × policy × label evaluated on the
test split at frozen thresholds, with the *constraint achievements* (validation
vs test) carried along, so a reader can see how well a val-fitted screening or
referral policy performs out-of-sample.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .calibration import brier_score, expected_calibration_error, negative_log_likelihood
from .metrics import auroc, auprc, binary_metrics
from .reproducibility import write_json
from .utils import setup_logging
from .statistics.data import (POLICIES, VARIANTS, frozen_threshold_records,
                              frozen_threshold_values, load_test_predictions,
                              variant_probs)

logger = setup_logging()

EXTENSION_ARMS = {"E": ("logistic", "fixed"), "F": ("logistic", "f1_optimal")}
ARM_LETTERS = ("A", "B", "C", "D", "E", "F")


def run_extension_arms() -> dict:
    df = load_test_predictions()
    threshold_map: Dict[str, Dict[str, Dict[str, float]]] = {
        variant: frozen_threshold_values(variant) for variant in VARIANTS}
    per_arm: Dict[str, Dict[str, object]] = {}

    for label in C.TARGET_LABELS:
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        for variant in VARIANTS:
            p = variant_probs(df, label, variant)
            for policy in POLICIES:
                tau = threshold_map[variant][policy][label]
                m = binary_metrics(y, p, tau)
                m["auroc"] = auroc(y, p)
                m["auprc"] = auprc(y, p)
                m["brier"] = brier_score(y, p)
                m["nll"] = negative_log_likelihood(y, p)
                m["ece"] = expected_calibration_error(y, p, C.CALIBRATION_BINS, C.CALIBRATION_STRATEGY)
                m["variant"] = variant
                m["policy"] = policy
                m["class"] = label
                key = (variant, policy)
                per_arm.setdefault(key, {"label": {}, "threshold": {}})
                per_arm[key]["label"][label] = m
                per_arm[key]["threshold"][label] = tau

    # build the long comparison CSV (all arms, all metrics)
    rows: List[dict] = []
    for label in C.TARGET_LABELS:
        for arm in ARM_LETTERS:
            variant, policy = _arm_to_variant_policy(arm)
            m = per_arm[(variant, policy)]["label"][label]
            rows.append({"arm": arm, "variant": variant, "policy": policy,
                         "class": label, "threshold": m["threshold"],
                         "f1": m["f1"], "precision": m["precision"],
                         "recall": m["recall"], "specificity": m["specificity"],
                         "accuracy": m["accuracy"], "auroc": m["auroc"],
                         "auprc": m["auprc"], "ece": m["ece"], "brier": m["brier"],
                         "nll": m["nll"], "extension": arm in EXTENSION_ARMS})
    df_ext = pd.DataFrame(rows)
    df_ext.to_csv(C.FINAL_RESULTS_DIR / "extension_arms.csv", index=False)

    # deltas vs the two published comparisons (E/F vs D, E/F vs C) — exploratory
    deltas: List[dict] = []
    for label in C.TARGET_LABELS:
        for ref_arm in ("C", "D"):
            for ext_arm in ("E", "F"):
                for metric in ("f1", "precision", "recall", "specificity"):
                    rv, rp = _arm_to_variant_policy(ref_arm)
                    ev, ep = _arm_to_variant_policy(ext_arm)
                    left = per_arm[(ev, ep)]["label"][label][metric]
                    right = per_arm[(rv, rp)]["label"][label][metric]
                    deltas.append({
                        "class": label, "arm_a": ref_arm, "arm_b": ext_arm,
                        "metric": metric, "delta": round(float(left - right), 6),
                        "pairs_identical_thresholds": None,
                    })
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(C.FINAL_RESULTS_DIR / "extension_deltas.csv", index=False)

    payload = {
        "note": ("extension arms E/F (logistic calibration) — never part of the "
                 "published A–D ablation; kept in master_results.csv untouched"),
        "arms": {k: {"variant": v, "policy": p} for k, (v, p) in _all_arms().items()},
        "per_label": {},
    }
    for label in C.TARGET_LABELS:
        payload["per_label"][label] = {
            arm: {**{m: per_arm[(v, p)]["label"][label][m] for m in
                     ("threshold", "accuracy", "precision", "recall", "specificity",
                      "f1", "auroc", "auprc", "ece", "brier", "nll")},
                  "extension": arm in EXTENSION_ARMS}
            for arm, (v, p) in _all_arms().items()
        }
    write_json(C.EXT_EXTENSION_ARMS_JSON, payload)
    logger.info("ext_extension_arms.json + extension_arms.csv + extension_deltas.csv written")
    return payload


def run_decision_policy_analysis() -> None:
    df = load_test_predictions()
    rows: List[dict] = []
    for label in C.TARGET_LABELS:
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        for variant in VARIANTS:
            p = variant_probs(df, label, variant)
            records = frozen_threshold_records(variant)
            for policy in POLICIES:
                rec = records[policy][label]
                tau = float(rec["threshold"])
                m = binary_metrics(y, p, tau)
                rows.append({
                    "class": label,
                    "variant": variant,
                    "policy": policy,
                    "threshold": round(tau, 6),
                    "constraint_met_on_val": rec.get("constraint_met"),
                    "target": rec.get("target"),
                    "val_achieved": rec.get("achieved_sensitivity", rec.get("achieved_precision")),
                    "val_achieved_metric": ("sensitivity" if "achieved_sensitivity" in rec
                                            else "precision" if "achieved_precision" in rec else None),
                    "test_accuracy": m["accuracy"],
                    "test_precision": m["precision"],
                    "test_recall": m["recall"],
                    "test_specificity": m["specificity"],
                    "test_f1": m["f1"],
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(C.DECISION_POLICY_ANALYSIS_CSV, index=False)
    logger.info("decision_policy_analysis.csv -> %d rows", len(df_out))


def _all_arms() -> Dict[str, tuple]:
    from .statistics.data import ARMS
    out = dict(ARMS)
    out.update(EXTENSION_ARMS)
    return out


def _arm_to_variant_policy(arm: str) -> tuple:
    return _all_arms()[arm]


if __name__ == "__main__":
    run_extension_arms()
    run_decision_policy_analysis()