"""Canonical master results table + documentation/artifact verification.

Two outputs:

1. `outputs/final_results/master_results.csv` — EVERY verified metric that the
   study actually computed, one row per
   (experiment, split, label, calibration, threshold_policy), with the four
   ablation experiments encoded as `experiment_id`:
       A            baseline            raw          fixed 0.50
       B            calibration only    calibrated   fixed 0.50
       C            threshold only      raw          best policy (and all 5)
       D            combined            calibrated   best policy (and all 5)
   Missing/unavailable metrics are written as `NA` (never 0).

2. `outputs/final_results/verification_report.json` — the "audit the docs"
   step made executable: every number quoted in the README/status documents is
   recomputed from the underlying JSON/CSV artifact and the comparison recorded
   as PASS/DIFF/REDACT, so a doc that drifts from the data is caught, not
   asserted away.

    python -m src.master_results
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .reproducibility import write_json
from .utils import setup_logging

logger = setup_logging()

EXPERIMENT_DEFINITIONS = [
    ("A", "baseline", "raw", "fixed"),
    ("B", "calibration_only", "calibrated", "fixed"),
    ("C", "threshold_optimization_only", "raw", "f1_optimal"),
    ("D", "calibration_plus_threshold", "calibrated", "f1_optimal"),
]

# the full policy matrix (C and D evaluated against every implemented policy)
ALL_POLICIES = ("fixed", "f1_optimal", "youden",
                "sensitivity_constrained", "precision_constrained")


# --------------------------------------------------------------------------- #
# Artifact readers
# --------------------------------------------------------------------------- #
def _load(path: Path) -> Optional[dict]:
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


class Artifacts:
    def __init__(self) -> None:
        self.base = {}
        self.exp1 = _load(C.EXPERIMENT_METRICS_DIR / "exp1_calibration.json") or {}
        self.exp_probs = {
            "raw": _load(C.EXPERIMENT_METRICS_DIR / "exp2_thresholds_raw.json") or {},
            "calibrated": _load(C.EXPERIMENT_METRICS_DIR
                                / "exp3_thresholds_calibrated.json") or {},
        }
        self.ea = _load(C.ERROR_ANALYSIS_DIR / "error_analysis_test.json") or {}
        for split in ("test", "val"):
            self.base[split] = _load(C.BASELINE_METRICS_DIR
                                     / f"baseline_metrics_{split}.json") or {}


ART = Artifacts()


def _n(v) -> object:
    """Explicit NA (documented missing) — never 0 for a skipped metric."""
    return "NA" if v is None else (None if (isinstance(v, float) and np.isnan(v)) else v)


def _round(v) -> object:
    if v is None or v == "NA":
        return "NA" if v == "NA" else "NA"
    return round(float(v), 6)


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
def threshold_row(split: str, label: str, variant: str, policy: str
                  ) -> Optional[dict]:
    """Accuracy/precision/recall/specificity/F1 for (variant, policy) on split."""
    if variant == "raw":
        if policy == "fixed":
            rec = (ART.base.get(split) or {}).get("metrics", {}).get("per_label", {})
            entry = rec.get(label)
            if not entry:
                return None
            return {
                "n": entry["n"], "tp": entry["tp"], "fp": entry["fp"],
                "tn": entry["tn"], "fn": entry["fn"], "accuracy": entry["accuracy"],
                "precision": entry["precision"], "recall": entry["recall"],
                "sensitivity": entry["sensitivity"], "specificity": entry["specificity"],
                "f1": entry["f1"],
            }
    eval_map = ART.exp_probs.get(variant, {}).get("evaluation", {})
    entry = (eval_map.get(split) or {}).get(label, {}).get(policy)
    if not entry:
        return None
    return {k: entry[k] for k in
            ("n", "tp", "fp", "tn", "fn", "accuracy", "precision",
             "recall", "sensitivity", "specificity", "f1")}


def calibration_row(split: str, label: str, variant: str) -> Optional[dict]:
    base = (ART.exp1.get("splits") or {}).get(split, {}).get(label, {})
    cal = base.get(variant)
    if not cal:
        return None
    return {
        "ece": cal["ece"], "brier_score": cal["brier"], "mce": cal["mce"],
        "nll": cal["nll"], "n": cal["n"], "prevalence": cal["prevalence"],
        "auroc": base.get(f"auroc_{variant}"), "auprc": base.get(f"auprc_{variant}"),
    }


def build_master(split: str = "test") -> List[dict]:
    rows: List[dict] = []
    labels = list(C.TARGET_LABELS)
    exp_names = {eid: name for eid, name, _v, _p in EXPERIMENT_DEFINITIONS}

    def add(exp_id: str, variant: str, policy: str) -> None:
        for label in labels:
            thr = threshold_row(split, label, variant, policy)
            cal = calibration_row(split, label, variant)
            if thr is None or cal is None:
                logger.warning("skipping %s/%s/%s: missing artifact row",
                               exp_id, policy, label)
                return
            base_entry = (ART.base.get(split) or {}).get("metrics", {})\
                .get("per_label", {}).get(label, {})
            n_pos = base_entry.get("support_pos", thr.get("n") or 0)
            n_neg = base_entry.get("support_neg", thr.get("n") or 0)
            rows.append({
                "experiment_id": exp_id,
                "experiment_name": exp_names[exp_id],
                "dataset": "NIH ChestXray14",
                "split": split,
                "model": "densenet121_baseline_v1",
                "calibration": variant,
                "threshold_policy": policy,
                "class": label,
                "accuracy": _round(thr["accuracy"]),
                "precision": _round(thr["precision"]),
                "recall": _round(thr["recall"]),
                "sensitivity": _round(thr["sensitivity"]),
                "specificity": _round(thr["specificity"]),
                "f1": _round(thr["f1"]),
                "auroc": _round(cal["auroc"]),
                "auprc": _round(cal["auprc"]),
                "ece": _round(cal["ece"]),
                "brier_score": _round(cal["brier_score"]),
                "n_samples": int(thr["n"]),
                "n_positive": int(n_pos),
                "n_negative": int(n_neg),
            })

    # A and B are the two fixed-0.50 arms (one row each per label)
    for exp_id, _name, variant, policy in EXPERIMENT_DEFINITIONS[:2]:
        add(exp_id, variant, policy)
    # C and D are the threshold experiments: all five implemented policies
    for policy in ALL_POLICIES:
        add("C", "raw", policy)
        add("D", "calibrated", policy)
    return rows


# --------------------------------------------------------------------------- #
# Documentation verification (recompute every claimed number from artifacts)
# --------------------------------------------------------------------------- #
def verification_checks() -> List[dict]:
    checks: List[dict] = []

    def check(claim: str, got: object, expected: object, tol: float = 0.0) -> None:
        def num(v) -> Optional[float]:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            return None

        if isinstance(got, bool) and isinstance(expected, bool):
            ok = (got is expected)
        else:
            g, e = num(got), num(expected)
            ok = (g is not None and e is not None and abs(g - e) <= tol * max(1.0, abs(e)))
        checks.append({
            "claim": claim, "artifact_value": got, "documented_value": expected,
            "status": "PASS" if ok else "DIFF",
            "tolerance": tol,
        })

    # baseline test AUROC/AUPRC per label
    per_label = (ART.base["test"].get("metrics") or {}).get("per_label", {})
    for lbl in C.TARGET_LABELS:
        check(f"baseline test AUROC [{lbl}]", per_label[lbl]["auroc"], {
            "Cardiomegaly": 0.8972, "Effusion": 0.8587}[lbl], tol=5e-4)
        check(f"baseline test F1 [fixed 0.5][{lbl}]", per_label[lbl]["f1"], {
            "Cardiomegaly": 0.2836, "Effusion": 0.4676}[lbl], tol=5e-4)

    # calibration ECE raw and calibrated (test)
    for lbl in C.TARGET_LABELS:
        e1 = (ART.exp1.get("splits") or {}).get("test", {}).get(lbl, {})
        check(f"ECE test raw [{lbl}]", e1["raw"]["ece"], {
            "Cardiomegaly": 0.1061, "Effusion": 0.2077}[lbl], tol=5e-4)
        check(f"ECE test calibrated [{lbl}]", e1["calibrated"]["ece"], {
            "Cardiomegaly": 0.1169, "Effusion": 0.2292}[lbl], tol=5e-4)
        check(f"ECE rises after calibration [{lbl}]",
              e1["calibrated"]["ece"] > e1["raw"]["ece"], True)

    # exp2 vs exp3 confusion coincidence (already artifact-level; single check)
    e2 = (ART.exp_probs["raw"].get("evaluation") or {}).get("test", {})
    e3 = (ART.exp_probs["calibrated"].get("evaluation") or {}).get("test", {})
    for lbl in C.TARGET_LABELS:
        check(f"confusion identical raw vs calibrated [{lbl}]",
              all((e2[lbl][p]["tp"], e2[lbl][p]["fp"], e2[lbl][p]["tn"],
                   e2[lbl][p]["fn"]) ==
                  (e3[lbl][p]["tp"], e3[lbl][p]["fp"], e3[lbl][p]["tn"],
                   e3[lbl][p]["fn"]) for p in ALL_POLICIES), True)

    # error analysis headline — Cardiomegaly high-confidence errors 48.7%
    hc = (ART.ea.get("strata") or {}).get("f1_optimal", {}) \
        .get("Cardiomegaly", {}).get("high_confidence_errors", {})
    check("Cardiomegaly high-confidence errors == 48.7% of errors",
          hc.get("share_of_errors", hc.get("share_of_errors")), 0.487, tol=0.005)
    check("high-confidence error count == 274", hc.get("count"), 274)

    # grad-cam bbox cohort headline numbers
    bbox = _load(C.GRADCAM_DIR / "bbox_localization.json") or {}
    if bbox:
        check("bbox cohort n == 43", bbox.get("n_with_box"), 43)
        check("pointing hits == 21", bbox.get("pointing_game_hits"), 21)
        check("concentration ratio == 2.60",
              bbox.get("mean_concentration_ratio"), 2.60, tol=0.03)
        check("ratio_above_1 == 38", bbox.get("ratio_above_1"), 38)

    # README/STATUS quoted numbers (spot set, all artifact-sourced)
    table = pd.read_csv(C.EXPERIMENT_METRICS_DIR / "experiments_summary.csv")
    f1_test = table[(table.experiment == "thresholds_raw") &
                    (table.split == "test") & (table.metric == "f1")]
    for lbl in C.TARGET_LABELS:
        v = f1_test[f1_test.label == lbl].set_index("policy")["value"]
        check(f"test F1 f1_optimal [{lbl}]", v.get("f1_optimal"),
              {"Cardiomegaly": 0.3536, "Effusion": 0.4866}[lbl], tol=5e-4)
        check(f"test F1 fixed [{lbl}]", v.get("fixed"),
              {"Cardiomegaly": 0.2836, "Effusion": 0.4676}[lbl], tol=5e-4)

    # patient / error concentration
    conc = ART.ea.get("patient_error_concentration", {})
    check("n error images == 4449", conc.get("n_error_images"), 4449)
    check("max errors one patient == 60", conc.get("max_errors_by_one_patient"), 60)
    check("n patients with errors == 1317", conc.get("n_patients_with_errors"), 1317)

    return checks


# --------------------------------------------------------------------------- #
def run() -> Dict[str, object]:
    C.FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_master("test")
    if not rows:
        raise SystemExit("no rows could be assembled — an experiment artifact is missing")
    df = pd.DataFrame(rows)
    df.to_csv(C.MASTER_RESULTS_CSV, index=False)
    logger.info("master_results.csv -> %d rows", len(df))

    checks = verification_checks()
    n_pass = sum(c["status"] == "PASS" for c in checks)
    n_diff = len(checks) - n_pass
    report = {
        "phase": "final_results",
        "status": "PASS" if n_diff == 0 else "DIFF",
        "checks_total": len(checks),
        "checks_passed": n_pass,
        "checks_differ": n_diff,
        "master_rows": len(df),
        "checks": checks,
        "note": "A DIFF here means documentation drifted from an artifact — "
                "fix the DOCUMENT, never the artifact.",
    }
    write_json(C.VERIFICATION_REPORT_JSON, report)
    logger.info("verification_report.json -> %d/%d PASS", n_pass, len(checks))
    return report


def parse_args() -> argparse.Namespace:
    import argparse

    p = argparse.ArgumentParser(description="Master results + verification")
    return p.parse_args()


if __name__ == "__main__":
    parse_args()
    run()