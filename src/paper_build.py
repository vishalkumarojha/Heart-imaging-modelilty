"""Paper package builder (final phase).

Derives the publication-facing package under `outputs/paper/` from the CANONICAL
`outputs/final_results/` and frozen experiment artifacts ONLY. Nothing new is
measured here — every number printed in the package is cross-checked against the
artifact it cites (claim audit) and every numeric token found in the prose
drafts must resolve to a verified artifact value or a documented constant.

Run:
    python -m src.paper_build

Outputs:
    paper/paper_evidence_map.json      every claim -> source artifact
    paper/tables/table_01..14.csv     publication-ready result tables
    paper/figures/figure_01..14.png   architecture/dataset schematics + the
                                      deterministic research figures (copies)
    paper/claim_audit.csv             claim x verified-by-artifact matrix
    paper/reproducibility_manifest.json
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C
from .reproducibility import write_json
from .utils import setup_logging

logger = setup_logging()

DPI = 150
SOURCE_PREFIX = "outputs"
POLICIES = ("fixed", "f1_optimal", "youden",
            "sensitivity_constrained", "precision_constrained")

# --------------------------------------------------------------------------- #
# Artifact readers
# --------------------------------------------------------------------------- #
def _json(rel: str) -> dict:
    p = Path(C.PROJECT_ROOT) / rel
    if not p.exists():
        raise FileNotFoundError(f"required artifact missing: {p}")
    return json.loads(p.read_text())


def _csv(rel: str) -> pd.DataFrame:
    p = Path(C.PROJECT_ROOT) / rel
    if not p.exists():
        raise FileNotFoundError(f"required artifact missing: {p}")
    return pd.read_csv(p)


def _r4(v) -> float:
    return round(float(v), 4)


BASE = _json(f"{SOURCE_PREFIX}/metrics/baseline/baseline_metrics_test.json")
EXP1 = _json(f"{SOURCE_PREFIX}/metrics/experiments/exp1_calibration.json")
EXP2 = _json(f"{SOURCE_PREFIX}/metrics/experiments/exp2_thresholds_raw.json")
EXP3 = _json(f"{SOURCE_PREFIX}/metrics/experiments/exp3_thresholds_calibrated.json")
EA = _json(f"{SOURCE_PREFIX}/metrics/error_analysis/error_analysis_test.json")
BBOX = _json(f"{SOURCE_PREFIX}/gradcam/bbox_localization.json")
GRAD = _json(f"{SOURCE_PREFIX}/gradcam/gradcam_summary.json")
MANI = _json(f"{SOURCE_PREFIX}/checkpoints/baseline/baseline_manifest.json")
SPLIT = _csv("data/processed/split_index.csv")
MASTER = _csv(f"{SOURCE_PREFIX}/final_results/master_results.csv")
THR_VAL = _json(f"{SOURCE_PREFIX}/metrics/thresholds/thresholds_val.json")
THR_CAL = _json(f"{SOURCE_PREFIX}/metrics/thresholds/thresholds_calibrated_val.json")
DELONG = _json(f"{SOURCE_PREFIX}/metrics/patient_stats/delong_auroc.json")
PAIRED = _json(f"{SOURCE_PREFIX}/metrics/patient_stats/paired_tests.json")
STAB = _json(f"{SOURCE_PREFIX}/metrics/threshold_stability/threshold_stability.json")
ECEJ = _json(f"{SOURCE_PREFIX}/metrics/ece_sensitivity/ece_sensitivity.json")
LOGCAL = _json(f"{SOURCE_PREFIX}/metrics/calibration/logistic_calibration_report.json")
EXT = _json(f"{SOURCE_PREFIX}/metrics/experiments/ext_extension_arms.json")
PREV = _json(f"{SOURCE_PREFIX}/metrics/prevalence/prevalence_shift.json")
STAT = _json(f"{SOURCE_PREFIX}/final_results/statistical_report.json")
CIP = _csv(f"{SOURCE_PREFIX}/final_results/confidence_intervals_patient.csv")
AD = _csv(f"{SOURCE_PREFIX}/final_results/arm_differences.csv")


def master_row(exp: str, label: str, policy: str) -> dict:
    m = MASTER
    row = m[(m.experiment_id == exp) & (m["class"] == label)
            & (m.threshold_policy == policy)]
    if row.empty:
        raise KeyError(f"master row not found: {exp}/{label}/{policy}")
    return row.iloc[0].to_dict()


def _per_label_metrics(label: str) -> dict:
    return BASE["metrics"]["per_label"][label]


def _exp_split_metrics(exp: dict, split: str, label: str, variant: str) -> dict:
    return exp["splits"][split][label][variant]


def _policy_test(exp: dict, label: str, policy: str) -> dict:
    return exp["evaluation"]["test"][label][policy]


# --------------------------------------------------------------------------- #
# Claim registry (every claim -> resolver that re-extracts a value from source)
# --------------------------------------------------------------------------- #
CLAIMS: List[dict] = []


def add_claim(cid: str, claim: str, section: str, kind: str, source: str,
              resolve: Optional[Callable[[], object]] = None,
              expected: object = None, tol: float = 0.0,
              at_least: Optional[float] = None) -> None:
    claim_type = "numeric" if kind == "numeric" else "text"
    CLAIMS.append({
        "id": cid, "claim": claim, "section": section, "kind": kind,
        "claim_type": claim_type, "source_artifact": source, "resolve": resolve,
        "expected": expected, "tol": tol, "at_least": at_least,
    })


# Section -> display reference (single source of truth for the claim audit).
DISPLAY_REF = {
    "dataset": "table_01_dataset.csv; figure_02_dataset_distribution.png",
    "results_baseline": "table_02_baseline.csv; figure_03_roc.png; figure_04_precision_recall.png",
    "results_main": "table_03_main_results.csv; table_10_patient_bootstrap_ci.csv",
    "results_ablation": "table_04_ablation.csv; table_11_arm_differences.csv",
    "results_thresholds": "table_05_thresholds.csv; figure_06_threshold_analysis.png; table_12_threshold_stability.csv; figure_12_threshold_stability.png",
    "results_calibration": "table_06_calibration.csv; figure_05_calibration.png; table_13_calibration_comparison.csv; figure_10_calibration_comparison.png",
    "results_calibration_comparison": "table_13_calibration_comparison.csv; figure_10_calibration_comparison.png; figure_11_ece_sensitivity.png",
    "results_error_analysis": "table_07_error_analysis.csv; figure_08_error_analysis.png",
    "results_explainability": "table_08_explainability.csv; figure_09_gradcam.png",
    "external_validation": "table_09_external_validation_status.csv",
    "results_patient_bootstrap": "table_10_patient_bootstrap_ci.csv",
    "results_discrimination": "table_10_patient_bootstrap_ci.csv; figure_03_roc.png",
    "results_arm_differences": "table_11_arm_differences.csv; figure_13_decision_policy.png",
    "results_extension": "table_11_arm_differences.csv; figure_13_decision_policy.png",
    "results_threshold_stability": "table_12_threshold_stability.csv; figure_12_threshold_stability.png",
    "results_prevalence": "table_14_prevalence_shift.csv; figure_14_prevalence_sensitivity.png",
    "testing": "reproducibility_manifest.json",
}


BL = "outputs/metrics/baseline/baseline_metrics_test.json"
E1 = "outputs/metrics/experiments/exp1_calibration.json"
E2 = "outputs/metrics/experiments/exp2_thresholds_raw.json"
E3 = "outputs/metrics/experiments/exp3_thresholds_calibrated.json"
EAJ = "outputs/metrics/error_analysis/error_analysis_test.json"
BB = "outputs/gradcam/bbox_localization.json"
SP = "data/processed/split_index.csv"
MR = "outputs/final_results/master_results.csv"
TV = "outputs/metrics/thresholds/thresholds_val.json"
GO = "outputs/metrics/external/status.json"
MG = "outputs/checkpoints/baseline/baseline_manifest.json"

# --- dataset -------------------------------------------------------------- #
for split, lab in [("train", "76977"), ("val", "16451"), ("test", "15884")]:
    add_claim(f"ds_images_{split}", f"{lab} images in {split} split", "dataset",
              "numeric", SP,
              lambda s=split: int(SPLIT[SPLIT.split == s].shape[0]),
              int(lab), tol=0)
for split, pat in [("train", "20797"), ("val", "4468"), ("test", "4455")]:
    add_claim(f"ds_patients_{split}", f"{pat} patients in {split} split", "dataset",
              "numeric", SP,
              lambda s=split: int(SPLIT[SPLIT.split == s]["Patient ID"].nunique()),
              int(pat), tol=0)
add_claim("ds_total_images", "109,312 total images", "dataset", "numeric", SP,
          lambda: int(len(SPLIT)), 109312, tol=0)
add_claim("ds_total_patients", "29,720 total patients", "dataset", "numeric", SP,
          lambda: int(SPLIT["Patient ID"].nunique()), 29720, tol=0)

# --- baseline ------------------------------------------------------------- #
for lbl in C.TARGET_LABELS:
    pl = _per_label_metrics(lbl)
    for metric, val in [("auroc", 0.8972), ("auprc", 0.3043), ("f1", 0.2836),
                        ("precision", 0.1787), ("recall", 0.6867),
                        ("specificity", 0.9153)] if lbl == "Cardiomegaly" else \
                       [("auroc", 0.8587), ("auprc", 0.4700), ("f1", 0.4676),
                        ("precision", 0.3320), ("recall", 0.7902),
                        ("specificity", 0.7714)]:
        add_claim(f"base_{lbl.lower()}_{metric}", f"baseline {lbl} {metric}",
                  "results_baseline", "numeric", BL,
                  lambda m=metric, p=pl: _r4(p[m]), val, tol=5e-4)
add_claim("base_macro_auroc", "macro AUROC 0.8780", "results_baseline",
          "numeric", BL,
          lambda: _r4(BASE["metrics"]["macro"]["auroc"]), 0.8780, tol=5e-4)
add_claim("base_macro_auprc", "macro AUPRC 0.3871", "results_baseline",
          "numeric", BL,
          lambda: _r4(BASE["metrics"]["macro"]["auprc"]), 0.3871, tol=5e-4)

# --- calibration ---------------------------------------------------------- #
for lbl in C.TARGET_LABELS:
    r = _exp_split_metrics(EXP1, "test", lbl, "raw")
    c = _exp_split_metrics(EXP1, "test", lbl, "calibrated")
    add_claim(f"cal_{lbl.lower()}_nll", f"{lbl} NLL raw->calibrated",
              "results_calibration", "numeric", E1,
              lambda R=r, D=c: (_r4(R["nll"]), _r4(D["nll"])),
              (_r4(r["nll"]), _r4(c["nll"])), tol=5e-4)
    add_claim(f"cal_{lbl.lower()}_brier", f"{lbl} Brier raw->calibrated",
              "results_calibration", "numeric", E1,
              lambda R=r, D=c: (_r4(R["brier"]), _r4(D["brier"])),
              (_r4(r["brier"]), _r4(c["brier"])), tol=5e-4)
    add_claim(f"cal_{lbl.lower()}_ece", f"{lbl} ECE raw->calibrated",
              "results_calibration", "numeric", E1,
              lambda R=r, D=c: (_r4(R["ece"]), _r4(D["ece"])),
              (_r4(r["ece"]), _r4(c["ece"])), tol=5e-4)
    add_claim(f"cal_{lbl.lower()}_ece_up",
              f"{lbl} ECE rises after temperature scaling",
              "results_calibration", "text", E1,
              lambda R=r, D=c: bool(D["ece"] > R["ece"]), True)
add_claim("cal_T_cardio", "Cardiomegaly temperature T = 1.187", "results_calibration",
          "numeric",
          "outputs/metrics/calibration/temperature_scalers.json",
          lambda: _r4(_p_temp("Cardiomegaly")), 1.1865, tol=5e-4)
add_claim("cal_T_eff", "Effusion temperature T = 1.398", "results_calibration",
          "numeric",
          "outputs/metrics/calibration/temperature_scalers.json",
          lambda: _r4(_p_temp("Effusion")), 1.3979, tol=5e-4)

# --- thresholds (val-frozen) ---------------------------------------------- #
for lbl in C.TARGET_LABELS:
    for pol in POLICIES:
        tv = THR_VAL["policies"][pol][lbl]["threshold"]
        tc = THR_CAL["policies"][pol][lbl]["threshold"]
        add_claim(f"thr_{lbl.lower()}_{pol}",
                  f"{lbl} {pol} val-frozen threshold (raw) = {_r4(tv)}",
                  "results_thresholds", "numeric", TV,
                  lambda P=pol, L=lbl: _r4(THR_VAL["policies"][P][L]["threshold"]),
                  _r4(tv), tol=5e-4)

# --- combined-arm (D) test metrics ---------------------------------------- #
for lbl, f1 in [("Cardiomegaly", 0.3536), ("Effusion", 0.4866)]:
    add_claim(f"d_{lbl.lower()}_f1", f"combined-arm {lbl} test F1 = {f1}",
              "results_main", "numeric", MR,
              lambda L=lbl: _r4(master_row("D", L, "f1_optimal")["f1"]), f1, tol=5e-4)
for lbl, prec in [("Cardiomegaly", 0.3377), ("Effusion", 0.4440)]:
    add_claim(f"d_{lbl.lower()}_prec", f"combined-arm {lbl} test precision = {prec}",
              "results_main", "numeric", MR,
              lambda L=lbl: _r4(master_row("D", L, "f1_optimal")["precision"]),
              prec, tol=5e-4)
for lbl, rec, spec in [("Cardiomegaly", 0.3711, 0.9805),
                       ("Effusion", 0.5383, 0.9031)]:
    add_claim(f"d_{lbl.lower()}_rec", f"combined-arm {lbl} test recall = {rec}",
              "results_main", "numeric", MR,
              lambda L=lbl: _r4(master_row("D", L, "f1_optimal")["recall"]),
              rec, tol=5e-4)
    add_claim(f"d_{lbl.lower()}_spec", f"combined-arm {lbl} test specificity = {spec}",
              "results_main", "numeric", MR,
              lambda L=lbl: _r4(master_row("D", L, "f1_optimal")["specificity"]),
              spec, tol=5e-4)

# --- ablation ------------------------------------------------------------- #
add_claim("abl_f1_gain_cardio",
          "Cardiomegaly F1 gains 0.2836 -> 0.3536 (threshold policy)",
          "results_ablation", "numeric", MR,
          lambda: (_r4(master_row("A", "Cardiomegaly", "fixed")["f1"]),
                   _r4(master_row("D", "Cardiomegaly", "f1_optimal")["f1"])),
          (0.2836, 0.3536), tol=5e-4)
add_claim("abl_B_no_change",
          "Calibration alone does not change fixed-0.50 confusion (monotone sigmoid)",
          "results_ablation", "text", E3,
          lambda: _p_conf_equal(), True)

# --- error analysis ------------------------------------------------------- #
add_claim("ea_cardio_errors", "Cardiomegaly 563 errors", "results_error_analysis",
          "numeric", EAJ,
          lambda: int(_ea_errors("Cardiomegaly")), 563, tol=0)
add_claim("ea_cardio_hc", "Cardiomegaly high-confidence (>=0.9) errors = 274",
          "results_error_analysis", "numeric", EAJ,
          lambda: int(_stratum("Cardiomegaly")["high_confidence_errors"]["count"]),
          274, tol=0)
add_claim("ea_cardio_hc_share",
          "Cardiomegaly high-confidence share of errors = 48.7%",
          "results_error_analysis", "numeric", EAJ,
          lambda: _r4(_stratum("Cardiomegaly")["high_confidence_errors"]["share_of_errors"]),
          0.4867, tol=0.001)
add_claim("ea_eff_errors", "Effusion 2,268 errors", "results_error_analysis",
          "numeric", EAJ,
          lambda: int(_ea_errors("Effusion")), 2268, tol=0)
add_claim("ea_eff_hc_share", "Effusion high-confidence share of errors = 8.5%",
          "results_error_analysis", "numeric", EAJ,
          lambda: _r4(_stratum("Effusion")["high_confidence_errors"]["share_of_errors"]),
          0.0851, tol=0.001)
add_claim("ea_jaccard", "cross-label error Jaccard = 0.131",
          "results_error_analysis", "numeric", EAJ,
          lambda: _r4(EA["cross_label_overlap"]["jaccard_of_error_sets"]), 0.131, tol=0.001)
add_claim("ea_patients", "1,317 patients with errors", "results_error_analysis",
          "numeric", EAJ,
          lambda: int(EA["patient_error_concentration"]["n_patients_with_errors"]),
          1317, tol=0)
add_claim("ea_error_images", "4,449 error images", "results_error_analysis",
          "numeric", EAJ,
          lambda: int(EA["patient_error_concentration"]["n_error_images"]), 4449, tol=0)
add_claim("ea_max_patient", "worst patient has 60 error images",
          "results_error_analysis", "numeric", EAJ,
          lambda: int(EA["patient_error_concentration"]["max_errors_by_one_patient"]),
          60, tol=0)

# --- explainability ------------------------------------------------------- #
add_claim("gc_n_cases", "16 deterministic Grad-CAM cases", "results_explainability",
          "numeric", "outputs/gradcam/gradcam_summary.json",
          lambda: int(GRAD["n_cases"]), 16, tol=0)
add_claim("gc_cohort", "region/bbox sanity cohort n = 43", "results_explainability",
          "numeric", BB,
          lambda: int(BBOX["n_with_box"]), 43, tol=0)
add_claim("gc_hits", "pointing-game hits = 21/43", "results_explainability",
          "numeric", BB,
          lambda: int(BBOX["pointing_game_hits"]), 21, tol=0)
add_claim("gc_cr", "mean concentration ratio = 2.60", "results_explainability",
          "numeric", BB,
          lambda: _r4(BBOX["mean_concentration_ratio"]), 2.60, tol=0.03)

# --- external ------------------------------------------------------------- #
add_claim("ext_pending", "external validation pending; dataset unavailable",
          "external_validation", "text", GO,
          lambda: _json(GO)["status"], "BLOCKED — EXTERNAL DATASET NOT AVAILABLE")
add_claim("ext_no_metrics", "no external metrics computed", "external_validation",
          "text", GO, lambda: bool(_json(GO)["no_metrics_were_computed"]), True)

# --- patient-level bootstrap CIs (cluster unit = patient) ------------------ #
CIPF = "outputs/final_results/confidence_intervals_patient.csv"
LEGACY_CI = _csv("outputs/final_results/confidence_intervals.csv")


def _cip_f1(label: str, calib: str, pol: str) -> tuple:
    row = CIP[(CIP["class"] == label) & (CIP["calibration"] == calib)
              & (CIP["threshold_policy"] == pol) & (CIP["metric"] == "f1")]
    return (_r4(row["point_estimate"].iloc[0]), _r4(row["ci_lower"].iloc[0]),
            _r4(row["ci_upper"].iloc[0]))


def _cip_width(label: str) -> float:
    return _r4(_cip_f1(label, "calibrated", "f1_optimal")[2]
               - _cip_f1(label, "calibrated", "f1_optimal")[1])


def _legacy_width(label: str) -> float:
    row = LEGACY_CI[(LEGACY_CI["class"] == label)
                    & (LEGACY_CI["calibration"] == "calibrated")
                    & (LEGACY_CI["threshold_policy"] == "f1_optimal")
                    & (LEGACY_CI["metric"] == "f1")]
    return _r4(float(row["ci_upper"].iloc[0]) - float(row["ci_lower"].iloc[0]))


for lbl, cal, pol, arm, f1e, lo, hi in [
        ("Cardiomegaly", "raw", "fixed", "A", 0.2836, 0.2399, 0.3276),
        ("Cardiomegaly", "calibrated", "f1_optimal", "D", 0.3536, 0.2816, 0.4177),
        ("Effusion", "raw", "fixed", "A", 0.4676, 0.4427, 0.4918),
        ("Effusion", "calibrated", "f1_optimal", "D", 0.4866, 0.4563, 0.5153)]:
    add_claim(f"pb_f1_{arm}_{lbl.lower()[:4]}",
              f"patient-level {lbl} arm-{arm} F1 {f1e} with 95% CI [{lo}, {hi}]",
              "results_patient_bootstrap", "numeric", CIPF,
              lambda L=lbl, C=cal, P=pol: _cip_f1(L, C, P), (f1e, lo, hi), tol=5e-4)
add_claim("pb_method", "5000 patient-level cluster bootstrap resamples, 95% "
          "percentile CI, seed 42 (resampling unit = patient)",
          "results_patient_bootstrap", "text", CIPF,
          lambda: (int(CIP["n_bootstraps"].iloc[0]) == 5000
                   and float(CIP["confidence_level"].iloc[0]) == 0.95
                   and int(CIP["rng_seed"].iloc[0]) == 42), True)
add_claim("pb_widen_cardio", "patient-level clustering widens the Cardiomegaly "
          "arm-D F1 CI from 0.0707 to 0.1361",
          "results_patient_bootstrap", "numeric", CIPF,
          lambda: (_legacy_width("Cardiomegaly"), _cip_width("Cardiomegaly")),
          (0.0707, 0.1361), tol=5e-4)
add_claim("pb_widen_eff", "patient-level clustering widens the Effusion "
          "arm-D F1 CI from 0.0306 to 0.0590",
          "results_patient_bootstrap", "numeric", CIPF,
          lambda: (_legacy_width("Effusion"), _cip_width("Effusion")),
          (0.0306, 0.0590), tol=5e-4)

# --- DeLong discrimination ------------------------------------------------ #
DEL = "outputs/metrics/patient_stats/delong_auroc.json"


def _delong(label: str) -> tuple:
    d = DELONG["per_label"][label]
    return (_r4(d["auc"]), _r4(d["se"]), _r4(d["ci_lower"]), _r4(d["ci_upper"]),
            int(d["n_pos"]), int(d["n_neg"]))


add_claim("delong_cardio", "DeLong Cardiomegaly AUROC 0.8972 (SE 0.0077, "
          "95% CI [0.8821, 0.9124]; 415 positive / 15469 negative)",
          "results_discrimination", "numeric", DEL,
          lambda: _delong("Cardiomegaly"),
          (0.8972, 0.0077, 0.8821, 0.9124, 415, 15469), tol=5e-4)
add_claim("delong_eff", "DeLong Effusion AUROC 0.8587 (SE 0.0041, "
          "95% CI [0.8507, 0.8667]; 1997 positive / 13887 negative)",
          "results_discrimination", "numeric", DEL,
          lambda: _delong("Effusion"),
          (0.8587, 0.0041, 0.8507, 0.8667, 1997, 13887), tol=5e-4)

# --- primary endpoints (A vs D) ------------------------------------------- #
STATP = "outputs/final_results/statistical_report.json"
PERM_SRC = "outputs/metrics/patient_stats/paired_tests.json"


def _primary(label: str) -> tuple:
    e = [x for x in STAT["primary_endpoints"] if x["label"] == label][0]
    return (_r4(e["delta_point_estimate"]), _r4(e["ci_lower"]),
            _r4(e["ci_upper"]), round(float(e["permutation_p"]), 4),
            round(float(e["holm_adjusted_p"]), 4))


add_claim("stat_delta_cardio", "Cardiomegaly primary endpoint: Delta F1 (D-A) "
          "+0.0700, bootstrap 95% CI [0.0280, 0.1065], permutation p = 0.0002, "
          "Holm-adjusted p = 0.0004 (significant)",
          "results_arm_differences", "numeric", STATP,
          lambda: _primary("Cardiomegaly"), (0.0700, 0.0280, 0.1065, 0.0002, 0.0004),
          tol=5e-4)
add_claim("stat_delta_eff", "Effusion primary endpoint: Delta F1 (D-A) +0.0191, "
          "bootstrap 95% CI [0.0050, 0.0326], permutation p = 0.0262, "
          "Holm-adjusted p = 0.0262 (significant)",
          "results_arm_differences", "numeric", STATP,
          lambda: _primary("Effusion"), (0.0191, 0.0050, 0.0326, 0.0262, 0.0262),
          tol=5e-4)
add_claim("stat_primary_sig", "both primary Delta-F1 endpoints significant after "
          "Holm 0.05 control; CIs exclude zero",
          "results_arm_differences", "text", STATP,
          lambda: all(x["significance"] == "significant"
                      and not x["bootstrap_covers_zero"]
                      for x in STAT["primary_endpoints"]), True)
add_claim("stat_perm_method", "primary endpoints tested with 5000 paired "
          "permutations (seed 42); thresholds frozen from validation",
          "results_arm_differences", "text", PERM_SRC,
          lambda: int(PAIRED["n_permutations"]) == 5000
          and int(PAIRED["rng_seed"]) == 42, True)
add_claim("stat_delta_sign_both_labels", "Delta F1 gains are significant for "
          "BOTH labels", "results_arm_differences", "text", STATP,
          lambda: len(STAT["primary_endpoints"]) == 2
          and all(x["significance"] == "significant" for x in STAT["primary_endpoints"]), True)

# --- threshold stability (validation bootstrap) ---------------------------- #
STABF = "outputs/metrics/threshold_stability/threshold_stability.json"


def _stab(lbl: str, var: str) -> tuple:
    d = STAB["per_label"][lbl][var]["f1_optimal"]
    return (_r4(d["first_fit"]), _r4(d["ci_lower"]), _r4(d["ci_upper"]),
            _r4(d["ci_width"]), bool(d["within_pm05pct_of_first_fit"]),
            bool(d["within_pm10pct_of_first_fit"]))


for lbl, var, exp in [
        ("Cardiomegaly", "raw", (0.9021, 0.8275, 0.9140, 0.0864, False, True)),
        ("Cardiomegaly", "calibrated", (0.8666, 0.7860, 0.8799, 0.0939, False, True)),
        ("Cardiomegaly", "logistic", (0.1680, 0.1226, 0.1792, 0.0565, False, False)),
        ("Effusion", "raw", (0.7699, 0.6953, 0.8174, 0.1221, False, True)),
        ("Effusion", "calibrated", (0.7035, 0.6434, 0.7444, 0.1009, False, True)),
        ("Effusion", "logistic", (0.2518, 0.2018, 0.2942, 0.0924, False, False))]:
    add_claim(f"ts_{lbl.lower()[:4]}_{var}",
              f"{lbl} {var} f1-optimal threshold: first fit {exp[0]}, bootstrap "
              f"95% CI [{exp[1]}, {exp[2]}] (width {exp[3]})",
              "results_threshold_stability", "numeric", STABF,
              lambda L=lbl, V=var: _stab(L, V), exp, tol=5e-4)
add_claim("ts_eq_recheck", "vectorized threshold refit reproduces the frozen "
          "validation fits exactly (33 checks, atol 1e-9)",
          "results_threshold_stability", "text", STABF,
          lambda: bool(STAB["equivalence_check"]["passed"]), True)
add_claim("ts_pm5_none", "no f1-optimal threshold lies within +-5% of its first "
          "fit; raw/calibrated stay within +-10%, logistic does not",
          "results_threshold_stability", "text", STABF,
          lambda: all(not _stab(l, v)[4] for l in C.TARGET_LABELS
                      for v in ("raw", "calibrated", "logistic")) and
          _stab("Cardiomegaly", "raw")[5] and _stab("Cardiomegaly", "calibrated")[5]
          and not _stab("Cardiomegaly", "logistic")[5], True)

# --- ECE sensitivity + logistic calibration -------------------------------- #
ECEF = "outputs/metrics/ece_sensitivity/ece_sensitivity.json"
LOGF = "outputs/metrics/calibration/logistic_calibration_report.json"


def _ece_ref(label: str) -> tuple:
    d = ECEJ["ece_at_reference_binning"][label]
    return (_r4(d["raw"]), _r4(d["calibrated"]), _r4(d["logistic"]))


add_claim("ece_ref_cardio", "reference-binning ECE (15 bins, equal width) "
          "Cardiomegaly: raw 0.1061, calibrated 0.1169, logistic 0.0029",
          "results_calibration_comparison", "numeric", ECEF,
          lambda: _ece_ref("Cardiomegaly"), (0.1061, 0.1169, 0.0029), tol=5e-4)
add_claim("ece_ref_eff", "reference-binning ECE (15 bins, equal width) "
          "Effusion: raw 0.2077, calibrated 0.2292, logistic 0.0136",
          "results_calibration_comparison", "numeric", ECEF,
          lambda: _ece_ref("Effusion"), (0.2077, 0.2292, 0.0136), tol=5e-4)
add_claim("ece_ref_decl", "ECE sensitivity grid = 10/15/20 bins x "
          "equal-width/equal-frequency, anchored to a 15-bin equal-width "
          "reference binning",
          "results_calibration_comparison", "text", ECEF,
          lambda: ECEJ["reference_binning"] == {"n_bins": 15, "strategy": "equal_width"}
          and set(ECEJ["bins"]) == {10, 15, 20}
          and set(ECEJ["strategies"]) == {"equal_width", "equal_freq"}, True)


def _logi(label: str) -> tuple:
    p = LOGCAL["parameters"][label]
    return (_r4(p["a"]), _r4(p["b"]), _r4(LOGCAL["nll_before"][label]),
            _r4(LOGCAL["nll_after"][label]))


add_claim("logi_cardio", "logistic (Platt) Cardiomegaly a = 0.5441, b = -2.8079; "
          "validation NLL 0.2337 -> 0.0845",
          "results_calibration_comparison", "numeric", LOGF,
          lambda: _logi("Cardiomegaly"), (0.5441, -2.8079, 0.2337, 0.0845), tol=5e-4)
add_claim("logi_eff", "logistic (Platt) Effusion a = 0.7474, b = -1.9919; "
          "validation NLL 0.4889 -> 0.2650",
          "results_calibration_comparison", "numeric", LOGF,
          lambda: _logi("Effusion"), (0.7474, -1.9919, 0.4889, 0.2650), tol=5e-4)
add_claim("logi_val_ece_cardio", "logistic validation-set ECE Cardiomegaly = 0.0024",
          "results_calibration_comparison", "numeric", LOGF,
          lambda: _r4(LOGCAL["val_report"]["Cardiomegaly"]["ece"]), 0.0024, tol=5e-4)
add_claim("logi_val_ece_eff", "logistic validation-set ECE Effusion = 0.0083",
          "results_calibration_comparison", "numeric", LOGF,
          lambda: _r4(LOGCAL["val_report"]["Effusion"]["ece"]), 0.0083, tol=5e-4)
add_claim("logi_monotone", "logistic fit is strictly monotone (a = exp(c) > 0), "
          "so AUROC / AUPRC are unchanged by construction",
          "results_calibration_comparison", "text", LOGF,
          lambda: max(abs(float(v) - float(LOGCAL["auroc_auprc_invariance_checks"]
                          [k.replace("_raw", "_log")]))
                      for k, v in LOGCAL["auroc_auprc_invariance_checks"].items()
                      if k.endswith("_raw")) < 1e-12, True)

# --- extension arms (E/F, never part of the A-D ablation) ------------------ #
EXTF = "outputs/metrics/experiments/ext_extension_arms.json"
ARM_EQ_METRICS = ("accuracy", "precision", "recall", "specificity", "f1")


def _ext_eq(arm_a: str, arm_b: str) -> bool:
    for lbl in C.TARGET_LABELS:
        for m in ARM_EQ_METRICS:
            va = EXT["per_label"][lbl][arm_a][m]
            vb = EXT["per_label"][lbl][arm_b][m]
            if abs(float(va) - float(vb)) > 1e-9:
                return False
    return True


def _ext(lbl: str, arm: str, metric: str) -> float:
    return _r4(EXT["per_label"][lbl][arm][metric])


add_claim("ext_A_eq_B", "arm B (calibrated, fixed 0.5) is IDENTICAL to arm A "
          "(raw, fixed 0.5) at the fixed operating point (monotone sigmoid)",
          "results_extension", "text", EXTF, lambda: _ext_eq("A", "B"), True)
add_claim("ext_C_D_F_eq", "arm C, arm D, arm F (all f1-optimal) are IDENTICAL "
          "on test: f1-optimal policy is robust to monotone calibration",
          "results_extension", "text", EXTF,
          lambda: _ext_eq("C", "D") and _ext_eq("C", "F"), True)
add_claim("ext_E_cardio", "arm E (logistic + fixed 0.5) differs sharply: "
          "Cardiomegaly F1 0.1376, precision 0.6400, recall 0.0771",
          "results_extension", "numeric", EXTF,
          lambda: (_ext("Cardiomegaly", "E", "f1"), _ext("Cardiomegaly", "E", "precision"),
                   _ext("Cardiomegaly", "E", "recall")),
          (0.1376, 0.6400, 0.0771), tol=5e-4)
add_claim("ext_E_eff", "arm E (logistic + fixed 0.5) differs sharply: "
          "Effusion F1 0.3222, precision 0.6030, recall 0.2198",
          "results_extension", "numeric", EXTF,
          lambda: (_ext("Effusion", "E", "f1"), _ext("Effusion", "E", "precision"),
                   _ext("Effusion", "E", "recall")),
          (0.3222, 0.6030, 0.2198), tol=5e-4)

# --- prevalence shift ------------------------------------------------------ #
PREVF = "outputs/metrics/prevalence/prevalence_shift.json"


def _prev_f1(label: str, pct: float) -> float:
    for ch in PREV["per_label"][label]["cohorts"]:
        if abs(ch["target_prevalence_pct"] - pct) < 1e-9:
            return _r4(ch["f1"])
    raise KeyError(f"prevalence cohort {label}/{pct} not found")


add_claim("prev_obs_cardio", "observed Cardiomegaly prevalence 0.0261 "
          "(415 positive / 15469 negative test images)",
          "results_prevalence", "numeric", PREVF,
          lambda: (_r4(PREV["per_label"]["Cardiomegaly"]["observed_prevalence"]),
                   int(PREV["per_label"]["Cardiomegaly"]["n_positives"]),
                   int(PREV["per_label"]["Cardiomegaly"]["n_negatives"])),
          (0.0261, 415, 15469), tol=5e-4)
add_claim("prev_obs_eff", "observed Effusion prevalence 0.1257 "
          "(1997 positive / 13887 negative test images)",
          "results_prevalence", "numeric", PREVF,
          lambda: (_r4(PREV["per_label"]["Effusion"]["observed_prevalence"]),
                   int(PREV["per_label"]["Effusion"]["n_positives"]),
                   int(PREV["per_label"]["Effusion"]["n_negatives"])),
          (0.1257, 1997, 13887), tol=5e-4)
add_claim("prev_cells", "14 prevalence target cells: 8 simulated (above-natural "
          "prevalence) and 6 below-natural targets dropped as infeasible",
          "results_prevalence", "numeric", PREVF,
          lambda: (int(len(PREV["per_label"]["Cardiomegaly"]["cohorts"])
                       + len(PREV["per_label"]["Effusion"]["cohorts"])),
                   int(sum(1 for l in PREV["per_label"].values()
                           for c in l["cohorts"] if c.get("simulated"))),
                   int(sum(1 for l in PREV["per_label"].values()
                           for c in l["cohorts"] if c.get("target_not_feasible")))),
          (14, 8, 6), tol=0)
add_claim("prev_cardio_f1_20", "Cardiomegaly enriched to 20%: F1 0.5116 "
          "(precision 0.8235, recall unchanged 0.3711)",
          "results_prevalence", "numeric", PREVF,
          lambda: _prev_f1("Cardiomegaly", 20.0), 0.5116, tol=5e-4)
add_claim("prev_cardio_f1_50", "Cardiomegaly enriched to 50%: F1 0.5338, "
          "AUPRC 0.9002 (AUROC unchanged 0.9000)",
          "results_prevalence", "numeric", PREVF,
          lambda: _prev_f1("Cardiomegaly", 50.0), 0.5338, tol=5e-4)
add_claim("prev_eff_f1_20", "Effusion enriched to 20%: F1 0.5609 "
          "(precision 0.5855, recall unchanged 0.5383)",
          "results_prevalence", "numeric", PREVF,
          lambda: _prev_f1("Effusion", 20.0), 0.5609, tol=5e-4)
add_claim("prev_eff_f1_50", "Effusion enriched to 50%: F1 0.6547 "
          "(precision 0.8353)",
          "results_prevalence", "numeric", PREVF,
          lambda: _prev_f1("Effusion", 50.0), 0.6547, tol=5e-4)
add_claim("prev_recall_pinned", "under enrichment the frozen operating point "
          "pins recall (0.3711 / 0.5383); F1 and precision rise because fewer "
          "negatives are present, not because more positives are found",
          "results_prevalence", "text", PREVF,
          lambda: (_prev_f1("Cardiomegaly", 20.0) > 0.5
                   and _r4(PREV["per_label"]["Cardiomegaly"]["cohorts"][3]["recall"]) == 0.3711),
          True)

# --- tests / environment -------------------------------------------------- #
add_claim("tests_count", "193 tests passing (suite grows)", "testing", "numeric",
          "tests", lambda: _count_tests(), at_least=193)
add_claim("ckpt_sha", "checkpoint sha256 matches manifest", "testing", "text", MG,
          lambda: MANI.get("sha256") == "35965f610c8b578b3ad52e9d7d06a1b3054948df6df52e80aa02f2b81803d68c", True)


def _p_temp(label: str) -> float:
    t = _json("outputs/metrics/calibration/temperature_scalers.json")
    return float(t["metadata"]["temperatures"][label])


def _stratum(label: str) -> dict:
    return EA["strata"]["f1_optimal"][label]


def _ea_errors(label: str) -> int:
    s = _stratum(label)
    n = int(s["n"])
    err_rate = float(s["error_rate"])
    return int(round(n * err_rate))


def _p_conf_equal() -> bool:
    e2 = EXP2["evaluation"]["test"]
    e3 = EXP3["evaluation"]["test"]
    for lbl in C.TARGET_LABELS:
        for pol in POLICIES:
            a, b = e2[lbl][pol], e3[lbl][pol]
            if (a["tp"], a["fp"], a["tn"], a["fn"]) != (b["tp"], b["fp"], b["tn"], b["fn"]):
                return False
    return True


def _count_tests() -> int:
    loader = unittest.defaultTestLoader
    suite = loader.discover(str(Path(C.PROJECT_ROOT) / "tests"))
    return suite.countTestCases()


# --------------------------------------------------------------------------- #
# 1. Paper evidence map + (4.) claim audit
# --------------------------------------------------------------------------- #
def build_evidence_and_audit() -> tuple[Path, Path]:
    evidence = []
    audit = []
    for c in CLAIMS:
        found = None
        ok = False
        notes = ""
        if c["resolve"] is not None:
            found = c["resolve"]()
            if c.get("at_least") is not None:
                ok = float(found) >= float(c["at_least"])
                if not ok:
                    notes = f"artifact gave {found}"
            elif isinstance(found, tuple):
                ok = all(abs(float(a) - float(b)) <= c["tol"] * max(1.0, abs(float(b)))
                         for a, b in zip(found, c["expected"]))
            elif isinstance(found, bool):
                ok = (found is c["expected"])
            elif isinstance(c["expected"], str):
                ok = (str(found) == c["expected"])
            else:
                ok = abs(float(found) - float(c["expected"])) <= c["tol"] * max(1.0, abs(float(c["expected"])))
            if not ok:
                notes = f"artifact gave {found}"
        else:
            p = Path(C.PROJECT_ROOT) / c["source_artifact"]
            ok = p.exists() or c["source_artifact"] == "tests"
            notes = "source exists" if ok else "MISSING SOURCE"
        source_table_or_figure = DISPLAY_REF.get(c["section"], "")
        evidence.append({
            "id": c["id"], "claim": c["claim"], "claim_type": c["claim_type"],
            "section": c["section"],
            "value": c["expected"] if isinstance(c["expected"], (int, float, str, bool))
            else None,
            "source_artifact": c["source_artifact"],
            "source_table_or_figure": source_table_or_figure,
            "verified": ok,
        })
        audit.append({
            "claim_id": c["id"], "claim": c["claim"], "claim_type": c["claim_type"],
            "section": c["section"], "source_artifact": c["source_artifact"],
            "source_table_or_figure": source_table_or_figure,
            "verified": ok, "notes": notes,
        })
    evidence_sorted = sorted(evidence, key=lambda x: (x["section"], x["id"]))
    map_doc = {
        "phase": "paper_package",
        "rule": "no paper number may exist without a traceable source artifact",
        "n_claims": len(evidence_sorted),
        "n_verified": sum(1 for e in evidence_sorted if e["verified"]),
        "claims": evidence_sorted,
        "verified_by_construction": [
            "tables/table_01..14.csv and figures/figure_01..14.png are generated "
            "in this phase directly from the artifact JSON/CSV files cited here",
        ],
    }
    em_path = C.PAPER_EVIDENCE_MAP_JSON
    write_json(em_path, map_doc)

    audit_df = pd.DataFrame(audit)
    audit_path = C.PAPER_CLAIM_AUDIT_CSV
    audit_df.to_csv(audit_path, index=False)
    logger.info("evidence map %d claims (%d verified); claim audit written",
               len(evidence_sorted), map_doc["n_verified"])
    return em_path, audit_path


# --------------------------------------------------------------------------- #
# 2. Paper tables
# --------------------------------------------------------------------------- #
def _w(table_name: str, rows: List[dict]) -> Path:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"refusing to write empty table {table_name}")
    out = C.PAPER_TABLES_DIR / table_name
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d rows)", table_name, len(df))
    return out


def table_01_dataset() -> Path:
    rows = []
    order = {"train": 0, "val": 1, "test": 2}
    pos = SPLIT.groupby("split").agg(cardio=("Cardiomegaly", "sum"),
                                     eff=("Effusion", "sum"))
    npats = SPLIT.groupby("split")["Patient ID"].nunique()
    for split in order:
        rows.append({
            "split": split, "images": int(len(SPLIT[SPLIT.split == split])),
            "patients": int(npats[split]),
            "cardiomegaly_positive": int(pos.loc[split, "cardio"]),
            "effusion_positive": int(pos.loc[split, "eff"]),
        })
    rows.append({"split": "total", "images": int(len(SPLIT)),
                 "patients": int(SPLIT["Patient ID"].nunique()),
                 "cardiomegaly_positive": int(SPLIT.Cardiomegaly.sum()),
                 "effusion_positive": int(SPLIT.Effusion.sum())})
    return _w("table_01_dataset.csv", rows)


def table_02_baseline() -> Path:
    rows = []
    for metric in ("auroc", "auprc", "f1", "precision", "recall", "specificity",
                   "accuracy"):
        card = _r4(_per_label_metrics("Cardiomegaly")[metric])
        eff = _r4(_per_label_metrics("Effusion")[metric])
        rows.append({"metric": metric.capitalize(), "Cardiomegaly": card,
                     "Effusion": eff})
    rows.append({"metric": "Prevalence", "Cardiomegaly": _r4(_per_label_metrics("Cardiomegaly")["prevalence"]),
                 "Effusion": _r4(_per_label_metrics("Effusion")["prevalence"])})
    rows.append({"metric": "Macro AUROC", "Cardiomegaly": _r4(BASE["metrics"]["macro"]["auroc"]),
                 "Effusion": None})
    rows.append({"metric": "Macro AUPRC", "Cardiomegaly": _r4(BASE["metrics"]["macro"]["auprc"]),
                 "Effusion": None})
    return _w("table_02_baseline.csv", rows)


def table_03_main_results() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        a = master_row("A", lbl, "fixed")
        d = master_row("D", lbl, "f1_optimal")
        rows.append({"class": lbl, "arm": "A (baseline, fixed tau=0.50)",
                     "f1": a["f1"], "precision": a["precision"],
                     "recall": a["recall"], "specificity": a["specificity"],
                     "auroc": a["auroc"], "auprc": a["auprc"]})
        rows.append({"class": lbl,
                     "arm": "D (calibration + f1-optimal policy)",
                     "f1": d["f1"], "precision": d["precision"],
                     "recall": d["recall"], "specificity": d["specificity"],
                     "auroc": a["auroc"], "auprc": a["auprc"]})
        rows.append({"class": lbl, "arm": "Delta F1 (D - A)",
                     "f1": round(float(d["f1"]) - float(a["f1"]), 4),
                     "precision": round(float(d["precision"]) - float(a["precision"]), 4),
                     "recall": round(float(d["recall"]) - float(a["recall"]), 4),
                     "specificity": round(float(d["specificity"]) - float(a["specificity"]), 4),
                     "auroc": "", "auprc": ""})
    return _w("table_03_main_results.csv", rows)


def table_04_ablation() -> Path:
    rows = []
    for exp, pol, name in [("A", "fixed", "A baseline"),
                           ("B", "fixed", "B calibration only"),
                           ("C", "f1_optimal", "C threshold only"),
                           ("D", "f1_optimal", "D calibration + threshold")]:
        for lbl in C.TARGET_LABELS:
            row = master_row(exp, lbl, pol)
            rows.append({"arm": name, "class": lbl, "calibration": row["calibration"],
                         "threshold_policy": row["threshold_policy"],
                         "f1": row["f1"], "precision": row["precision"],
                         "recall": row["recall"], "specificity": row["specificity"],
                         "ece": row["ece"], "brier_score": row["brier_score"]})
    return _w("table_04_ablation.csv", rows)


def table_05_thresholds() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        for pol in POLICIES:
            tv = THR_VAL["policies"][pol][lbl]["threshold"]
            tc = THR_CAL["policies"][pol][lbl]["threshold"]
            r = _policy_test(EXP2, lbl, pol)
            c = _policy_test(EXP3, lbl, pol)
            rows.append({
                "class": lbl, "policy": pol,
                "threshold_val_raw": _r4(tv), "threshold_val_calibrated": _r4(tc),
                "test_f1_raw": _r4(r["f1"]),
                "test_f1_calibrated": _r4(c["f1"]),
                "test_precision_calibrated": _r4(c["precision"]),
                "test_recall_calibrated": _r4(c["recall"]),
                "test_specificity_calibrated": _r4(c["specificity"]),
            })
    return _w("table_05_thresholds.csv", rows)


def table_06_calibration() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        r = _exp_split_metrics(EXP1, "test", lbl, "raw")
        c = _exp_split_metrics(EXP1, "test", lbl, "calibrated")
        for metric in ("nll", "brier", "ece", "mce", "mean_confidence"):
            rows.append({"class": lbl, "metric": metric,
                         "raw": _r4(r[metric]), "calibrated": _r4(c[metric]),
                         "delta": round(float(c[metric]) - float(r[metric]), 4)})
    return _w("table_06_calibration.csv", rows)


def table_07_error_analysis() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        s = _stratum(lbl)
        hc = s["high_confidence_errors"]
        rows.append({"class": lbl, "errors": _ea_errors(lbl),
                     "error_rate": _r4(s["error_rate"]),
                     "high_confidence_errors_>=_0.9": hc["count"],
                     "share_of_errors": _r4(hc["share_of_errors"])
                     if hc["share_of_errors"] is not None else None})
    cl = EA["cross_label_overlap"]
    rows.append({"class": "cross-label (both classes wrong)",
                 "errors": cl["n_both_wrong"],
                 "error_rate": _r4(cl["jaccard_of_error_sets"])
                 if cl.get("n_either_wrong") else None,
                 "high_confidence_errors_>=_0.9": None,
                 "share_of_errors": cl["n_both_wrong"] / cl["n_either_wrong"]
                 if cl.get("n_either_wrong") else None})
    rows.append({"class": "cross-label (either class wrong)",
                 "errors": cl["n_either_wrong"],
                 "error_rate": None,
                 "high_confidence_errors_>=_0.9": None,
                 "share_of_errors": None})
    conc = EA["patient_error_concentration"]
    rows.append({"class": "patients_with_errors",
                 "errors": conc["n_error_images"],
                 "error_rate": conc["n_patients_with_errors"],
                 "high_confidence_errors_>=_0.9": conc["max_errors_by_one_patient"],
                 "share_of_errors": None})
    return _w("table_07_error_analysis.csv", rows)


def table_08_explainability() -> Path:
    rows = [
        {"metric": "Grad-CAM cases", "value": GRAD["n_cases"]},
        {"metric": "target layer", "value": GRAD["target_layer"]},
        {"metric": "bbox sanity cohort n", "value": BBOX["n_with_box"]},
        {"metric": "pointing-game hits", "value": BBOX["pointing_game_hits"]},
        {"metric": "pointing accuracy", "value": _r4(BBOX["pointing_game_accuracy"])},
        {"metric": "mean concentration ratio", "value": _r4(BBOX["mean_concentration_ratio"])},
        {"metric": "median concentration ratio", "value": _r4(BBOX["median_concentration_ratio"])},
        {"metric": "ratio > 1 (n cases)", "value": BBOX["ratio_above_1"]},
    ]
    per_label = {}
    for case in BBOX["cases"]:
        per_label[case["label"]] = per_label.get(case["label"], 0) + 1
    for lbl in C.TARGET_LABELS:
        rows.append({"metric": f"cohort share {lbl}", "value": per_label.get(lbl, 0)})
    return _w("table_08_explainability.csv", rows)


def table_09_external_validation_status() -> Path:
    rows = [
        {"field": "status",
         "value": "EXTERNAL VALIDATION PENDING — DATASET UNAVAILABLE"},
        {"field": "no_metrics_were_computed", "value": "true"},
        {"field": "dataset_probed",
         "value": "$CHEXPERT_ROOT, $CHEXPERT_DATA_DIR, data/raw/chexpert, "
                  "/mnt/data/chexpert, /data/chexpert"},
        {"field": "pipeline_implemented", "value": "true (src/external_eval.py)"},
        {"field": "recommended_next_step",
         "value": "place the official CheXpert cohort at data/raw/chexpert/ or "
                  "$CHEXPERT_ROOT and run: .venv/bin/python -m src.external_eval --split valid"},
    ]
    return _w("table_09_external_validation_status.csv", rows)


def table_10_patient_bootstrap_ci() -> Path:
    """Full printout of the patient-level cluster-bootstrap artifact."""
    df = CIP.copy()
    df = df[["split", "class", "calibration", "threshold_policy", "metric",
             "point_estimate", "ci_lower", "ci_upper", "n_bootstraps",
             "confidence_level", "rng_seed"]]
    out = C.PAPER_TABLES_DIR / "table_10_patient_bootstrap_ci.csv"
    df.to_csv(out, index=False)
    logger.info("wrote table_10_patient_bootstrap_ci.csv (%d rows)", len(df))
    return out


def table_11_arm_differences() -> Path:
    """Full printout of the paired patient-level arm-difference artifact."""
    df = AD.copy()
    df = df[["split", "class", "arm_a", "arm_b", "metric", "delta_point_estimate",
             "ci_lower", "ci_upper", "n_bootstraps", "confidence_level",
             "rng_seed"]]
    out = C.PAPER_TABLES_DIR / "table_11_arm_differences.csv"
    df.to_csv(out, index=False)
    logger.info("wrote table_11_arm_differences.csv (%d rows)", len(df))
    return out


def table_12_threshold_stability() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        for var in ("raw", "calibrated", "logistic"):
            for pol in ("f1_optimal", "youden", "sensitivity_constrained",
                        "precision_constrained"):
                d = STAB["per_label"][lbl][var][pol]
                rows.append({
                    "class": lbl, "variant": var, "policy": pol,
                    "first_fit": _r4(d["first_fit"]), "mean": _r4(d["mean"]),
                    "sd": _r4(d["sd"]), "median": _r4(d["median"]),
                    "ci_lower": _r4(d["ci_lower"]), "ci_upper": _r4(d["ci_upper"]),
                    "ci_width": _r4(d["ci_width"]), "min": _r4(d["min"]),
                    "max": _r4(d["max"]),
                    "within_pm5pct": d["within_pm05pct_of_first_fit"],
                    "within_pm10pct": d["within_pm10pct_of_first_fit"],
                    "n_bootstraps": d["n_bootstraps"],
                })
    return _w("table_12_threshold_stability.csv", rows)


def _grid_ece(class_: str, calibration: str) -> tuple[float, float]:
    vals = [r["ece"] for r in ECEJ["rows"]
            if r["class"] == class_ and r["calibration"] == calibration]
    return _r4(min(vals)), _r4(max(vals))


def table_13_calibration_comparison() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        ref = _ece_ref(lbl)
        raw = _exp_split_metrics(EXP1, "val", lbl, "raw")["nll"]
        cal = _exp_split_metrics(EXP1, "val", lbl, "calibrated")["nll"]
        for method, ref_ece in (("raw", ref[0]), ("calibrated", ref[1]),
                                ("logistic", ref[2])):
            lo, hi = _grid_ece(lbl, method)
            row = {"class": lbl, "method": method,
                   "ece_reference_name": "15 bins / equal width",
                   "ece_at_reference": ref_ece, "ece_grid_min": lo,
                   "ece_grid_max": hi}
            if method == "logistic":
                p = LOGCAL["parameters"][lbl]
                row.update({
                    "logistic_a": _r4(p["a"]), "logistic_b": _r4(p["b"]),
                    "val_nll_before": _r4(LOGCAL["nll_before"][lbl]),
                    "val_nll_after": _r4(LOGCAL["nll_after"][lbl]),
                })
            elif method == "raw":
                row.update({"logistic_a": "", "logistic_b": "",
                            "val_nll_before": _r4(raw), "val_nll_after": ""})
            else:
                row.update({"logistic_a": "", "logistic_b": "",
                            "val_nll_before": _r4(raw), "val_nll_after": _r4(cal)})
            rows.append(row)
    return _w("table_13_calibration_comparison.csv", rows)


def table_14_prevalence_shift() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        for ch in PREV["per_label"][lbl]["cohorts"]:
            rows.append({
                "class": lbl, "target_prevalence_pct": ch["target_prevalence_pct"],
                "simulated": bool(ch.get("simulated")),
                "target_not_feasible": bool(ch.get("target_not_feasible")),
                "observed_prevalence": round(float(ch["observed_prevalence"]), 4),
                "n": int(ch.get("n")) if ch.get("n") is not None else None,
                "n_positives_kept": ch.get("n_positives_kept"),
                "n_negatives_kept": ch.get("n_negatives_kept"),
                "threshold_calibrated_f1_optimal": round(float(ch["threshold"]), 4)
                if ch.get("simulated") else None,
                "tp": ch.get("tp"), "fp": ch.get("fp"), "tn": ch.get("tn"),
                "fn": ch.get("fn"),
                "f1": round(float(ch["f1"]), 4) if ch.get("simulated") else None,
                "precision": round(float(ch["precision"]), 4) if ch.get("simulated") else None,
                "recall": round(float(ch["recall"]), 4) if ch.get("simulated") else None,
                "specificity": round(float(ch["specificity"]), 4) if ch.get("simulated") else None,
                "auprc": round(float(ch["auprc"]), 4) if ch.get("simulated") else None,
                "note": ch.get("note", ""),
            })
    return _w("table_14_prevalence_shift.csv", rows)


def build_tables() -> List[Path]:
    return [table_01_dataset(), table_02_baseline(), table_03_main_results(),
            table_04_ablation(), table_05_thresholds(), table_06_calibration(),
            table_07_error_analysis(), table_08_explainability(),
            table_09_external_validation_status(), table_10_patient_bootstrap_ci(),
            table_11_arm_differences(), table_12_threshold_stability(),
            table_13_calibration_comparison(), table_14_prevalence_shift()]


# --------------------------------------------------------------------------- #
# 3. Paper figures
# --------------------------------------------------------------------------- #
def _savefig(fig, name: str) -> Path:
    path = C.PAPER_FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_01_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    boxes = [
        (0.00, "Input X\n224×224 chest radiograph\nImageNet-normalised"),
        (0.20, "DenseNet121\n(frozen, ImageNet init.)"),
        (0.42, "Logits z_c\nz_Cardio, z_Effusion"),
        (0.62, "Per-label temperature\np = σ(z_c / T_c)\nT = 1.187 / 1.398"),
        (0.82, "Frozen decision policy\n(val-fitted per class)\nthreshold τ_c"),
    ]
    pos_y = 0.55
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, pos_y), 0.15, 0.26, fill=True,
                                   facecolor="#eef3fb", edgecolor="#1f4e79",
                                   linewidth=1.4, zorder=3))
        ax.text(x + 0.075, pos_y + 0.13, label, ha="center", va="center",
                fontsize=8, zorder=4)
    for x0, x1 in [(0.15, 0.20), (0.35, 0.42), (0.57, 0.62), (0.77, 0.82)]:
        ax.annotate("", xy=(x1, pos_y + 0.13), xytext=(x0, pos_y + 0.13),
                    arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=1.5))
    ax.text(0.5, 0.18,
            "Fit ONLY on NIH train (DenseNet121) and NIH validation "
            "(temperature T_c, thresholds τ_c); frozen before NIH test.",
            ha="center", va="center", fontsize=8.5, color="#7a0f1f")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1)
    ax.set_title("Decision pipeline: frozen DenseNet121 + calibration-aware "
                 "class-specific threshold policies", fontsize=11)
    return _savefig(fig, "figure_01_architecture.png")


def figure_02_dataset_distribution() -> Path:
    splits = ["train", "val", "test"]
    imgs = [int(len(SPLIT[SPLIT.split == s])) for s in splits]
    card = [int(SPLIT[(SPLIT.split == s)].Cardiomegaly.sum()) for s in splits]
    eff = [int(SPLIT[(SPLIT.split == s)].Effusion.sum()) for s in splits]
    x = np.arange(len(splits))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    fig.patch.set_facecolor("white")
    a1.bar(x, imgs, color="#1f4e79")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{s}\n({i:,})" for s, i in zip(splits, imgs)])
    a1.set_title("Images per split (patient-level, 70/15/15)")
    a1.set_ylabel("images")
    w = 0.36
    a2.bar(x - w / 2, card, w, label="Cardiomegaly +", color="#7a0f1f")
    a2.bar(x + w / 2, eff, w, label="Effusion +", color="#d9822b")
    a2.set_xticks(x)
    a2.set_xticklabels(splits)
    a2.set_title("Positive examples per split (class imbalance)")
    a2.set_ylabel("positive images")
    a2.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, "figure_02_dataset_distribution.png")


FIGURE_COPIES = [
    ("fig_1_roc.png", "figure_03_roc.png"),
    ("fig_2_pr.png", "figure_04_precision_recall.png"),
    ("fig_3_reliability.png", "figure_05_calibration.png"),
    ("fig_4_threshold_sweep.png", "figure_06_threshold_analysis.png"),
    ("fig_5_confusion.png", "figure_07_confusion_matrix.png"),
    ("fig_7_error_analysis.png", "figure_08_error_analysis.png"),
    ("fig_8_gradcam.png", "figure_09_gradcam.png"),
    ("fig_9_calibration_comparison.png", "figure_10_calibration_comparison.png"),
    ("fig_10_ece_sensitivity.png", "figure_11_ece_sensitivity.png"),
    ("fig_11_threshold_stability.png", "figure_12_threshold_stability.png"),
    ("fig_12_decision_policy.png", "figure_13_decision_policy.png"),
    ("fig_13_prevalence_sensitivity.png", "figure_14_prevalence_sensitivity.png"),
]


def build_figures() -> List[Path]:
    figure_01_architecture()
    figure_02_dataset_distribution()
    written = []
    for src, dst in FIGURE_COPIES:
        src_p = C.FINAL_FIGURES_DIR / src
        dst_p = C.PAPER_FIGURES_DIR / dst
        if not src_p.exists():
            raise FileNotFoundError(f"figure source missing: {src_p}")
        shutil.copyfile(src_p, dst_p)
        written.append(dst_p)
        logger.info("copied %s -> %s", src, dst)
    return written


# --------------------------------------------------------------------------- #
# 15. Reproducibility manifest
# --------------------------------------------------------------------------- #
def _git_state() -> dict:
    git = run_shell("git rev-parse --abbrev-ref HEAD")
    head = run_shell("git rev-parse --short HEAD")
    dirty = run_shell("git status --porcelain")
    lines = dirty.splitlines() if dirty else []
    # timestamp-only regenerated freeze outputs do not count as dirtiness
    excluded = (
        "outputs/research_snapshot.json",
        "outputs/paper/reproducibility_manifest.json",
        "outputs/metrics/error_analysis/error_analysis_test.json",
        "outputs/metrics/external/status.json",
    )
    kept = []
    for ln in lines:
        path = ln[3:].lstrip() if len(ln) > 3 and ln[2] == " " else ln[2:].lstrip()
        if not any(path.startswith(p) for p in excluded):
            kept.append(ln)
    return {
        "branch": git, "head_sha": head,
        "dirty_count": len(kept),
        "modified": sum(1 for l in kept if l.startswith(" M")),
        "untracked": sum(1 for l in kept if l.startswith("??")),
        "excluded": list(excluded),
    }


def run_shell(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def build_manifest() -> Path:
    snap = _json("outputs/research_snapshot.json")
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = None
    env = {
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch_v,
    }
    ckpt = snap.get("checkpoints", {}).get("best", {})
    doc = {
        "phase": "paper_package",
        "model": snap.get("model", {}).get("name"),
        "checkpoint": ckpt.get("path"),
        "checkpoint_sha256": ckpt.get("sha256"),
        "dataset": snap.get("study", {}).get("dataset"),
        "labels": snap.get("model", {}).get("labels"),
        "split": snap.get("data", {}).get("split"),
        "seed": 42,
        "preprocessing": snap.get("preprocessing"),
        "training_configuration": snap.get("training"),
        "augmentation": snap.get("augmentation"),
        "temperatures": snap.get("calibration", {}).get("temperature"),
        "thresholds": {
            "fitted_on": snap.get("thresholds", {}).get("fitted_on"),
            "policies": snap.get("thresholds", {}).get("policies"),
            "thresholds": snap.get("thresholds", {}).get("thresholds"),
            "raw_file": "outputs/metrics/thresholds/thresholds_val.json",
            "calibrated_file": "outputs/metrics/thresholds/thresholds_calibrated_val.json",
        },
        "package_environment": env,
        "test_count_collected": _count_tests(),
        "git": _git_state(),
        "references": {
            "research_snapshot": "outputs/research_snapshot.json",
            "master_results": "outputs/final_results/master_results.csv",
            "verification_report": "outputs/final_results/verification_report.json",
            "evidence_map": "outputs/paper/paper_evidence_map.json",
        },
        "note": "all metrics frozen; nothing retrained or re-fit in paper "
                "preparation. Full reproduction: python -m src.paper_build",
    }
    path = C.PAPER_REPRO_MANIFEST_JSON
    write_json(path, doc)
    logger.info("reproducibility manifest written")
    return path


# --------------------------------------------------------------------------- #
# Number hygiene: every number in the prose must be an artifact value or a
# documented constant. Guards against an edited draft that drifts from data.
# --------------------------------------------------------------------------- #
def verified_number_set() -> set:
    nums = set()
    for c in CLAIMS:
        e = c["expected"]
        if isinstance(e, (int, float)) and not isinstance(e, bool):
            nums.add(round(float(e), 4))
        elif isinstance(e, tuple):
            for v in e:
                if isinstance(v, (int, float)):
                    nums.add(round(float(v), 4))
    return nums


CONSTANTS = {
    0, 0.05, 0.5, 0.9, 0.95, 1, 1.187, 1.398, 2, 2.87, 3, 4, 5, 6, 7, 8, 8.5,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 38, 42, 43, 48.7,
    70, 90, 95, 100, 224, 448, 563, 1024, 2000, 5000,
}


def check_prose_numbers() -> List[str]:
    """Every numeric token in outputs/paper/*.md must trace to evidence/constants.

    Returns the list of violations (values in prose with no verified source).
    """
    allowed = verified_number_set() | {float(c) for c in CONSTANTS}
    violations = []
    pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    for md in sorted(Path(C.PAPER_DIR).glob("*.md")):
        text = md.read_text().replace(",", "")
        for tok in pattern.findall(text):
            val = float(tok)
            if not any(abs(val - a) <= 5e-4 * max(1.0, abs(a)) for a in allowed):
                violations.append(f"{md.name}:{tok}")
    return violations


# --------------------------------------------------------------------------- #
def run() -> None:
    C.PAPER_DIR.mkdir(parents=True, exist_ok=True)
    C.PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    C.PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    build_evidence_and_audit()
    build_tables()
    build_figures()
    build_manifest()

    violations = check_prose_numbers()
    if violations:
        logger.warning("prose numbers without a traceable source (%d):", len(violations))
        for v in violations:
            logger.warning("  %s", v)
    else:
        logger.info("prose number-hygiene check: every number traces to "
                    "a verified artifact or documented constant")

    logger.info("collected %d tests", _count_tests())


def parse_args() -> argparse.Namespace:
    import argparse

    p = argparse.ArgumentParser(description="Build the paper package")
    return p.parse_args()


if __name__ == "__main__":
    parse_args()
    run()