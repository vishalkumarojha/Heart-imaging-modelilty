"""Assemble the consolidated statistical report for the IEEE upgrade.

Pulls together every upgrade artifact into one machine-readable document and
derives the *primary endpoint verdicts*:

    per label: ΔF1 (arm D = calibrated + F1-optimal) − (arm A = raw + fixed 0.50)
    * paired patient-bootstrap CI  (confidence_intervals_patient / arm_differences)
    * paired permutation p-value, Holm-Bonferroni across the two labels

Nothing is measured here — every number is read from an existing artifact, which
is the same invariant the rest of `src/paper_build.py` enforces.

Artifact: outputs/final_results/statistical_report.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config as C
from .reproducibility import write_json
from .utils import setup_logging
from .statistics.data import load_test_predictions

logger = setup_logging()


def _row_from_csv(csv_path: Path, predicate) -> Optional[pd.Series]:
    df = pd.read_csv(csv_path)
    hit = df[predicate(df)]
    if len(hit) != 1:
        return None
    return hit.iloc[0]


def _load_json(path: Path, required: bool = True) -> Optional[dict]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"missing required artifact: {path}")
        return None
    return json.loads(path.read_text())


def run_statistical_report() -> dict:
    df = load_test_predictions()

    diffs = pd.read_csv(C.ARM_DIFFERENCES_CSV)
    paired = _load_json(C.PATIENT_STATS_DIR / "paired_tests.json")
    delong = _load_json(C.PATIENT_STATS_DIR / "delong_auroc.json")
    stability = _load_json(C.THRESHOLD_STABILITY_JSON)
    ece = _load_json(C.ECE_SENSITIVITY_JSON)
    extension = _load_json(C.EXT_EXTENSION_ARMS_JSON)
    prevalence = _load_json(C.PREVALENCE_SHIFT_JSON, required=False)
    if C.DECISION_POLICY_ANALYSIS_CSV.exists():
        decision = pd.read_csv(C.DECISION_POLICY_ANALYSIS_CSV).to_dict(orient="records")
    else:
        decision = None

    primary: List[dict] = []
    for label in C.TARGET_LABELS:
        row = _row_from_csv(
            C.ARM_DIFFERENCES_CSV,
            lambda d: (d["class"] == label) & (d["arm_a"] == "A") & (d["arm_b"] == "D")
                      & (d["metric"] == "f1"),
        )
        if row is None:
            raise RuntimeError(f"missing arm_differences row for primary endpoint {label}")
        perm = None
        if paired:
            for pr in paired["per_label_results"]:
                if (pr["label"], pr["arm_a"], pr["arm_b"], pr["metric"]) == \
                        (label, "A", "D", "f1"):
                    perm = pr
                    break
        primary.append({
            "label": label,
            "arm_a": "A",
            "arm_b": "D",
            "metric": "f1",
            "delta_point_estimate": float(row["delta_point_estimate"]),
            "ci_lower": float(row["ci_lower"]),
            "ci_upper": float(row["ci_upper"]),
            "bootstrap_covers_zero": bool(row["ci_lower"] <= 0.0 <= row["ci_upper"]),
            "permutation_p": perm["p_value"] if perm else None,
            "holm_adjusted_p": perm.get("holm_adjusted_p") if perm else None,
            "significance": "significant" if (perm and perm.get("holm_adjusted_p")
                                              and perm["holm_adjusted_p"] < 0.05) else "n.s.",
            "single_primary_hypothesis": True,
        })

    payload = {
        "generated_utc": None,
        "population": {
            "split": "test",
            "n_images": int(len(df)),
            "n_patients": int(df["Patient ID"].nunique()),
        },
        "method_summary": {
            "bootstrap": {
                "unit": "patient (cluster)",
                "n_iterations": int(C.PATIENT_BOOTSTRAP_ITERATIONS),
                "ci": C.PATIENT_BOOTSTRAP_CI_LEVEL,
                "seed": C.PATIENT_BOOTSTRAP_SEED,
                "paired_across_arms": True,
                "artifacts": [str(C.CONFIDENCE_INTERVALS_PATIENT_CSV),
                              str(C.ARM_DIFFERENCES_CSV)],
            },
            "deLong": delong["per_label"] if delong else None,
            "threshold_stability": stability["equivalence_check"]["passed"] if stability else None,
        },
        "primary_endpoints": primary,
        "exploratory": {
            "arm_differences": diffs.to_dict(orient="records"),
            "permutation": paired,
            "delong": delong,
            "threshold_stability": stability,
            "ece_sensitivity": ece,
            "extension_arms": extension,
            "prevalence_shift": prevalence,
            "decision_policies": decision,
        },
        "sources": [str(p) for p in (
            C.CONFIDENCE_INTERVALS_PATIENT_CSV, C.ARM_DIFFERENCES_CSV,
            C.PATIENT_STATS_DIR / "paired_tests.json", C.PATIENT_STATS_DIR / "delong_auroc.json",
            C.THRESHOLD_STABILITY_JSON, C.ECE_SENSITIVITY_JSON, C.EXT_EXTENSION_ARMS_JSON,
            C.DECISION_POLICY_ANALYSIS_CSV, C.PREVALENCE_SHIFT_JSON) if p.exists()],
    }
    write_json(C.STATISTICAL_REPORT_JSON, payload)
    logger.info("statistical_report.json -> %d primary endpoints", len(primary))
    return payload


if __name__ == "__main__":
    run_statistical_report()