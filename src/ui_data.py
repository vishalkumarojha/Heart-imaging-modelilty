"""Read-only loader for the frozen research dashboard evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as C

LABELS = ("Cardiomegaly", "Effusion")
ROOT = C.PROJECT_ROOT


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        try: display = path.relative_to(ROOT)
        except ValueError: display = path
        raise FileNotFoundError(f"Research artifact unavailable: {display}")
    return json.loads(path.read_text())


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        try: display = path.relative_to(ROOT)
        except ValueError: display = path
        raise FileNotFoundError(f"Research artifact unavailable: {display}")
    return pd.read_csv(path)


def load_research_data() -> dict[str, Any]:
    """Load authoritative result files. Missing files are reported, never recreated."""
    base = C.OUTPUTS_DIR
    return {
        "manifest": _json(base / "final_research_manifest.json"),
        "baseline_test": _json(base / "metrics/baseline/baseline_metrics_test.json"),
        "calibration": _json(base / "metrics/calibration/calibration_report_val.json"),
        "temperatures": _json(base / "metrics/calibration/temperature_scalers.json"),
        "logistic": _json(base / "metrics/calibration/logistic_calibration_report.json"),
        "thresholds_raw": _json(base / "metrics/thresholds/thresholds_val.json"),
        "thresholds_cal": _json(base / "metrics/thresholds/thresholds_calibrated_val.json"),
        "thresholds_logistic": _json(base / "metrics/thresholds/thresholds_logistic_val.json"),
        "ece_sensitivity": _json(base / "metrics/ece_sensitivity/ece_sensitivity.json"),
        "stability": _json(base / "metrics/threshold_stability/threshold_stability.json"),
        "external": _json(base / "metrics/external/status.json"),
        "paired_tests": _json(base / "metrics/patient_stats/paired_tests.json"),
        "delong": _json(base / "metrics/patient_stats/delong_auroc.json"),
        "prevalence": _json(base / "metrics/prevalence/prevalence_shift.json"),
        "arm_differences": _csv(base / "final_results/arm_differences.csv"),
        "policy": _csv(base / "final_results/decision_policy_analysis.csv"),
        "error": _csv(base / "final_results/error_analysis.csv"),
        "bootstrap": _csv(base / "final_results/confidence_intervals_patient.csv"),
        "extension": _csv(base / "final_results/extension_arms.csv"),
        "figures": {
            "calibration": base / "paper/figures/figure_10_calibration_comparison.png",
            "ece": base / "paper/figures/figure_11_ece_sensitivity.png",
            "stability": base / "paper/figures/figure_12_threshold_stability.png",
            "policy": base / "paper/figures/figure_13_decision_policy.png",
            "prevalence": base / "paper/figures/figure_14_prevalence_sensitivity.png",
            "error": base / "paper/figures/figure_08_error_analysis.png",
            "gradcam": base / "paper/figures/figure_09_gradcam.png",
        },
    }


def temperature_probabilities(logits: list[float], temperatures: list[float]) -> list[float]:
    """Apply frozen per-label temperature scaling: sigmoid(z / T)."""
    import math
    if len(logits) != len(LABELS) or len(temperatures) != len(LABELS):
        raise ValueError("Expected logits and temperatures in Cardiomegaly, Effusion order")
    return [1.0 / (1.0 + math.exp(-float(z) / float(t))) for z, t in zip(logits, temperatures)]
