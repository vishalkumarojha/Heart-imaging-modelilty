"""Frozen prediction / parameter loaders shared by the statistics modules.

Everything here reads the *frozen* artifacts (never the live training loop):

    outputs/predictions/raw/test__<hash>.csv           logits + raw probabilities
    outputs/metrics/calibration/temperature_scalers.json
    outputs/metrics/calibration/logistic_scalers.json   (IEEE-upgrade)
    outputs/metrics/thresholds/thresholds_{val,calibrated_val}.json
    outputs/metrics/thresholds/thresholds_logistic_val.json   (IEEE-upgrade)

Probabilities are re-derived from logits through the frozen parameters instead
of trusting a cached column, so a loading bug cannot silently decouple the test
pipeline from the calibrated/threshold artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .. import config as C
from ..calibration import sigmoid

VARIANTS = ("raw", "calibrated", "logistic")
POLICIES = C.THRESHOLD_POLICIES

# arm id -> (probability variant, threshold policy)
ARMS = {
    "A": ("raw", "fixed"),
    "B": ("calibrated", "fixed"),
    "C": ("raw", "f1_optimal"),
    "D": ("calibrated", "f1_optimal"),
    "E": ("logistic", "fixed"),
    "F": ("logistic", "f1_optimal"),
}

PAIRED_COMPARISONS = (("A", "C"), ("A", "D"), ("B", "D"), ("C", "D"))


def raw_predictions_file(split: str) -> Path:
    matches = sorted(Path(C.RAW_PREDICTIONS_DIR).glob(f"{split}__*.csv"))
    if not matches:
        raise FileNotFoundError(f"no frozen {split} predictions under {C.RAW_PREDICTIONS_DIR}")
    return matches[0]


def load_predictions(split: str) -> pd.DataFrame:
    df = pd.read_csv(raw_predictions_file(split))
    assert (df["split"] == split).all(), f"expected {split}-only predictions"
    return df


def load_test_predictions() -> pd.DataFrame:
    return load_predictions("test")


def load_val_predictions() -> pd.DataFrame:
    return load_predictions("val")


def load_temperatures() -> Dict[str, float]:
    return {l: float(t) for l, t in
            json.loads(C.TEMPERATURE_FILE.read_text())["metadata"]["temperatures"].items()}


def load_logistic_params() -> Dict[str, Dict[str, float]]:
    """Return {label: {"a": slope, "b": intercept, "method": ...}}."""
    state = json.loads(C.LOGISTIC_SCALERS_FILE.read_text())
    params: Dict[str, Dict[str, float]] = {}
    for label, rec in state["parameters"].items():
        params[label] = {"a": float(rec["a"]), "b": float(rec["b"]),
                         "method": str(rec["method"])}
    return params


def calibrated_logits(logits: np.ndarray, label: str) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64) / load_temperatures()[label]


def logistic_logits(logits: np.ndarray, label: str) -> np.ndarray:
    p = load_logistic_params()[label]
    out = p["a"] * np.asarray(logits, dtype=np.float64) + p["b"]
    if p["method"] == "platt_logistic":
        return out
    raise ValueError(f"unknown logistic method {p['method']!r}")


def variant_logits(logits: np.ndarray, label: str, variant: str) -> np.ndarray:
    if variant == "raw":
        return np.asarray(logits, dtype=np.float64)
    if variant == "calibrated":
        return calibrated_logits(logits, label)
    if variant == "logistic":
        return logistic_logits(logits, label)
    raise ValueError(f"unknown variant {variant!r}; known: {', '.join(VARIANTS)}")


def variant_probs(df: pd.DataFrame, label: str, variant: str) -> np.ndarray:
    """Frozen probabilities for `label` on `df` under `variant` (re-derived)."""
    logits = df[f"logit_{label}"].to_numpy(dtype=np.float64)
    if variant == "raw":
        return df[f"prob_{label}"].to_numpy(dtype=np.float64)
    return sigmoid(variant_logits(logits, label, variant))


def frozen_threshold_records(variant: str) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Raw frozen threshold records: {policy: {label: record}}."""
    file = {
        "raw": C.THRESHOLDS_FILE,
        "calibrated": C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json",
        "logistic": C.LOGISTIC_THRESHOLDS_FILE,
    }[variant]
    if not file.exists():
        raise FileNotFoundError(
            f"no frozen {variant} thresholds at {file} — run the fitting first: "
            f"thresholds are fitted on validation only"
        )
    return json.loads(file.read_text())["policies"]


def frozen_threshold_values(variant: str) -> Dict[str, Dict[str, float]]:
    return {policy: {lbl: float(rec["threshold"]) for lbl, rec in per_label.items()}
            for policy, per_label in frozen_threshold_records(variant).items()}


def arm_thresholds() -> Dict[str, Dict[str, float]]:
    """{arm: {label: tau}} built from the frozen val threshold files."""
    out: Dict[str, Dict[str, float]] = {}
    for arm, (variant, policy) in ARMS.items():
        out[arm] = {lbl: frozen_threshold_values(variant)[policy][lbl]
                    for lbl in C.TARGET_LABELS}
    return out


def per_variant_threshold_map(variant: str) -> Dict[str, float]:
    """{policy: {label: float}} flattened to per-label maps per policy."""
    return {policy: {lbl: float(rec["threshold"])
                     for lbl, rec in per_label.items()}
            for policy, per_label in frozen_threshold_records(variant).items()}


def all_labeled(labels: Sequence[str] = C.TARGET_LABELS) -> None:
    """Fail loudly if any frozen infrastructure for the goal labels is missing."""
    for variant in VARIANTS:
        frozen_threshold_values(variant)
    for label in labels:
        load_temperatures()
        load_logistic_params()