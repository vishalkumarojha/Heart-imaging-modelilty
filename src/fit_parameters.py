"""Fit the *research* parameters on validation predictions and freeze them.

Everything this script writes is a **parameter**, not a result: it is derived
from the validation split only, then treated as fixed and applied unchanged to
test / external data. Re-running it on another split is refused
(`src.calibration` / `src.thresholds` raise), which is the machine-checkable
form of the rule in `docs/PROJECT_SCOPE.md` §4.

Artifacts (Phase 4 — IMPLEMENTED):

    outputs/metrics/calibration/temperature_scalers.json      T per label
    outputs/metrics/thresholds/thresholds_val.json            policies on RAW probs
    outputs/metrics/thresholds/thresholds_calibrated_val.json policies on CALIBRATED probs

Two threshold sets exist because temperature scaling changes the probability
scale: a threshold tuned on raw sigmoid outputs is not the right threshold for
calibrated outputs. Both are fitted on the same validation rows and both are
frozen for evaluation.

    python -m src.fit_parameters                 # fit on the validation split
    python -m src.fit_parameters --check         # print what is currently frozen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C
from .calibration import TemperatureScaler, calibration_report, sigmoid
from .inference import as_arrays, ensure_predictions, label_names_from
from .reproducibility import sha256_file, write_json
from .thresholds import fit_thresholds, thresholds_to_json
from .utils import setup_logging

logger = setup_logging()


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_all(
    checkpoint: Path = C.BASELINE_CHECKPOINT_BEST,
    split: str = C.CALIBRATION_FIT_SPLIT,
    refit: bool = False,
) -> Dict[str, object]:
    """Fit temperatures + both threshold sets on one (validation) split."""
    split_l = split.lower()
    if any(bad in split_l for bad in ("test", "external", "chexpert")):
        raise ValueError(
            f"refusing to fit parameters on split '{split}' — calibration and "
            "thresholds must come from validation predictions"
        )

    ckpt = Path(checkpoint)
    preds = ensure_predictions(ckpt, split)
    labels = label_names_from(preds)
    arrays = as_arrays(preds, labels)
    y, logits, probs = arrays["y_true"], arrays["logits"], arrays["probs"]

    # ---- 1. temperature scaling -------------------------------------------
    scaler = TemperatureScaler(labels).fit(logits, y, split=split)
    cal_probs = scaler.probabilities(logits)

    if C.TEMPERATURE_FILE.exists() and not refit:
        existing = json.loads(C.TEMPERATURE_FILE.read_text())
        logger.info(
            "temperature_scalers.json already exists (use --refit to replace); "
            "current T = %s", existing.get("temperatures"))
    else:
        scaler.save(C.TEMPERATURE_FILE)
        logger.info("Fitted temperatures on '%s': %s", split, repr(scaler))

    # ---- 2. thresholds on raw and on calibrated probabilities --------------
    raw_fit = fit_thresholds(y, probs, labels, split=split)
    cal_fit = fit_thresholds(y, cal_probs, labels, split=split)
    _write_if_allowed(C.THRESHOLDS_FILE, thresholds_to_json(raw_fit), refit, "raw")
    cal_path = C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json"
    _write_if_allowed(cal_path, thresholds_to_json(cal_fit), refit, "calibrated")

    # ---- 3. record the validation calibration report (pre/post) ------------
    report = {
        "split": split,
        "labels": labels,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "n_rows": int(len(preds)),
        "n_bins": C.CALIBRATION_BINS,
        "binning": C.CALIBRATION_STRATEGY,
        "raw": {l: calibration_report(y[:, i], probs[:, i]) for i, l in enumerate(labels)},
        "calibrated": {l: calibration_report(y[:, i], cal_probs[:, i])
                       for i, l in enumerate(labels)},
        "temperatures": {l: float(t) for l, t in zip(labels, scaler.temperatures_)},
        "thresholds_raw": _threshold_table(raw_fit, labels),
        "thresholds_calibrated": _threshold_table(cal_fit, labels),
        "status": "PARAMETERS FITTED ON VALIDATION ONLY",
    }
    out = C.CALIBRATION_METRICS_DIR / f"calibration_report_{split}.json"
    write_json(out, report)
    logger.info("Validation calibration report -> %s", out)

    _print_summary(report)
    return report


def _write_if_allowed(path: Path, payload, refit: bool, kind: str) -> None:
    if path.exists() and not refit:
        logger.info("%s already exists (%s) — kept; use --refit to replace",
                    path.name, kind)
        return
    write_json(path, payload)
    logger.info("Fitted %s thresholds -> %s", kind, path)


def _threshold_table(fitted: Dict[str, Dict[str, Dict[str, object]]],
                     labels) -> Dict[str, Dict[str, float]]:
    return {
        policy: {lbl: float(rec["threshold"]) for lbl, rec in per_label.items()}
        for policy, per_label in fitted.items()
    }


def _print_summary(report: Dict[str, object]) -> None:
    logger.info("\n--- validation parameters (%s, n=%s) ---", report["split"], report["n_rows"])
    for lbl in report["labels"]:  # type: ignore[index]
        raw, cal = report["raw"][lbl], report["calibrated"][lbl]
        logger.info(
            "%s: T=%.3f  ECE %.4f -> %.4f | Brier %.4f -> %.4f",
            lbl, report["temperatures"][lbl],
            raw["ece"], cal["ece"], raw["brier"], cal["brier"],
        )
    for policy, per in report["thresholds_raw"].items():  # type: ignore[index]
        logger.info("%-24s %s", policy,
                    "  ".join(f"{l}={v:.3f}" for l, v in per.items()))


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def check() -> None:
    files = {
        "temperatures": C.TEMPERATURE_FILE,
        "thresholds (raw)": C.THRESHOLDS_FILE,
        "thresholds (calibrated)": C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json",
        "validation calibration report": C.CALIBRATION_METRICS_DIR / "calibration_report_val.json",
    }
    for name, path in files.items():
        if not path.exists():
            print(f"MISSING  {name}: {path}")
            continue
        state = json.loads(path.read_text())
        if "temperatures" in state:
            detail = state["temperatures"]
        elif "policies" in state:
            detail = {
                p: {l: round(r["threshold"], 4) for l, r in v.items()}
                for p, v in list(state["policies"].items())[:5]
            }
        else:
            detail = {k: state.get(k) for k in ("split", "n_rows", "temperatures")}
        print(f"OK       {name}: {path.name}")
        print(f"         {detail}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit calibration + thresholds on validation")
    p.add_argument("--checkpoint", type=Path, default=C.BASELINE_CHECKPOINT_BEST)
    p.add_argument("--split", default=C.CALIBRATION_FIT_SPLIT,
                   choices=["train", "val"])   # test/external are not offered
    p.add_argument("--refit", action="store_true",
                   help="overwrite existing fitted-parameter files")
    p.add_argument("--check", action="store_true", help="show frozen parameters")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check:
        check()
    else:
        fit_all(args.checkpoint, args.split, args.refit)
