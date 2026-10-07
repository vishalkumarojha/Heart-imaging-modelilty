"""The four Phase-5 experiments (plus the plots they produce).

Every experiment
    * consumes the *frozen* cached predictions (`src.inference`), the *frozen*
      temperature parameters and the *frozen* validation thresholds;
    * writes one JSON payload under `outputs/metrics/experiments/` plus derived
      CSV/plots;
    * records `fit_split` and `eval_split` so no reader can confuse a fitted
      parameter with a measured result.

Experiments
    1  exp1_calibration            raw vs temperature-scaled calibration on
                                   validation AND test (ECE/MCE/Brier/NLL,
                                   reliability diagrams, AUROC-invariance check)
    2  exp2_thresholds_raw         five policies on RAW probabilities,
                                   fitted on val, evaluated on test
    3  exp2...calibrated           the same policies on CALIBRATED probabilities,
                                   with a head-to-head against experiment 2
    4  exp4_robustness             how sensitive are the conclusions?
                                   (a) ECE under 5/10/15/20 bins × equal-width/
                                   equal-frequency, (b) constraint-target sweeps,
                                   (c) bootstrap stability of the val thresholds

    python -m src.experiments --exp all
    python -m src.experiments --exp 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .calibration import TemperatureScaler, calibration_report, sigmoid
from .inference import as_arrays, ensure_predictions, label_names_from
from .metrics import auroc, auprc, binary_metrics
from .plots import (
    calibration_bars,
    curves_with_thresholds,
    ece_before_after,
    reliability_plot,
    threshold_sweep,
)
from .reproducibility import capture_environment, sha256_file, write_json
from .thresholds import POLICIES, find_threshold, fit_thresholds
from .utils import format_metrics_table, setup_logging

logger = setup_logging()

CHECKPOINT = C.BASELINE_CHECKPOINT_BEST
EXP_DIR = C.EXPERIMENT_METRICS_DIR
CAL_DIR = C.CALIBRATION_METRICS_DIR
THR_DIR = C.THRESHOLD_METRICS_DIR
SUMMARY_CSV = EXP_DIR / "experiments_summary.csv"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _preds(split: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    df = ensure_predictions(CHECKPOINT, split)
    arr = as_arrays(df)
    return df, arr, arr["labels"]


def _scaler(labels: Sequence[str]) -> TemperatureScaler:
    if not C.TEMPERATURE_FILE.exists():
        raise SystemExit(
            f"Missing {C.TEMPERATURE_FILE} — run `python -m src.fit_parameters` first."
        )
    return TemperatureScaler.load(C.TEMPERATURE_FILE)


def _thresholds(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise SystemExit(f"Missing {path} — run `python -m src.fit_parameters` first.")
    state = json.loads(path.read_text())
    return {
        policy: {lbl: float(rec["threshold"]) for lbl, rec in per.items()}
        for policy, per in state["policies"].items()
    }


def _base_payload(exp: str, title: str, **extra) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "experiment": exp,
        "title": title,
        "status": "COMPLETED",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "parameter_fit_split": C.THRESHOLD_FIT_SPLIT,
        "environment": capture_environment(include_packages=False),
    }
    payload.update(extra)
    return payload


def _append_summary(rows: List[Dict[str, object]]) -> Path:
    df = pd.DataFrame(rows)
    if SUMMARY_CSV.exists():
        old = pd.read_csv(SUMMARY_CSV)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in
                                        ("experiment", "split", "label", "policy",
                                         "variant", "metric") if c in df.columns],
                                keep="last")
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    return SUMMARY_CSV


# --------------------------------------------------------------------------- #
# Experiment 1 — calibration quality
# --------------------------------------------------------------------------- #
def exp1_calibration(splits: Sequence[str] = ("val", "test")) -> Dict[str, object]:
    labels = list(C.TARGET_LABELS)
    scaler = _scaler(labels)
    results: Dict[str, object] = {}
    summary: List[Dict[str, object]] = []
    invariant_ok = True

    for split in splits:
        df, arr, label_names = _preds(split)
        y, logits, probs = arr["y_true"], arr["logits"], arr["probs"]
        cal_probs = scaler.probabilities(logits)

        per_split: Dict[str, object] = {}
        for i, lbl in enumerate(label_names):
            raw_rep = calibration_report(y[:, i], probs[:, i])
            cal_rep = calibration_report(y[:, i], cal_probs[:, i])
            per_split[lbl] = {"raw": raw_rep, "calibrated": cal_rep}

            # temperature scaling is monotone ⇒ ranking must be untouched
            a_raw, a_cal = auroc(y[:, i], probs[:, i]), auroc(y[:, i], cal_probs[:, i])
            p_raw, p_cal = auprc(y[:, i], probs[:, i]), auprc(y[:, i], cal_probs[:, i])
            if abs(a_raw - a_cal) > 1e-9 or abs(p_raw - p_cal) > 1e-7:
                invariant_ok = False
            per_split[lbl]["auroc_raw"] = a_raw
            per_split[lbl]["auroc_calibrated"] = a_cal
            per_split[lbl]["auprc_raw"] = p_raw
            per_split[lbl]["auprc_calibrated"] = p_cal

            reliability_plot(
                y[:, i], {"raw": probs[:, i], "temperature-scaled": cal_probs[:, i]},
                f"{split} — {lbl}",
                C.PLOTS_DIR / "calibration" / f"reliability_{split}_{lbl.lower()}.png",
            )
            for variant, p in (("raw", probs[:, i]), ("calibrated", cal_probs[:, i])):
                summary.append({
                    "experiment": "exp1_calibration", "split": split, "label": lbl,
                    "policy": "", "variant": variant, "metric": "ece",
                    "value": raw_rep["ece"] if variant == "raw" else cal_rep["ece"],
                })
                summary.append({
                    "experiment": "exp1_calibration", "split": split, "label": lbl,
                    "policy": "", "variant": variant, "metric": "brier",
                    "value": raw_rep["brier"] if variant == "raw" else cal_rep["brier"],
                })

        bars = {
            metric: {lbl: {"raw": per_split[lbl]["raw"][metric],
                           "calibrated": per_split[lbl]["calibrated"][metric]}
                     for lbl in label_names}
            for metric in ("ece", "brier")
        }
        calibration_bars(bars, C.PLOTS_DIR / "calibration" / f"ece_brier_{split}.png",
                         title=f"Calibration quality — {split}")
        ece_before_after(
            {lbl: per_split[lbl]["raw"]["ece"] for lbl in label_names},
            {lbl: per_split[lbl]["calibrated"]["ece"] for lbl in label_names},
            C.PLOTS_DIR / "calibration" / f"ece_before_after_{split}.png", split,
        )

        # persist calibrated predictions (Phase-4 storage contract)
        out_df = df.copy()
        for i, lbl in enumerate(label_names):
            out_df[f"prob_cal_{lbl}"] = cal_probs[:, i]
        pred_path = C.PREDICTIONS_DIR / "calibrated" / f"calibrated_{split}_predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(pred_path, index=False)

        per_split["predictions_file"] = str(pred_path)
        results[split] = per_split
        logger.info("exp1: %s done (%d rows)", split, len(df))

    payload = _base_payload(
        "exp1_calibration",
        "Calibration quality of the frozen model, raw vs temperature-scaled",
        temperatures={l: float(t) for l, t in zip(labels, scaler.temperatures_)},
        calibration_fitted_on=scaler.metadata_.get("split"),
        n_bins=C.CALIBRATION_BINS,
        binning=C.CALIBRATION_STRATEGY,
        splits=results,
        auroc_auprc_invariant_to_temperature=invariant_ok,
        interpretation=(
            "ECE is measured with equal-width bins and may *increase* after "
            "temperature scaling: T is fitted by maximum likelihood (NLL), not by "
            "minimising ECE — the two objectives disagree when the score "
            "distribution is heavily concentrated in the lowest bin. Brier and "
            "NLL are the proper scoring rules used to judge the fit."
        ),
    )
    path = write_json(EXP_DIR / "exp1_calibration.json", payload)
    _append_summary(summary)
    logger.info("exp1 -> %s (AUROC/AUPRC invariant: %s)", path, invariant_ok)
    return payload


# --------------------------------------------------------------------------- #
# Experiments 2 & 3 — threshold policies
# --------------------------------------------------------------------------- #
def _evaluate_policies(
    split: str,
    variant: str,
    probs: np.ndarray,
    label_names: Sequence[str],
    y: np.ndarray,
    thresholds: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Apply frozen thresholds to one split; return {label: {policy: metrics}} + rows."""
    per_label: Dict[str, object] = {}
    summary: List[Dict[str, object]] = []
    for i, lbl in enumerate(label_names):
        fixed = binary_metrics(y[:, i], probs[:, i], 0.5)
        entry: Dict[str, object] = {}
        for policy in POLICIES:
            tau = thresholds[policy][lbl]
            m = binary_metrics(y[:, i], probs[:, i], tau)
            m["threshold"] = tau
            m["delta_f1_vs_fixed"] = m["f1"] - fixed["f1"]
            m["delta_precision_vs_fixed"] = m["precision"] - fixed["precision"]
            m["delta_recall_vs_fixed"] = m["recall"] - fixed["recall"]
            entry[policy] = m
            for metric in ("precision", "recall", "specificity", "f1", "accuracy",
                           "sensitivity"):
                summary.append({
                    "experiment": f"thresholds_{variant}", "split": split, "label": lbl,
                    "policy": policy, "variant": variant, "metric": metric,
                    "value": m[metric],
                })
        entry["_fixed_0.50"] = fixed
        per_label[lbl] = entry
    return per_label, summary


def exp2_thresholds(variant: str = "raw") -> Dict[str, object]:
    """Policies on RAW (experiment 2) or CALIBRATED (experiment 3) probabilities."""
    if variant not in ("raw", "calibrated"):
        raise ValueError("variant must be 'raw' or 'calibrated'")
    labels = list(C.TARGET_LABELS)
    scaler = _scaler(labels)
    thr_file = C.THRESHOLDS_FILE if variant == "raw" else \
        THR_DIR / "thresholds_calibrated_val.json"
    thresholds = _thresholds(thr_file)
    exp_name = "exp2_thresholds_raw" if variant == "raw" else "exp3_thresholds_calibrated"

    results: Dict[str, object] = {}
    summary: List[Dict[str, object]] = []
    fit_metrics: Dict[str, object] = {}
    thresholded_file: Optional[str] = None

    state = json.loads(thr_file.read_text())
    for policy, per in state["policies"].items():
        fit_metrics[policy] = {
            lbl: {"threshold": rec["threshold"],
                  "fit_split": rec["split"],
                  "fit_sensitivity": rec["fit_metrics"]["sensitivity"],
                  "fit_precision": rec["fit_metrics"]["precision"],
                  "fit_f1": rec["fit_metrics"]["f1"],
                  "constraint_met": rec.get("constraint_met")}
            for lbl, rec in per.items()
        }

    for split in ("val", "test"):
        df, arr, label_names = _preds(split)
        y = arr["y_true"]
        probs = arr["probs"] if variant == "raw" else scaler.probabilities(arr["logits"])
        per_label, rows = _evaluate_policies(
            split, variant, probs, label_names, y, thresholds)
        results[split] = per_label
        summary.extend(rows)

        # plots: curves annotated with every policy's operating point
        for i, lbl in enumerate(label_names):
            curves_with_thresholds(
                y[:, i], probs[:, i],
                {p: thresholds[p][lbl] for p in POLICIES},
                f"{split} / {variant} — {lbl}",
                C.PLOTS_DIR / "threshold" / f"{variant}_roc_{split}_{lbl.lower()}.png",
                C.PLOTS_DIR / "threshold" / f"{variant}_pr_{split}_{lbl.lower()}.png",
            )
            threshold_sweep(
                y[:, i], probs[:, i], lbl,
                C.PLOTS_DIR / "threshold" / f"{variant}_sweep_{split}_{lbl.lower()}.png",
                markers={p: thresholds[p][lbl] for p in POLICIES},
            )

        # store thresholded predictions for the test split
        if split == "test":
            out = df.copy()
            for i, lbl in enumerate(label_names):
                for policy in POLICIES:
                    out[f"pred_{policy}_{lbl}"] = (
                        probs[:, i] >= thresholds[policy][lbl]).astype(int)
            tpath = (C.PREDICTIONS_DIR / "thresholded" /
                     f"{variant}__thresholded_test_predictions.csv")
            tpath.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(tpath, index=False)
            thresholded_file = str(tpath)

    payload = _base_payload(
        exp_name,
        f"Disease-specific thresholds ({variant} probabilities), fitted on val, "
        "evaluated on test",
        variant=variant,
        threshold_parameters_file=str(thr_file),
        thresholded_predictions_file=thresholded_file,
        thresholds_fit_on=fit_metrics,
        evaluation=results,
        comparison_note=(
            "Every policy row already carries delta_*_vs_fixed, i.e. the change "
            "relative to the baseline operating point at τ=0.50."
        ),
    )
    path = write_json(EXP_DIR / f"{exp_name}.json", payload)
    _append_summary(summary)
    logger.info("%s -> %s", exp_name, path)

    # console table for the test split
    rows = []
    for lbl in labels:
        entry = results["test"][lbl]
        for policy in POLICIES:
            m = entry[policy]
            rows.append({"label": lbl, "policy": policy, "tau": round(m["threshold"], 3),
                         "prec": round(m["precision"], 4), "rec": round(m["recall"], 4),
                         "spec": round(m["specificity"], 4), "f1": round(m["f1"], 4),
                         "d_f1": round(m["delta_f1_vs_fixed"], 4)})
    logger.info("\n%s", format_metrics_table(
        rows, ["label", "policy", "tau", "prec", "rec", "spec", "f1", "d_f1"]))
    return payload


# --------------------------------------------------------------------------- #
# Experiment 4 — robustness / ablation
# --------------------------------------------------------------------------- #
def exp4_robustness(n_bootstrap: int = 10, seed: int = C.SEED) -> Dict[str, object]:
    labels = list(C.TARGET_LABELS)
    scaler = _scaler(labels)

    # --- (a) ECE sensitivity to binning -------------------------------------
    binning: Dict[str, object] = {}
    summary: List[Dict[str, object]] = []
    for split in ("val", "test"):
        _, arr, label_names = _preds(split)
        y, probs = arr["y_true"], arr["probs"]
        cal = scaler.probabilities(arr["logits"])
        for n_bins in (5, 10, 15, 20):
            for strategy in ("equal_width", "equal_freq"):
                key = f"{split}/bins={n_bins}/{strategy}"
                binning[key] = {
                    "raw": {lbl: calibration_report(y[:, i], probs[:, i],
                                                    n_bins, strategy)["ece"]
                            for i, lbl in enumerate(label_names)},
                    "calibrated": {lbl: calibration_report(y[:, i], cal[:, i],
                                                           n_bins, strategy)["ece"]
                                   for i, lbl in enumerate(label_names)},
                }
                for variant in ("raw", "calibrated"):
                    for lbl in label_names:
                        summary.append({
                            "experiment": "exp4_robustness", "split": split,
                            "label": lbl, "policy": f"bins={n_bins}/{strategy}",
                            "variant": variant, "metric": "ece",
                            "value": binning[key][variant][lbl],
                        })

    # --- (b) constraint-target sweeps ---------------------------------------
    _, val_arr, label_names = _preds("val")
    _, test_arr, _ = _preds("test")
    sweeps: Dict[str, object] = {}
    for policy, targets in (
        ("sensitivity_constrained", (0.80, 0.90, 0.95)),
        ("precision_constrained", (0.40, 0.50, 0.60)),
    ):
        rows = []
        for target in targets:
            fitted = fit_thresholds(
                val_arr["y_true"], val_arr["probs"], label_names,
                policies=[policy], split="val",
                sensitivity_target=target, precision_target=target,
            )
            for lbl in label_names:
                rec = fitted[policy][lbl]
                j = label_names.index(lbl)
                m_test = binary_metrics(test_arr["y_true"][:, j],
                                        test_arr["probs"][:, j],
                                        rec["threshold"])
                rows.append({
                    "target": target, "label": lbl,
                    "threshold": rec["threshold"],
                    "constraint_met": rec.get("constraint_met"),
                    "test_precision": m_test["precision"],
                    "test_recall": m_test["recall"],
                    "test_f1": m_test["f1"],
                    "test_specificity": m_test["specificity"],
                })
                summary.append({
                    "experiment": "exp4_robustness", "split": "test", "label": lbl,
                    "policy": f"{policy}@target={target}", "variant": "raw",
                    "metric": "f1", "value": m_test["f1"],
                })
        sweeps[policy] = rows

    # --- (c) bootstrap stability of the validation thresholds ----------------
    frozen_thresholds = _thresholds(C.THRESHOLDS_FILE)
    rng = np.random.default_rng(seed)
    n_val = len(val_arr["y_true"])
    boot: Dict[str, object] = {}
    for policy in POLICIES:
        collected: Dict[str, List[float]] = {lbl: [] for lbl in label_names}
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_val, n_val)
            for i, lbl in enumerate(label_names):
                rec = find_threshold(val_arr["y_true"][idx, i],
                                     val_arr["probs"][idx, i], policy)
                collected[lbl].append(float(rec["threshold"]))
        boot[policy] = {
            lbl: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "frozen_threshold": frozen_thresholds[policy][lbl],
            }
            for lbl, v in collected.items()
        }

    payload = _base_payload(
        "exp4_robustness",
        "Sensitivity of the conclusions to binning, constraint targets and "
        "validation resampling",
        ece_binning_sensitivity=binning,
        constraint_target_sweeps=sweeps,
        bootstrap_threshold_stability={
            "n_bootstrap": n_bootstrap, "seed": seed, "results": boot},
        notes=[
            "ECE is a binned statistic: its value moves with the bin count and "
            "with the binning strategy — report it together with Brier/NLL.",
            "Bootstrap thresholds are resampled from validation only; test data "
            "is used exclusively to evaluate the resulting operating points.",
        ],
    )
    path = write_json(EXP_DIR / "exp4_robustness.json", payload)
    _append_summary(summary)
    logger.info("exp4 -> %s", path)
    return payload


# --------------------------------------------------------------------------- #
def run_all() -> Dict[str, object]:
    out = {
        "exp1": exp1_calibration(),
        "exp2": exp2_thresholds("raw"),
        "exp3": exp2_thresholds("calibrated"),
        "exp4": exp4_robustness(),
    }
    logger.info("all experiments complete: %s", ", ".join(out))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase-5 experiments")
    p.add_argument("--exp", default="all",
                   choices=["1", "2", "3", "4", "all"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.exp == "all":
        run_all()
    elif args.exp == "1":
        exp1_calibration()
    elif args.exp == "2":
        exp2_thresholds("raw")
    elif args.exp == "3":
        exp2_thresholds("calibrated")
    elif args.exp == "4":
        exp4_robustness()
