"""Phase 7a — error analysis on the frozen test split.

Everything here is computed from the *cached* prediction tables; the model is
never re-run, so these numbers are reproducible and consistent with
`outputs/metrics/experiments/`.

What is analysed (per label, per frozen policy)
    * confusion strata TP/FP/TN/FN + mean confidence inside each stratum;
    * high-confidence errors (confidence >= 0.90) — the clinically dangerous
      bucket: the model is sure and wrong;
    * error rate by confidence bin (where in the confidence range errors live);
    * cross-label error overlap (images wrong on *both* labels);
    * per-patient error concentration (repeat offenders = label-noise signal);
    * a deterministic Grad-CAM case list: fixed k examples per stratum,
      sorted by (probability desc, Image Index asc) so the selection is
      reproducible without a seed.

    python -m src.error_analysis --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as C
from .metrics import binary_metrics
from .plots import error_rate_by_confidence, strata_bars
from .reproducibility import capture_environment, write_json
from .thresholds import POLICIES
from .utils import setup_logging

logger = setup_logging()

HIGH_CONFIDENCE = 0.90
ERROR_BINS = 10
CASES_PER_STRATUM = 2
STRATA = ("TP", "FP", "FN", "TN")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_analysis_table(split: str = "test", variant: str = "calibrated") -> pd.DataFrame:
    """Return the prediction table for `split` with a `prob_<label>` column
    set to the requested variant (raw or temperature-calibrated)."""
    if variant not in ("raw", "calibrated"):
        raise ValueError(f"variant must be raw|calibrated, got {variant!r}")
    if variant == "calibrated":
        path = C.PREDICTIONS_DIR / "calibrated" / f"calibrated_{split}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python -m src.experiments --exp 1` first")
        df = pd.read_csv(path)
        for lbl in C.TARGET_LABELS:
            df[f"prob_{lbl}"] = df[f"prob_cal_{lbl}"]
    else:
        from .inference import load_predictions

        df = load_predictions(C.BASELINE_CHECKPOINT_BEST, split)
        if df is None:
            raise FileNotFoundError(
                "raw prediction cache missing — run `python -m src.inference "
                f"--split {split}`")
    return df


def frozen_thresholds(variant: str = "calibrated") -> Dict[str, Dict[str, float]]:
    """Thresholds fitted on validation and frozen for every evaluation."""
    if variant == "calibrated":
        path = C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json"
    else:
        path = C.THRESHOLDS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m src.fit_parameters` first")
    payload = json.loads(Path(path).read_text())
    return {policy: {lbl: float(entry["threshold"]) for lbl, entry in per_label.items()}
            for policy, per_label in payload["policies"].items()}


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def stratum_label(y: int, pred: int) -> str:
    if pred == y:
        return "TP" if y == 1 else "TN"
    return "FP" if pred == 1 else "FN"


def confusion_strata(df: pd.DataFrame, label: str, policy: str, threshold: float
                     ) -> Dict[str, object]:
    y = df[f"true_{label}"].to_numpy()
    p = df[f"prob_{label}"].to_numpy()
    pred = (p >= threshold).astype(int)
    metrics = binary_metrics(y, p, threshold)
    strata: Dict[str, Dict[str, float]] = {}
    for s in STRATA:
        mask = np.array([stratum_label(int(a), int(b)) == s for a, b in zip(y, pred)])
        strata[s] = {
            "count": int(mask.sum()),
            "share": float(mask.mean()),
            "mean_confidence": float(p[mask].mean()) if mask.any() else None,
            "confidence_range": ([float(p[mask].min()), float(p[mask].max())]
                                 if mask.any() else None),
        }
    errors = pred != y
    # confidence = certainty of the *predicted* class
    certainty = np.where(pred == 1, p, 1 - p)
    high_mask = errors & (certainty >= HIGH_CONFIDENCE)
    return {
        "label": label,
        "policy": policy,
        "threshold": float(threshold),
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        **{k: metrics[k] for k in ("precision", "recall", "specificity", "f1")},
        "error_rate": float(errors.mean()),
        "mean_certainty": float(certainty.mean()),
        "high_confidence_errors": {
            "confidence_floor": HIGH_CONFIDENCE,
            "count": int(high_mask.sum()),
            "share_of_errors": float(high_mask.sum() / max(errors.sum(), 1)),
            "share_of_cohort": float(high_mask.mean()),
        },
        "strata": strata,
    }


def confidence_error_bins(y: np.ndarray, p: np.ndarray,
                          n_bins: int = ERROR_BINS) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                         "count": 0, "error_rate": None, "mean_p": None})
            continue
        rows.append({
            "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "count": int(mask.sum()),
            "error_rate": float(np.mean((p[mask] >= 0.5).astype(int) != y[mask])),
            "mean_p": float(p[mask].mean()),
        })
    return rows


def cross_label_overlap(df: pd.DataFrame) -> Dict[str, object]:
    """Images the model gets wrong on both labels simultaneously."""
    wrong = {}
    for lbl in C.TARGET_LABELS:
        wrong[lbl] = (df[f"prob_{lbl}"] >= C.DECISION_THRESHOLD).to_numpy() != \
                     df[f"true_{lbl}"].to_numpy()
    both = wrong[C.TARGET_LABELS[0]] & wrong[C.TARGET_LABELS[1]]
    either = wrong[C.TARGET_LABELS[0]] | wrong[C.TARGET_LABELS[1]]
    return {
        "threshold": C.DECISION_THRESHOLD,
        "n_both_wrong": int(both.sum()),
        "n_either_wrong": int(either.sum()),
        "jaccard_of_error_sets": float(both.sum() / max(either.sum(), 1)),
    }


def patient_error_concentration(df: pd.DataFrame) -> Dict[str, object]:
    """Patients contributing many errors — a label-noise / confounder signal."""
    wrong = np.zeros(len(df), dtype=bool)
    for lbl in C.TARGET_LABELS:
        wrong |= (df[f"prob_{lbl}"] >= C.DECISION_THRESHOLD).to_numpy() != \
                 df[f"true_{lbl}"].to_numpy()
    err = df.loc[wrong, "Patient ID"].value_counts()
    return {
        "n_patients_with_errors": int(err.shape[0]),
        "n_error_images": int(wrong.sum()),
        "max_errors_by_one_patient": int(err.max()) if len(err) else 0,
        "top5_patients": [{"patient_id": int(i), "error_images": int(n)}
                          for i, n in err.head(5).items()],
    }


def select_gradcam_cases(df: pd.DataFrame,
                         thresholds: Dict[str, Dict[str, float]],
                         policy: str = "f1_optimal",
                         k: int = CASES_PER_STRATUM) -> List[Dict[str, object]]:
    """Deterministic case list: k examples of each stratum, per label.

    Ordering is (certainty desc, Image Index asc) — no RNG, so two runs on two
    machines select byte-identical cases.
    """
    cases: List[Dict[str, object]] = []
    for lbl in C.TARGET_LABELS:
        y = df[f"true_{lbl}"].to_numpy()
        p = df[f"prob_{lbl}"].to_numpy()
        tau = thresholds[policy][lbl]
        pred = (p >= tau).astype(int)
        certainty = np.where(pred == 1, p, 1 - p)
        order = pd.DataFrame({
            "stratum": [stratum_label(int(a), int(b)) for a, b in zip(y, pred)],
            "certainty": certainty,
            "probability": p,
            "Image Index": df["Image Index"].to_numpy(),
        })
        for s in STRATA:
            sel = order[order["stratum"] == s].sort_values(
                ["certainty", "Image Index"], ascending=[False, True]).head(k)
            for _, row in sel.iterrows():
                cases.append({
                    "case_id": f"{lbl}_{s}_{row['Image Index'].replace('.png', '')}",
                    "label": lbl,
                    "stratum": s,
                    "policy": policy,
                    "threshold": float(tau),
                    "certainty": float(row["certainty"]),
                    "probability": float(row["probability"]),
                    "predicted_positive": s in ("TP", "FP"),
                    "true_positive": s in ("TP", "FN"),
                    "image_index": row["Image Index"],
                })
    return cases


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(split: str = "test", variant: str = "calibrated") -> Dict[str, object]:
    df = load_analysis_table(split, variant)
    thresholds = frozen_thresholds(variant)
    labels = list(C.TARGET_LABELS)

    strata_rows: List[Dict[str, object]] = []
    strata_payload: Dict[str, Dict[str, Dict[str, object]]] = {}
    for policy in POLICIES:
        strata_payload[policy] = {}
        for lbl in labels:
            entry = confusion_strata(df, lbl, policy, thresholds[policy][lbl])
            strata_payload[policy][lbl] = entry
            strata_rows.append({
                "split": split, "variant": variant, "policy": policy, "label": lbl,
                "threshold": entry["threshold"], "n": entry["n"],
                "precision": entry["precision"], "recall": entry["recall"],
                "specificity": entry["specificity"], "f1": entry["f1"],
                "error_rate": entry["error_rate"],
                "high_conf_errors": entry["high_confidence_errors"]["count"],
                **{f"n_{s.lower()}": entry["strata"][s]["count"] for s in STRATA},
            })

    conf_bins: Dict[str, List[Dict[str, float]]] = {}
    for lbl in labels:
        conf_bins[lbl] = confidence_error_bins(
            df[f"true_{lbl}"].to_numpy(), df[f"prob_{lbl}"].to_numpy())

    cases = select_gradcam_cases(df, thresholds)

    payload: Dict[str, object] = {
        "phase": "error_analysis",
        "status": "COMPLETED",
        "split": split,
        "variant": variant,
        "n_images": int(len(df)),
        "labels": labels,
        "policies": list(POLICIES),
        "primary_policy_for_cases": "f1_optimal",
        "cases_per_stratum": CASES_PER_STRATUM,
        "strata": strata_payload,
        "confidence_error_bins": conf_bins,
        "cross_label_overlap": cross_label_overlap(df),
        "patient_error_concentration": patient_error_concentration(df),
        "gradcam_case_list": cases,
        "provenance": {
            "predictions": str(C.PREDICTIONS_DIR / "calibrated" /
                               f"calibrated_{split}_predictions.csv")
                           if variant == "calibrated" else "raw cache",
            "thresholds": "outputs/metrics/thresholds (fitted on val, frozen)",
            "environment": capture_environment(include_packages=False),
        },
    }

    # deterministic Grad-CAM case list: resolve image paths BEFORE the report
    # is written, so the published JSON and case_list.json agree byte for byte
    from .dataset import scan_image_paths

    path_map = scan_image_paths()
    for case in cases:
        case["path"] = path_map.get(case["image_index"])
    missing = [c["image_index"] for c in cases if not c["path"]]
    if missing:
        raise FileNotFoundError(f"case images not on disk: {missing}")

    C.ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = write_json(C.ERROR_ANALYSIS_DIR / f"error_analysis_{split}.json", payload)
    pd.DataFrame(strata_rows).to_csv(
        C.ERROR_ANALYSIS_DIR / f"error_strata_{split}.csv", index=False)

    write_json(C.GRADCAM_DIR / "case_list.json", {
        "split": split, "variant": variant, "policy": "f1_optimal",
        "n_cases": len(cases), "cases": cases,
    })

    # figures
    strata_bars(strata_rows, C.PLOTS_DIR / "error_analysis" /
                f"strata_{split}.png")
    for lbl in labels:
        error_rate_by_confidence(conf_bins[lbl],
                                 f"{lbl} — {split}, {variant}",
                                 C.PLOTS_DIR / "error_analysis" /
                                 f"confidence_error_{split}_{lbl.lower()}.png")

    logger.info("Error analysis -> %s (%d cases for Grad-CAM)", out, len(cases))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7a: error analysis")
    p.add_argument("--split", default="test")
    p.add_argument("--variant", default="calibrated", choices=("raw", "calibrated"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.split, args.variant)
