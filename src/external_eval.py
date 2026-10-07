"""External validation on an out-of-distribution cohort (CheXpert).

STATUS OF THIS PHASE: **BLOCKED — EXTERNAL DATASET NOT AVAILABLE**.
No CheXpert (or any other external) data exists under `data/raw/` in this
repository. The *pipeline* is implemented and tested; the moment the official
files are dropped into `data/raw/chexpert/`, the same command produces real
numbers. Until then `outputs/metrics/external/status.json` records the blocked
state and no external metric is ever printed, guessed or stored.

Design rules (same as every other experiment in this project)
    * the model, the temperature and the thresholds are the **frozen** ones from
      validation — nothing is refit on external data;
    * uncertainty handling is explicit and configurable
      (`EXTERNAL_UNCERTAINTY_POLICY`, default `u_zeroes`: CheXpert's −1 → 0);
    * NIH `Effusion` maps to CheXpert `Pleural Effusion`
      (`C.CHEXPERT_LABEL_MAP`);
    * evaluation transforms only — no augmentation, deterministic row order.

    python -m src.external_eval --status           # refresh the BLOCKED status
    python -m src.external_eval --split valid      # run (or report BLOCKED)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .calibration import TemperatureScaler, calibration_report
from .inference import load_model
from .metrics import auprc, auroc, binary_metrics, multilabel_report, report_to_rows
from .plots import curves_with_thresholds
from .reproducibility import capture_environment, sha256_file, write_json
from .thresholds import POLICIES, apply_thresholds
from .utils import format_metrics_table, set_seed, setup_logging

logger = setup_logging()

BLOCKED_STATUS = "BLOCKED — EXTERNAL DATASET NOT AVAILABLE"


class ExternalDataNotAvailable(RuntimeError):
    """Raised when an external run is requested but the cohort is absent."""


# --------------------------------------------------------------------------- #
# Data availability
# --------------------------------------------------------------------------- #
# Candidate roots are probed for the official layout in this order; the FIRST
# candidate that has BOTH the label CSV and the image directory wins.
def _candidate_roots(csv_path: Optional[Path],
                     images_root: Optional[Path]) -> List[str]:
    roots = [C.CHEXPERT_DIR]
    for var in ("CHEXPERT_ROOT", "CHEXPERT_DATA_DIR"):
        val = os.environ.get(var)
        if val:
            roots.append(Path(val))
    roots.append(Path("/mnt/data/chexpert"))
    roots.append(Path("/data/chexpert"))
    seen: List[str] = []
    for r in roots:
        if str(r) not in seen:
            seen.append(str(r))
    return seen


def check_external_data(
    csv_path: Optional[Path] = None, images_root: Optional[Path] = None
) -> Dict[str, object]:
    """Detect (never invent) whether the external cohort is present.

    Looks up the official `valid.csv` + `images/` layout under, in order:
      1. the configured `C.CHEXPERT_DIR`,
      2. `$CHEXPERT_ROOT` / `$CHEXPERT_DATA_DIR` (env-var override),
      3. common read-only mounts `/mnt/data/chexpert`, `/data/chexpert`.
    The first candidate with BOTH files is selected. If none match, the phase
    stays BLOCKED and the problems list explains every probe that failed.
    """
    candidates = _candidate_roots(csv_path, images_root)
    problems: List[str] = []
    for root in candidates:
        cand_csv = Path(csv_path) if csv_path else Path(root) / "valid.csv"
        cand_img = Path(images_root) if images_root else Path(root) / "images"
        if cand_csv.exists() and cand_img.exists():
            return {
                "available": True,
                "label_csv": str(cand_csv),
                "images_dir": str(cand_img),
                "resolved_from": str(root),
                "problems": [],
                "expected_layout": [
                    "data/raw/chexpert/valid.csv     (studies + labels)",
                    "data/raw/chexpert/images/…      (per-study image folders)",
                ],
                "uncertainty_policy": C.EXTERNAL_UNCERTAINTY_POLICY,
                "label_map": dict(C.CHEXPERT_LABEL_MAP),
            }
        missing = []
        if not cand_csv.exists():
            missing.append(f"valid.csv")
        if not cand_img.exists():
            missing.append(f"images/")
        problems.append(f"{root} (missing {', '.join(missing)})")
    return {
        "available": False,
        "label_csv": None,
        "images_dir": None,
        "resolved_from": None,
        "problems": problems,
        "expected_layout": [
            "data/raw/chexpert/valid.csv     (studies + labels)",
            "data/raw/chexpert/images/…      (per-study image folders)",
        ],
        "uncertainty_policy": C.EXTERNAL_UNCERTAINTY_POLICY,
        "label_map": dict(C.CHEXPERT_LABEL_MAP),
    }


PROJECT_STATUS = {
    "core_research": "COMPLETE",
    "external_validation": "PENDING",
}


def project_status_blocked() -> Dict[str, object]:
    return {
        "project": "chest_xray_research_migration",
        "core_research": "COMPLETE",
        "external_validation": "PENDING",
        "explanation": (
            "The core research pipeline is frozen and its results are verified "
            "(final_results/). External validity is a SEPARATE open question: it "
            "stays PENDING until an external cohort is available and the "
            "`src.external_eval` run completes."
        ),
    }


def write_blocked_status(reason: str) -> Dict[str, object]:
    payload = {
        "phase": "external_validation",
        "status": BLOCKED_STATUS,
        "project_status": project_status_blocked(),
        "reason": reason,
        "what_is_implemented": [
            "CheXpert label loader with an explicit uncertainty policy",
            "external dataset/dataloader with the frozen evaluation transform",
            "evaluation with the frozen model + frozen temperature + frozen "
            "validation thresholds (no refitting on external data)",
            "AUROC / AUPRC / ECE / Brier / per-policy operating points + plots",
        ],
        "what_is_missing": check_external_data()["problems"],
        "how_to_unblock": (
            "Place the official CheXpert files under data/raw/chexpert/ (or point "
            "$CHEXPERT_ROOT / $CHEXPERT_DATA_DIR at a mounted copy) and re-run "
            "`python -m src.external_eval --split valid`."
        ),
        "no_metrics_were_computed": True,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(C.EXTERNAL_STATUS_JSON, payload)
    logger.warning("%s | %s", BLOCKED_STATUS, reason)
    return payload


# --------------------------------------------------------------------------- #
# Label handling
# --------------------------------------------------------------------------- #
def apply_uncertainty_policy(values: pd.Series, policy: str = C.EXTERNAL_UNCERTAINTY_POLICY
                             ) -> np.ndarray:
    """CheXpert {1, 0, −1, NaN} → {1, 0}.

    policy = "u_zeroes"  uncertain(−1) → 0   (default; conservative)
    policy = "u_ones"    uncertain(−1) → 1
    NaN (label not mentioned) → 0 under both policies.
    """
    s = pd.to_numeric(values, errors="coerce")
    if policy == "u_zeroes":
        out = s.replace(-1.0, 0.0)
    elif policy == "u_ones":
        out = s.replace(-1.0, 1.0)
    else:
        raise ValueError(f"unknown uncertainty policy {policy!r}")
    out = out.fillna(0.0)
    if not set(np.unique(out.to_numpy())) <= {0.0, 1.0}:
        raise ValueError(f"labels did not reduce to 0/1: {np.unique(out.to_numpy())}")
    return out.to_numpy().astype(np.int64)


def load_external_frame(
    csv_path: Optional[Path] = None,
    images_root: Optional[Path] = None,
    split: str = "valid",
    uncertainty_policy: str = C.EXTERNAL_UNCERTAINTY_POLICY,
) -> pd.DataFrame:
    """Build {image_path, split, true_<label>} for the external cohort.

    Raises ExternalDataNotAvailable when the cohort is absent — this is the
    guard that keeps a BLOCKED phase from ever producing a number.
    """
    status = check_external_data(csv_path, images_root)
    if not status["available"]:
        raise ExternalDataNotAvailable(
            f"{BLOCKED_STATUS}: " + "; ".join(status["problems"])  # type: ignore[arg-type]
        )
    csv_path = Path(status["label_csv"])  # type: ignore[arg-type]
    images_root = Path(status["images_dir"])  # type: ignore[arg-type]

    raw = pd.read_csv(csv_path)
    if "Path" not in raw.columns:
        raise ValueError(f"{csv_path} has no 'Path' column — expected the official "
                         "CheXpert CSV layout")

    out = pd.DataFrame()
    out["image_path"] = raw["Path"].astype(str).map(lambda p: str(images_root / p))
    out["split"] = split
    out["Image Index"] = raw["Path"].astype(str)
    for target, che_column in C.CHEXPERT_LABEL_MAP.items():
        if che_column not in raw.columns:
            raise ValueError(
                f"label column '{che_column}' not found in {csv_path.name}; "
                f"available columns: {list(raw.columns)}"
            )
        out[f"true_{target}"] = apply_uncertainty_policy(raw[che_column],
                                                          uncertainty_policy)

    missing = ~out["image_path"].map(lambda p: Path(p).exists())
    if missing.any():
        raise FileNotFoundError(
            f"{int(missing.sum())} of {len(out)} images referenced by "
            f"{csv_path.name} are absent under {images_root}"
        )
    logger.info("Loaded external frame: %d rows, labels %s",
                len(out), [c for c in out.columns if c.startswith("true_")])
    return out


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_external(
    frame: pd.DataFrame,
    checkpoint: Path = C.BASELINE_CHECKPOINT_BEST,
    batch_size: int = 32,
    device=None,
) -> pd.DataFrame:
    """Run the frozen model over the external frame (eval transforms only)."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    from .dataset import build_transforms  # reuses the frozen eval transform

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(C.TARGET_LABELS)
    model, label_names = load_model(Path(checkpoint), device)
    transform = build_transforms(train=False)

    class _Frame(Dataset):  # local: the external cohort is not a NIH split
        def __len__(self):
            return len(frame)

        def __getitem__(self, i):
            from PIL import Image

            row = frame.iloc[i]
            with Image.open(row["image_path"]) as im:
                img = im.convert("RGB")
            return (
                transform(image=np.array(img))["image"],
                np.array([row[f"true_{l}"] for l in label_names], dtype=np.float32),
            )

    loader = DataLoader(_Frame(), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    amp = bool(C.AMP and device.type == "cuda")
    logits: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(images)
            logits.append(out.float().cpu().numpy())
            targets.append(y.numpy())

    logits_arr = np.concatenate(logits)
    targets_arr = np.concatenate(targets)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    result = frame.reset_index(drop=True).copy()
    if len(result) != len(logits_arr):
        raise RuntimeError("row-count mismatch on the external cohort")
    for i, lbl in enumerate(label_names):
        result[f"true_{lbl}"] = targets_arr[:, i].astype(int)
        result[f"logit_{lbl}"] = logits_arr[:, i]
        result[f"prob_{lbl}"] = probs[:, i]
    return result


# --------------------------------------------------------------------------- #
# Evaluation (frozen parameters, no refitting)
# --------------------------------------------------------------------------- #
def evaluate_external(predictions: pd.DataFrame, split: str = "valid"
                      ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Metrics on the external cohort using the frozen val-derived parameters."""
    if not C.TEMPERATURE_FILE.exists() or not C.THRESHOLDS_FILE.exists():
        raise SystemExit("Frozen parameters missing — run `python -m src.fit_parameters`.")
    scaler = TemperatureScaler.load(C.TEMPERATURE_FILE)
    raw_thr = json.loads(C.THRESHOLDS_FILE.read_text())
    cal_thr = json.loads(
        (C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json").read_text())
    labels = list(C.TARGET_LABELS)

    logits = predictions[[f"logit_{l}" for l in labels]].to_numpy()
    y = predictions[[f"true_{l}" for l in labels]].to_numpy()
    raw = predictions[[f"prob_{l}" for l in labels]].to_numpy()
    cal = scaler.probabilities(logits)

    report = multilabel_report(y, cal, labels, C.DECISION_THRESHOLD)
    results: Dict[str, object] = {}
    summary_rows: List[Dict[str, object]] = []
    for variant, probs in (("raw", raw), ("calibrated", cal)):
        thresholds = (raw_thr if variant == "raw" else cal_thr)["policies"]
        per_label: Dict[str, object] = {}
        for i, lbl in enumerate(labels):
            entry: Dict[str, object] = {
                "calibration": calibration_report(y[:, i], probs[:, i]),
                "auroc": auroc(y[:, i], probs[:, i]),
                "auprc": auprc(y[:, i], probs[:, i]),
                "prevalence": float(y[:, i].mean()),
            }
            for policy in POLICIES:
                tau = float(thresholds[policy][lbl]["threshold"])
                m = binary_metrics(y[:, i], probs[:, i], tau)
                m["threshold"] = tau
                m["fitted_on"] = "val (frozen)"
                entry[policy] = m
                summary_rows.append({
                    "split": split, "label": lbl, "policy": policy,
                    "variant": variant, "tau": tau, "precision": m["precision"],
                    "recall": m["recall"], "f1": m["f1"],
                    "specificity": m["specificity"],
                })
            per_label[lbl] = entry
            curves_with_thresholds(
                y[:, i], probs[:, i],
                {p: float(thresholds[p][lbl]["threshold"]) for p in POLICIES},
                f"external {split} / {variant} — {lbl}",
                C.PLOTS_DIR / "external" / f"{variant}_roc_{split}_{lbl.lower()}.png",
                C.PLOTS_DIR / "external" / f"{variant}_pr_{split}_{lbl.lower()}.png",
            )
        results[variant] = per_label

    rows = [{"label": l, **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in results["calibrated"][l].items()
                            if k in ("auroc", "auprc")}}
            for l in labels]
    logger.info("\n%s", format_metrics_table(rows, ["label", "auroc", "auprc"]))

    payload = {
        "phase": "external_validation",
        "status": "COMPLETED",
        "project_status": {
            "core_research": "COMPLETE",
            "external_validation": "COMPLETED for " + split,
        },
        "dataset": "CheXpert (external)",
        "split": split,
        "n_images": int(len(predictions)),
        "checkpoint": str(C.BASELINE_CHECKPOINT_BEST),
        "checkpoint_sha256": sha256_file(C.BASELINE_CHECKPOINT_BEST),
        "parameters": {
            "temperature_file": str(C.TEMPERATURE_FILE),
            "thresholds_file": str(C.THRESHOLDS_FILE),
            "fit_split": "val",
            "refit_on_external_data": False,
        },
        "uncertainty_policy": C.EXTERNAL_UNCERTAINTY_POLICY,
        "label_map": dict(C.CHEXPERT_LABEL_MAP),
        "multilabel_report_calibrated_fixed_threshold": report,
        "results": results,
        "environment": capture_environment(include_packages=False),
    }
    return payload, summary_rows


def run(
    split: str = "valid", checkpoint: Path = C.BASELINE_CHECKPOINT_BEST,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """Full external run, or an explicit BLOCKED status when the data is absent."""
    status = check_external_data()
    if not status["available"]:
        return write_blocked_status("; ".join(status["problems"]))  # type: ignore[arg-type]

    frame = load_external_frame(split=split)
    if limit:
        frame = frame.head(int(limit))
        logger.warning("limit=%d → evaluating a subset (never quote subset as full)",
                       limit)
    predictions = predict_external(frame, checkpoint)
    pred_path = (C.EXTERNAL_PREDICTIONS_DIR /
                 f"external_{split}_predictions.csv")
    predictions.to_csv(pred_path, index=False)

    payload, rows = evaluate_external(predictions, split)
    payload["predictions_file"] = str(pred_path)
    payload["limit"] = limit
    out = write_json(C.EXTERNAL_METRICS_DIR / f"external_eval_{split}.json", payload)
    pd.DataFrame(rows).to_csv(
        C.EXTERNAL_METRICS_DIR / f"external_eval_{split}.csv", index=False)
    write_json(C.EXTERNAL_STATUS_JSON, {
        "phase": "external_validation", "status": "COMPLETED",
        "project_status": {
            "core_research": "COMPLETE",
            "external_validation": "COMPLETED for " + split,
        },
        "split": split, "n_images": int(len(predictions)),
        "record": str(out),
    })
    logger.info("External evaluation -> %s", out)
    return payload


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External (CheXpert) evaluation")
    p.add_argument("--split", default="valid")
    p.add_argument("--checkpoint", type=Path, default=C.BASELINE_CHECKPOINT_BEST)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--status", action="store_true",
                   help="only refresh outputs/metrics/external/status.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        s = check_external_data()
        if s["available"]:
            logger.info("External data present — run without --status to evaluate")
        else:
            write_blocked_status("; ".join(s["problems"]))  # type: ignore[arg-type]
    else:
        try:
            run(args.split, args.checkpoint, args.limit)
        except ExternalDataNotAvailable as exc:
            write_blocked_status(str(exc))
