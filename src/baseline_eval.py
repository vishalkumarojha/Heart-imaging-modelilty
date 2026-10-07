"""Baseline evaluation — the frozen experimental control (Phase 2).

Runs the *unmodified* DenseNet121 baseline at the fixed decision threshold 0.50
and writes machine-readable results:

    outputs/metrics/baseline/baseline_metrics.json        aggregate (all splits)
    outputs/metrics/baseline/baseline_metrics_<split>.json  one split, immutable
    outputs/metrics/baseline/baseline_metrics_<split>.csv   tidy rows per label
    outputs/predictions/baseline/baseline_<split>_predictions.csv

No calibration, no threshold tuning, no model changes — this script only
*measures* the control.  It reads logits/probabilities through the shared
cached-inference layer (`src.inference`) so later experiments score against
exactly the same predictions.

    python -m src.baseline_eval                       # test split, threshold 0.5
    python -m src.baseline_eval --split val
    python -m src.baseline_eval --refresh             # recompute cached logits
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config as C
from .inference import as_arrays, ensure_predictions, label_names_from
from .metrics import multilabel_report, report_to_rows
from .reproducibility import (
    capture_environment,
    capture_split_statistics,
    sha256_file,
    write_json,
)
from .utils import format_metrics_table, setup_logging

logger = setup_logging()

DISPLAY_COLUMNS = [
    "label", "auroc", "auprc", "sensitivity", "specificity",
    "precision", "f1", "accuracy", "tp", "fp", "fn", "tn",
    "support_pos", "support_neg",
]


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def evaluate_baseline(
    checkpoint: Path = C.BASELINE_CHECKPOINT_BEST,
    split: str = "test",
    threshold: float = C.DECISION_THRESHOLD,
    refresh: bool = False,
) -> Dict[str, object]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"Baseline checkpoint not found: {checkpoint}\n"
            "The frozen control is outputs/checkpoints/baseline/densenet121_best.pt "
            "(see docs/BASELINE.md)."
        )

    preds = ensure_predictions(checkpoint, split, refresh=refresh)
    labels = label_names_from(preds)
    arrays = as_arrays(preds, labels)
    report = multilabel_report(arrays["y_true"], arrays["probs"], labels, threshold)

    rows = report_to_rows(report, labels)
    display = [
        {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
        for row in rows
    ]
    logger.info(
        "\n===== BASELINE (%s, split=%s, threshold=%.2f) =====\n%s",
        checkpoint.name, split, threshold,
        format_metrics_table(display, DISPLAY_COLUMNS),
    )
    logger.info(
        "macro AUROC=%.4f  macro AUPRC=%.4f",
        report["macro"]["auroc"], report["macro"]["auprc"],
    )

    # ---- persist predictions ------------------------------------------------
    pred_out = C.BASELINE_PREDICTIONS_DIR / f"baseline_{split}_predictions.csv"
    preds.to_csv(pred_out, index=False)
    logger.info("Predictions -> %s", pred_out)

    # ---- persist metrics (per-split files are immutable; the aggregate merges) --
    payload: Dict[str, object] = {
        "experiment": f"baseline_densenet121_{split}",
        "status": "COMPLETED",
        "role": "experimental control (frozen)",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": split,
        "threshold": threshold,
        "labels": labels,
        "n_images": int(len(preds)),
        "metrics": report,
        "predictions_file": str(pred_out),
        "baseline_configuration": baseline_configuration(),
        "split_statistics": capture_split_statistics(
            pd.read_csv(C.SPLIT_CACHE_CSV)
        ),
        "environment": capture_environment(),
        "notes": [
            "Raw (uncalibrated) sigmoid probabilities; single fixed threshold 0.50.",
            "AUPRC was not part of the original Phase 1 evaluation; it is added "
            "here as the class-imbalance-aware companion to AUROC.",
        ],
    }
    split_json = C.BASELINE_METRICS_DIR / f"baseline_metrics_{split}.json"
    split_csv = C.BASELINE_METRICS_DIR / f"baseline_metrics_{split}.csv"
    write_json(split_json, payload)

    tidy = pd.DataFrame(rows)
    tidy.insert(0, "split", split)
    tidy.insert(0, "experiment", payload["experiment"])
    tidy.to_csv(split_csv, index=False)
    logger.info("Metrics -> %s and %s", split_json, split_csv)

    _merge_aggregate(payload)
    _write_manifest(checkpoint, payload)
    return payload


def _merge_aggregate(payload: Dict[str, object]) -> None:
    """Accumulate every evaluated split into outputs/metrics/baseline/baseline_metrics.json.

    Re-running one split never erases the others (each split also lives in its
    own baseline_metrics_<split>.json/csv).
    """
    split = str(payload["split"])
    agg: Dict[str, object] = {}
    if C.BASELINE_METRICS_JSON.exists():
        import json

        agg = json.loads(C.BASELINE_METRICS_JSON.read_text())
    agg.setdefault("experiment", "baseline_densenet121")
    agg.setdefault("status", "COMPLETED")
    agg.setdefault("role", "experimental control (frozen)")
    agg["source_files"] = "outputs/metrics/baseline/baseline_metrics_<split>.json"
    splits = dict(agg.get("splits", {}))  # type: ignore[arg-type]
    splits[split] = payload
    agg["splits"] = splits
    agg["splits_completed"] = sorted(splits)
    write_json(C.BASELINE_METRICS_JSON, agg)
    logger.info("Aggregate metrics -> %s (splits: %s)",
                C.BASELINE_METRICS_JSON, ", ".join(sorted(splits)))


def baseline_configuration() -> Dict[str, object]:
    """The frozen baseline hyper-parameters, read from the config that owns them."""
    return {
        "dataset": "NIH ChestX-ray14",
        "labels": list(C.TARGET_LABELS),
        "architecture": "densenet121 (torchvision, ImageNet-1K V1 pretrained)",
        "classifier": "Linear(1024, 2), logits out; sigmoid applied for metrics",
        "image_size": C.IMAGE_SIZE,
        "normalization": {"mean": list(C.IMAGENET_MEAN), "std": list(C.IMAGENET_STD)},
        "augmentation": {
            "rotate_deg": C.AUG_ROTATION_DEG,
            "brightness": C.AUG_BRIGHTNESS,
            "contrast": C.AUG_CONTRAST,
            "horizontal_flip": False,
        },
        "split": "patient-level 70/15/15, seed 42",
        "loss": "BCEWithLogitsLoss(pos_weight from train prevalence)",
        "optimizer": "Adam (weight_decay=0), no LR scheduler",
        "schedule": {
            "phase1_epochs": C.PHASE1_EPOCHS, "phase1_lr": C.PHASE1_LR,
            "phase2_epochs": C.PHASE2_EPOCHS, "phase2_lr": C.PHASE2_LR,
        },
        "batch_size": C.BATCH_SIZE,
        "amp": C.AMP,
        "seed": C.SEED,
        "checkpoint_selection": "max mean validation AUROC",
    }


def _write_manifest(checkpoint: Path, payload: Dict[str, object]) -> None:
    """Hash-anchor the baseline so any later tampering is detectable."""
    if Path(checkpoint).resolve() != C.BASELINE_CHECKPOINT_BEST.resolve():
        return  # only the frozen baseline gets a manifest
    if C.BASELINE_MANIFEST.exists():
        return  # never overwrite the freeze record
    write_json(C.BASELINE_MANIFEST, {
        "status": "FROZEN BASELINE — do not overwrite",
        "checkpoint": str(checkpoint),
        "sha256": payload["checkpoint_sha256"],
        "last_checkpoint_sha256": sha256_file(C.BASELINE_CHECKPOINT_LAST),
        "source_of_truth": "docs/BASELINE.md",
        "metrics_record": str(C.BASELINE_METRICS_JSON),
        "baseline_configuration": payload["baseline_configuration"],
    })
    logger.info("Baseline manifest -> %s", C.BASELINE_MANIFEST)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the frozen baseline at threshold 0.5")
    p.add_argument("--checkpoint", type=Path, default=C.BASELINE_CHECKPOINT_BEST)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=C.DECISION_THRESHOLD)
    p.add_argument("--refresh", action="store_true", help="recompute cached logits")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_baseline(args.checkpoint, args.split, args.threshold, args.refresh)
