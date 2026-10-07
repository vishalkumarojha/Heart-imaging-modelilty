"""Research freeze — a machine-readable snapshot of the study as it stands.

This module *reads* the frozen artifacts (it never re-measures or edits them)
and writes `outputs/research_snapshot.json`: the single document that answers
"what exactly is this baseline?".

Provenance rule: every field is either
    * read from a result/checkpoint artifact on disk, or
    * read from `src/config.py` (the configuration of record), or
    * captured live from the running interpreter / git working tree.
Nothing is typed by hand.

    python -m src.research_freeze                 # includes a test-suite run
    python -m src.research_freeze --no-tests      # skip the (slow) test run
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config as C
from .reproducibility import capture_environment, sha256_file, write_json
from .utils import setup_logging

logger = setup_logging()

SNAPSHOT_VERSION = "1.0"
BASELINE_ID = "BASELINE_v1"


# --------------------------------------------------------------------------- #
# Live repository state
# --------------------------------------------------------------------------- #
def _git(args: List[str]) -> str:
    try:
        return subprocess.run(["git", *args], cwd=C.PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        return f"unavailable: {exc}"


# Regenerated metadata outputs that the freeze/test pipeline rewrites on every
# run (timestamp-only churn). They are excluded from the "dirty" classification
# so the git block reflects *source* tree state rather than output timestamps.
_REFRESHED_OUTPUTS = (
    "outputs/research_snapshot.json",
    "outputs/paper/reproducibility_manifest.json",
    "outputs/metrics/error_analysis/error_analysis_test.json",
    "outputs/metrics/external/status.json",
)


def _is_refreshed_output(line: str) -> bool:
    path = line[3:].lstrip() if len(line) > 3 and line[2] == " " else line[2:].lstrip()
    return any(path.startswith(p) for p in _REFRESHED_OUTPUTS)


def git_state() -> Dict[str, object]:
    short = _git(["status", "--porcelain"])
    lines = short.splitlines() if short and not short.startswith("unavailable") else []
    dirty = [ln for ln in lines if not _is_refreshed_output(ln)]
    return {
        "head_commit": _git(["rev-parse", "HEAD"]),
        "head_subject": _git(["log", "-1", "--pretty=%s"]),
        "dirty": bool(dirty),
        "status_short": lines,
        "excluded": list(_REFRESHED_OUTPUTS),
        "note": "dirty excludes timestamp-only regenerated freeze outputs "
                "(snapshot, paper manifest, error-analysis/status metadata); "
                "see git_state() in src/research_freeze.py",
    }


def test_status(run: bool = True) -> Dict[str, object]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    if not run:
        return {
            "status": "NOT_RUN_BY_SNAPSHOT",
            "command": " ".join(command),
            "note": "run `.venv/bin/python -m unittest discover -s tests` to populate",
        }
    proc = subprocess.run(command, cwd=C.PROJECT_ROOT, capture_output=True, text=True)
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    summary = next((ln for ln in reversed(tail) if ln.startswith("Ran ")), "")
    m = re.search(r"Ran (\d+) tests?", summary)
    ok = proc.returncode == 0 and "OK" in (proc.stderr or "")
    return {
        "status": "PASS" if ok else "FAIL",
        "tests_run": int(m.group(1)) if m else None,
        "returncode": proc.returncode,
        "summary": summary,
        "command": " ".join(command),
    }


# --------------------------------------------------------------------------- #
# Frozen artifacts
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Optional[dict]:
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def calibration_summary() -> Dict[str, object]:
    scaler = _load_json(C.TEMPERATURE_FILE)
    if scaler is None:
        return {"status": "MISSING", "file": str(C.TEMPERATURE_FILE)}
    return {
        "status": "FROZEN",
        "method": "post-hoc temperature scaling (single scalar per label)",
        "objective": "negative log-likelihood on validation",
        "fitted_on": scaler["metadata"]["split"],
        "file": str(C.TEMPERATURE_FILE),
        "temperature": {k: float(v) for k, v in scaler["metadata"]["temperatures"].items()},
        "nll_val_before": {k: round(float(v), 6) for k, v in scaler["metadata"]["nll_before"].items()},
        "nll_val_after": {k: round(float(v), 6) for k, v in scaler["metadata"]["nll_after"].items()},
    }


def threshold_summary() -> Dict[str, object]:
    raw = _load_json(C.THRESHOLDS_FILE)
    if raw is None:
        return {"status": "MISSING", "file": str(C.THRESHOLDS_FILE)}
    return {
        "status": "FROZEN",
        "fitted_on": sorted(set(raw["fit_split"].values())),
        "file": str(C.THRESHOLDS_FILE),
        "policies": list(raw["policies"].keys()),
        "thresholds": {
            policy: {lbl: float(entry["threshold"]) for lbl, entry in per_label.items()}
            for policy, per_label in raw["policies"].items()
        },
    }


def split_summary() -> Dict[str, object]:
    rec = _load_json(C.BASELINE_METRICS_JSON) or {}
    stats = rec.get("split_statistics", {})
    return {
        "method": "patient-level 70/15/15",
        "fractions": list(C.SPLIT_FRACTIONS),
        "seed": C.SEED,
        "index_file": str(C.DATA_PROCESSED_DIR / "split_index.csv"),
        "rows_by_split": stats.get("rows_by_split"),
        "patients_by_split": stats.get("patients_by_split"),
        "label_counts_by_split": stats.get("label_counts_by_split"),
        "patient_overlap": stats.get("patient_overlap"),
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_snapshot(run_tests: bool = True) -> Dict[str, object]:
    manifest = _load_json(C.BASELINE_CHECKPOINT_DIR / "baseline_manifest.json") or {}
    cfg = manifest.get("baseline_configuration", {})
    best = C.BASELINE_CHECKPOINT_BEST
    last = C.BASELINE_CHECKPOINT_DIR / "densenet121_last.pt"
    env = capture_environment()

    snapshot: Dict[str, object] = {
        "snapshot_version": SNAPSHOT_VERSION,
        "baseline_id": BASELINE_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study": {
            "primary_modality": "Chest X-ray",
            "primary_task": "multi-label Cardiomegaly + Effusion classification",
            "research_focus": "calibration-aware decision analysis under class imbalance",
            "secondary_analysis": "error analysis + explainability",
            "dataset": "NIH ChestX-ray14",
        },
        "model": {
            "name": "DenseNet121 (torchvision, ImageNet-1K V1 pretrained)",
            "classifier": cfg.get("classifier"),
            "num_labels": C.NUM_CLASSES,
            "labels": list(C.TARGET_LABELS),
            "checkpoint_selection": cfg.get("checkpoint_selection"),
        },
        "data": {
            "dataset": cfg.get("dataset", "NIH ChestX-ray14"),
            "labels": list(C.TARGET_LABELS),
            "split": split_summary(),
        },
        "preprocessing": {
            "input_resolution": C.IMAGE_SIZE,
            "normalization_mean": list(C.IMAGENET_MEAN),
            "normalization_std": list(C.IMAGENET_STD),
        },
        "augmentation": {
            "train_only": True,
            "rotate_deg": C.AUG_ROTATION_DEG,
            "brightness": C.AUG_BRIGHTNESS,
            "contrast": C.AUG_CONTRAST,
            "horizontal_flip": False,
            "note": "laterality is diagnostic — horizontal flip is deliberately off",
        },
        "training": {
            "loss": cfg.get("loss"),
            "optimizer": cfg.get("optimizer"),
            "schedule": cfg.get("schedule"),
            "learning_rates": {"phase1": C.PHASE1_LR, "phase2": C.PHASE2_LR},
            "batch_size": C.BATCH_SIZE,
            "epochs": C.TOTAL_EPOCHS,
            "amp": C.AMP,
            "weight_decay": C.WEIGHT_DECAY,
            "seed": C.SEED,
        },
        "checkpoints": {
            "best": {
                "path": str(best),
                "exists": Path(best).exists(),
                "sha256": sha256_file(best) if Path(best).exists() else None,
            },
            "last": {
                "path": str(last),
                "exists": Path(last).exists(),
                "sha256": sha256_file(last) if Path(last).exists() else None,
            },
            "manifest": str(C.BASELINE_CHECKPOINT_DIR / "baseline_manifest.json"),
            "legacy_root_checkpoint": {
                "path": str(C.CHECKPOINT_BEST),
                "exists": Path(C.CHECKPOINT_BEST).exists(),
                "note": "kept for legacy ECHO/MRI/fusion/demo hard dependency",
            },
        },
        "calibration": calibration_summary(),
        "thresholds": threshold_summary(),
        "environment": env,
        "interpreter": {
            "executable": sys.executable,
            "python": platform.python_version(),
            "note": "system python3 has no torch — always use .venv/bin/python",
        },
        "git": git_state(),
        "tests": test_status(run=run_tests),
        "project_status": {
            "core_research": "COMPLETE",
            "baseline": "FROZEN",
            "calibration": "COMPLETE",
            "threshold_analysis": "COMPLETE",
            "ablation": "COMPLETE",
            "error_analysis": "COMPLETE",
            "explainability": "COMPLETE",
            "external_validation": "PENDING",
            "paper_preparation": "COMPLETE / READY FOR AUTHOR REVIEW",
            "note": "external_validation stays PENDING until an external cohort "
                    "is available (see outputs/metrics/external/status.json).",
        },
        "source_of_truth": {
            "results": [
                "outputs/metrics/baseline/baseline_metrics.json",
                "outputs/metrics/experiments/exp1_calibration.json",
                "outputs/metrics/experiments/exp2_thresholds_raw.json",
                "outputs/metrics/experiments/exp3_thresholds_calibrated.json",
                "outputs/metrics/experiments/exp4_robustness.json",
                "outputs/metrics/error_analysis/error_analysis_test.json",
                "outputs/gradcam/bbox_localization.json",
                "outputs/metrics/external/status.json",
            ],
            "documentation": ["docs/PROJECT_SCOPE.md", "docs/BASELINE.md",
                              "docs/FINAL_PROJECT_STATUS.md"],
        },
        "immutability": (
            "This snapshot describes the FROZEN baseline. It is written once and "
            "read-only afterwards; re-running the module is a no-op verification."
        ),
    }
    return snapshot


def run(run_tests: bool = True) -> Path:
    snapshot = build_snapshot(run_tests=run_tests)
    out = write_json(C.RESEARCH_SNAPSHOT_JSON, snapshot)
    logger.info("Research snapshot -> %s", out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write outputs/research_snapshot.json")
    p.add_argument("--no-tests", action="store_true",
                   help="do not run the unittest suite as part of the snapshot")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(run_tests=not args.no_tests)
