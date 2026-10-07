"""Central configuration for the Chest X-ray pipeline (baseline + research).

Everything tunable lives here so the rest of the code stays declarative.
Keep this file free of heavy imports / model logic so it can be imported
cheaply from scripts, notebooks and the multi-modality fusion code.

Two layers live in this file:

1. BASELINE (Phase 1) — the frozen experimental control.  Do not change these
   values: `docs/BASELINE.md` records them and `outputs/checkpoints/baseline/`
   holds the immutable, hash-verified copies of the trained artifacts.
2. RESEARCH OUTPUT STRUCTURE — where calibration / threshold / external
   evaluation artifacts are written (self-describing names, never overwritten).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_ENTRY_CSV = DATA_RAW_DIR / "Data_Entry_2017.csv"
BBOX_CSV = DATA_RAW_DIR / "BBox_List_2017.csv"
# Official NIH split files (used automatically if present anywhere under data/raw/)
OFFICIAL_TRAINVAL_LIST = "train_val_list.txt"
OFFICIAL_TEST_LIST = "test_list.txt"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
LOG_DIR = OUTPUTS_DIR / "logs"
METRICS_CSV = LOG_DIR / "metrics.csv"

# Cache of the resolved filename -> absolute path map + split assignment.
SPLIT_CACHE_CSV = DATA_PROCESSED_DIR / "split_index.csv"
IMAGE_PATH_CACHE = DATA_PROCESSED_DIR / "image_path_index.csv"

# --------------------------------------------------------------------------- #
# Checkpoint hierarchy:  baseline (frozen control)  vs  research (new runs)
# --------------------------------------------------------------------------- #
# NOTE: outputs/checkpoints/densenet121_{best,last}.pt (root) remain the live
# training targets AND are referenced by the legacy fusion code
# (src/fusion_config.py) and the demo app (demo_app.py).  They must not move.
BASELINE_CHECKPOINT_DIR = CHECKPOINT_DIR / "baseline"
RESEARCH_CHECKPOINT_DIR = CHECKPOINT_DIR / "research"
BASELINE_CHECKPOINT_BEST = BASELINE_CHECKPOINT_DIR / "densenet121_best.pt"
BASELINE_CHECKPOINT_LAST = BASELINE_CHECKPOINT_DIR / "densenet121_last.pt"
BASELINE_MANIFEST = BASELINE_CHECKPOINT_DIR / "baseline_manifest.json"

# --------------------------------------------------------------------------- #
# Analysis output structure (all machine-readable, self-describing filenames)
# --------------------------------------------------------------------------- #
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
PLOTS_DIR = OUTPUTS_DIR / "plots"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

BASELINE_METRICS_DIR = METRICS_DIR / "baseline"
BASELINE_PREDICTIONS_DIR = PREDICTIONS_DIR / "baseline"
RAW_PREDICTIONS_DIR = PREDICTIONS_DIR / "raw"          # cached logits/probabilities
CALIBRATION_METRICS_DIR = METRICS_DIR / "calibration"
THRESHOLD_METRICS_DIR = METRICS_DIR / "thresholds"
EXPERIMENT_METRICS_DIR = METRICS_DIR / "experiments"
ERROR_ANALYSIS_DIR = METRICS_DIR / "error_analysis"
EXTERNAL_METRICS_DIR = METRICS_DIR / "external"
EXTERNAL_PREDICTIONS_DIR = PREDICTIONS_DIR / "external"

BASELINE_METRICS_JSON = BASELINE_METRICS_DIR / "baseline_metrics.json"  # aggregate (all splits)
# per-split records are written by src.baseline_eval as
# outputs/metrics/baseline/baseline_metrics_{test,val}.json / .csv

# --------------------------------------------------------------------------- #
# Research settings — calibration & thresholds (Phase 4)
# --------------------------------------------------------------------------- #
# Calibration and threshold fitting may ONLY see validation predictions.
# `src.calibration` / `src.thresholds` enforce this at runtime (ValueError on
# 'test'/'external'), and the tests assert the guard exists.
CALIBRATION_FIT_SPLIT = "val"
THRESHOLD_FIT_SPLIT = "val"

# Expected calibration error / reliability diagrams
CALIBRATION_BINS = 15                      # standard ECE bin count
CALIBRATION_STRATEGY = "equal_width"       # or "equal_freq"
TEMPERATURE_BOUNDS = (0.05, 50.0)          # search range for the scalar T

# Threshold policies (validation-derived; evaluated frozen on test/external)
THRESHOLD_POLICIES = (
    "fixed",
    "f1_optimal",
    "youden",
    "sensitivity_constrained",
    "precision_constrained",
)
SENSITIVITY_TARGET = 0.90                  # screening-style operating point
PRECISION_TARGET = 0.50                    # referral-style operating point

# Fitted-parameter artifacts (written once per experiment, then frozen)
TEMPERATURE_FILE = CALIBRATION_METRICS_DIR / "temperature_scalers.json"
THRESHOLDS_FILE = THRESHOLD_METRICS_DIR / "thresholds_val.json"
THRESHOLDS_FROZEN_MARK = THRESHOLD_METRICS_DIR / "thresholds_frozen.json"

# --------------------------------------------------------------------------- #
# External validation (Phase 6) — NOT AVAILABLE in this repository
# --------------------------------------------------------------------------- #
# The pipeline below is IMPLEMENTED; the data is absent, so every external
# result is reported as "BLOCKED — EXTERNAL DATASET NOT AVAILABLE".
# Expected official CheXpert layout (unchosen/unverified until the data lands):
#   data/raw/chexpert/valid.csv      studies + labels for the validation cohort
#   data/raw/chexpert/train.csv      optional larger labeled cohort
#   data/raw/chexpert/images/…       per-study image folders
CHEXPERT_DIR = DATA_RAW_DIR / "chexpert"
CHEXPERT_VALID_CSV = CHEXPERT_DIR / "valid.csv"
CHEXPERT_TRAIN_CSV = CHEXPERT_DIR / "train.csv"
CHEXPERT_IMAGES_DIR = CHEXPERT_DIR / "images"
EXTERNAL_STATUS_JSON = EXTERNAL_METRICS_DIR / "status.json"

# NIH label -> CheXpert column (Pleural Effusion is the CheXpert name).
CHEXPERT_LABEL_MAP = {
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
}
# CheXpert marks uncertain findings with -1. "u_zeroes" = uncertain -> absent,
# the conservative choice for a two-class (present/absent) evaluation.
EXTERNAL_UNCERTAINTY_POLICY = "u_zeroes"

# --------------------------------------------------------------------------- #
# Paper-ready evidence (final phase) — canonical roll-up of verified results
# --------------------------------------------------------------------------- #
# Everything under final_results/ is DERIVED from the experiment artifacts above;
# nothing new is measured here. If an artifact is missing, the producing script
# fails loudly rather than writing an empty cell.
FINAL_RESULTS_DIR = OUTPUTS_DIR / "final_results"
FINAL_FIGURES_DIR = FINAL_RESULTS_DIR / "figures"
FINAL_GRADCAM_DIR = FINAL_RESULTS_DIR / "gradcam"
MASTER_RESULTS_CSV = FINAL_RESULTS_DIR / "master_results.csv"
CONFIDENCE_INTERVALS_CSV = FINAL_RESULTS_DIR / "confidence_intervals.csv"
ERROR_ANALYSIS_CSV = FINAL_RESULTS_DIR / "error_analysis.csv"
REGION_ANALYSIS_JSON = FINAL_RESULTS_DIR / "region_analysis.json"
RESEARCH_SUMMARY_JSON = FINAL_RESULTS_DIR / "research_summary.json"
RESEARCH_SNAPSHOT_JSON = OUTPUTS_DIR / "research_snapshot.json"
VERIFICATION_REPORT_JSON = FINAL_RESULTS_DIR / "verification_report.json"
FINAL_TABLES_DIR = FINAL_RESULTS_DIR / "tables"

# --------------------------------------------------------------------------- #
# Paper package (final phase) — publication-facing roll-up derived ONLY from the
# canonical final_results/ artifacts. Nothing new is measured here; the evidence
# map ties every paper number back to a source artifact.
# --------------------------------------------------------------------------- #
PAPER_DIR = OUTPUTS_DIR / "paper"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_EVIDENCE_MAP_JSON = PAPER_DIR / "paper_evidence_map.json"
PAPER_CLAIM_AUDIT_CSV = PAPER_DIR / "claim_audit.csv"
PAPER_REPRO_MANIFEST_JSON = PAPER_DIR / "reproducibility_manifest.json"
PAPER_OUTLINE_MD = PAPER_DIR / "paper_outline.md"
PAPER_RESULTS_MD = PAPER_DIR / "draft_results.md"
PAPER_DISCUSSION_MD = PAPER_DIR / "draft_discussion.md"
PAPER_LIMITATIONS_MD = PAPER_DIR / "draft_limitations.md"
PAPER_ABSTRACT_MD = PAPER_DIR / "abstract.md"
PAPER_CONCLUSION_MD = PAPER_DIR / "conclusion.md"

# Statistical settings for the uncertainty analysis (fixed seed -> reproducible)
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_SEED = 42

for _d in (
    DATA_PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR,
    BASELINE_CHECKPOINT_DIR, RESEARCH_CHECKPOINT_DIR,
    BASELINE_METRICS_DIR, BASELINE_PREDICTIONS_DIR, RAW_PREDICTIONS_DIR,
    CALIBRATION_METRICS_DIR, THRESHOLD_METRICS_DIR, EXPERIMENT_METRICS_DIR,
    ERROR_ANALYSIS_DIR, EXTERNAL_METRICS_DIR, EXTERNAL_PREDICTIONS_DIR,
    PLOTS_DIR / "training", PLOTS_DIR / "roc", PLOTS_DIR / "pr",
    PLOTS_DIR / "calibration", PLOTS_DIR / "threshold", PLOTS_DIR / "external",
    PLOTS_DIR / "error_analysis", GRADCAM_DIR,
    FINAL_RESULTS_DIR, FINAL_FIGURES_DIR, FINAL_GRADCAM_DIR, FINAL_TABLES_DIR,
    PAPER_DIR, PAPER_TABLES_DIR, PAPER_FIGURES_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
# NOTE: Order matters. label_tensor[i] corresponds to TARGET_LABELS[i].
TARGET_LABELS: List[str] = ["Cardiomegaly", "Effusion"]
NUM_CLASSES = len(TARGET_LABELS)

# --------------------------------------------------------------------------- #
# Image / preprocessing
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentation (training only). Chest X-ray laterality matters -> NO horizontal flip.
AUG_ROTATION_DEG = 10
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST = 0.15

# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test  (patient-level)
SEED = 42

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
NUM_WORKERS = 4

# Two-phase schedule
PHASE1_EPOCHS = 5          # frozen backbone (final dense block + classifier train)
PHASE1_LR = 1e-4
PHASE2_EPOCHS = 5          # fully unfrozen
PHASE2_LR = 1e-5
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

WEIGHT_DECAY = 0.0
AMP = True                 # mixed precision (helps a lot on 4 GB GPUs)

# --quick_test overrides
QUICK_TEST_TRAIN_SAMPLES = 500
QUICK_TEST_VAL_SAMPLES = 100
QUICK_TEST_EPOCHS = 1

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
CHECKPOINT_BEST = CHECKPOINT_DIR / "densenet121_best.pt"
CHECKPOINT_LAST = CHECKPOINT_DIR / "densenet121_last.pt"  # written every epoch, for --resume
PREDICTIONS_CSV = LOG_DIR / "test_predictions.csv"
DECISION_THRESHOLD = 0.5


@dataclass
class RunConfig:
    """A mutable snapshot of the knobs a single run actually used.

    Scripts build one of these (optionally patched by CLI flags) and pass it
    around instead of reaching back into module globals. Makes runs
    reproducible and keeps `train.py` / `evaluate.py` testable.
    """

    image_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    seed: int = SEED

    phase1_epochs: int = PHASE1_EPOCHS
    phase1_lr: float = PHASE1_LR
    phase2_epochs: int = PHASE2_EPOCHS
    phase2_lr: float = PHASE2_LR
    weight_decay: float = WEIGHT_DECAY
    amp: bool = AMP

    target_labels: List[str] = field(default_factory=lambda: list(TARGET_LABELS))

    quick_test: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.target_labels)

    def apply_quick_test(self) -> "RunConfig":
        self.quick_test = True
        self.phase1_epochs = QUICK_TEST_EPOCHS
        self.phase2_epochs = 0
        return self
