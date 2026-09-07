"""Central configuration for the Chest X-ray (Phase 1) pipeline.

Everything tunable lives here so the rest of the code stays declarative.
Keep this file free of heavy imports / model logic so it can be imported
cheaply from scripts, notebooks and (later) the multi-modality fusion code.
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

for _d in (DATA_PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR):
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
