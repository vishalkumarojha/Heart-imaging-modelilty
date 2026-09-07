"""Phase 2 (EchoNet-Dynamic) configuration — EF-category classification.

Mirrors the structure of `src/config.py` (Phase 1). Kept import-light so it can
be pulled into the fusion code later without dragging video deps along.

Label contract (documented, fixed):
    EF >= 55            -> class 2  "Normal"
    40 <= EF < 55       -> class 1  "Mildly Reduced"
    EF < 40             -> class 0  "Reduced"
Ascending index == ascending heart function. Matches clinical severity ordering
(0 = worst). `bucket_ef()` is the single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ECHO_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "echonet" / "EchoNet-Dynamic"
VIDEO_DIR = ECHO_RAW_DIR / "Videos"
FILELIST_CSV = ECHO_RAW_DIR / "FileList.csv"
VOLUMETRACINGS_CSV = ECHO_RAW_DIR / "VolumeTracings.csv"  # not used this phase

ECHO_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "echo"
NORM_STATS_JSON = ECHO_PROCESSED_DIR / "norm_stats.json"      # computed channel mean/std
INDEX_CACHE_CSV = ECHO_PROCESSED_DIR / "echo_index.csv"       # FileName -> path/label/split

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ECHO_CKPT_DIR = OUTPUTS_DIR / "checkpoints" / "echo"
ECHO_LOG_DIR = OUTPUTS_DIR / "logs" / "echo"
METRICS_CSV = ECHO_LOG_DIR / "metrics.csv"
PREDICTIONS_CSV = ECHO_LOG_DIR / "test_predictions.csv"
CKPT_BEST = ECHO_CKPT_DIR / "echo_cnn_lstm_best.pt"
CKPT_LAST = ECHO_CKPT_DIR / "echo_cnn_lstm_last.pt"

for _d in (ECHO_PROCESSED_DIR, ECHO_CKPT_DIR, ECHO_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
EF_BOUNDARY_REDUCED = 40      # EF < 40  -> Reduced
EF_BOUNDARY_NORMAL = 55      # EF >= 55 -> Normal ; between -> Mildly Reduced

CLASS_NAMES = ["Reduced", "Mildly Reduced", "Normal"]   # index 0,1,2
NUM_CLASSES = 3


def bucket_ef(ef: float) -> int:
    """Continuous EF (%) -> ordinal class index (0 Reduced, 1 Mildly, 2 Normal)."""
    if ef < EF_BOUNDARY_REDUCED:
        return 0
    if ef < EF_BOUNDARY_NORMAL:
        return 1
    return 2


# --------------------------------------------------------------------------- #
# Video / preprocessing
# --------------------------------------------------------------------------- #
FRAMES_PER_CLIP = 16          # evenly-spaced sampled frames per video (fixed-size input)
FRAME_SIZE = 112              # videos are already 112x112; loader still resizes defensively
VIDEO_CHANNELS = 3

# Augmentation (train only). Echo apical-4-chamber has a conventional orientation;
# horizontal flip produces a view a sonographer never acquires -> OFF by default
# (see echo_explore notes). Safe photometric + tiny rotation only.
AUG_ROTATION_DEG = 10
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST = 0.15
AUG_HFLIP = False

# Channel mean/std: computed from a TRAIN subset by echo_explore.py and cached to
# NORM_STATS_JSON. These are only a fallback if the file is missing.
FALLBACK_MEAN = (0.129, 0.129, 0.129)
FALLBACK_STD = (0.190, 0.190, 0.190)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
D_ECHO = 1024                 # fusion contract: EchoEncoder.forward -> [B, 1024]
CNN_BACKBONE = "resnet18"     # per-frame feature extractor (512-d)
CNN_FEATURE_DIM = 512
RNN_TYPE = "lstm"
RNN_HIDDEN = 512              # bidirectional -> 2*512 = 1024 = D_ECHO
RNN_LAYERS = 1
RNN_BIDIRECTIONAL = True
RNN_DROPOUT = 0.0

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
SEED = 42
BATCH_SIZE = 8               # video tensors are large; profiled before the full run
NUM_WORKERS = 4
AMP = True

PHASE1_EPOCHS = 3            # CNN frozen (LSTM + head train)
PHASE1_LR = 1e-4
PHASE2_EPOCHS = 6            # unfrozen fine-tune (trimmed from 12: EchoNet train set is 10x smaller than Phase 1's -> overfit risk)
PHASE2_LR = 1e-5
WEIGHT_DECAY = 1e-4          # Phase 1 summary flagged 0 wd -> late overfitting
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

# class-imbalance handling: CrossEntropy weight = inverse train frequency (filled at runtime)
USE_CLASS_WEIGHTS = True

# checkpoint selection metric: "macro_auroc" (OvR) — accuracy is dominated by the
# ~70% Normal class, macro AUROC weights all 3 classes equally and is threshold-free.
CKPT_METRIC = "macro_auroc"

QUICK_TEST_TRAIN = 60
QUICK_TEST_VAL = 24
QUICK_TEST_EPOCHS = 1


@dataclass
class EchoRunConfig:
    frames_per_clip: int = FRAMES_PER_CLIP
    frame_size: int = FRAME_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    seed: int = SEED
    amp: bool = AMP

    phase1_epochs: int = PHASE1_EPOCHS
    phase1_lr: float = PHASE1_LR
    phase2_epochs: int = PHASE2_EPOCHS
    phase2_lr: float = PHASE2_LR
    weight_decay: float = WEIGHT_DECAY

    d_echo: int = D_ECHO
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    quick_test: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def apply_quick_test(self) -> "EchoRunConfig":
        self.quick_test = True
        self.phase1_epochs = QUICK_TEST_EPOCHS
        self.phase2_epochs = 0
        self.num_workers = min(self.num_workers, 2)
        return self
