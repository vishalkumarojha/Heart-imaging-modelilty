"""Phase 3 (ACDC cardiac MRI) configuration — 5-class diagnosis classification.

Mirrors src/config.py (Phase 1) and src/echo_config.py (Phase 2). Import-light.

Label contract (documented, fixed) — alphabetical, stable:
    0 DCM   dilated cardiomyopathy
    1 HCM   hypertrophic cardiomyopathy
    2 MINF  myocardial infarction (prior MI, reduced LV EF)
    3 NOR   normal
    4 RV    abnormal right ventricle
`GROUP_TO_IDX` is the single source of truth.

Input strategy (see mri_explore.py notes): 2D-slice-sequence, NOT a 3D-CNN.
  * ACDC volumes have only 6-18 slices (median 9) and 100 patients total -> a
    from-scratch 3D-CNN would overfit hard. A 2D ImageNet-pretrained backbone
    per slice + a sequence aggregator over the slice axis reuses the proven
    Phase 2 architecture and its pretrained features.
  * Each slice is turned into a 3-channel image [ED, ES, ED-ES]: the two key
    cardiac phases plus their difference (which directly encodes contraction —
    the signal separating DCM / HCM / MINF / RV).
  * Fixed input: N_SLICES evenly-sampled slices, resized to SLICE_HW.
    Tensor shape: [B, N_SLICES, 3, SLICE_HW, SLICE_HW]  (cf. echo [B,16,3,112,112]).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACDC_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "acdc" / "training"

MRI_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "mri"
SPLIT_CSV = MRI_PROCESSED_DIR / "split_index.csv"
NORM_STATS_JSON = MRI_PROCESSED_DIR / "norm_stats.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MRI_CKPT_DIR = OUTPUTS_DIR / "checkpoints" / "mri"
MRI_LOG_DIR = OUTPUTS_DIR / "logs" / "mri"
METRICS_CSV = MRI_LOG_DIR / "metrics.csv"
PREDICTIONS_CSV = MRI_LOG_DIR / "test_predictions.csv"
CKPT_BEST = MRI_CKPT_DIR / "mri_resnet18_bilstm_best.pt"
CKPT_LAST = MRI_CKPT_DIR / "mri_resnet18_bilstm_last.pt"

for _d in (MRI_PROCESSED_DIR, MRI_CKPT_DIR, MRI_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
CLASS_NAMES: List[str] = ["DCM", "HCM", "MINF", "NOR", "RV"]
GROUP_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- #
# Volume preprocessing
# --------------------------------------------------------------------------- #
N_SLICES = 10          # evenly-sampled short-axis slices (median in data is 9-10)
SLICE_HW = 128         # in-plane resize (median native ~216x256 @ ~1.5 mm)
IN_CHANNELS = 3        # [ED, ES, ED-ES] per slice

# per-volume robust normalisation before any dataset-level stat
CLIP_PCT_LOW = 0.5
CLIP_PCT_HIGH = 99.5

# Augmentation (train only). NO flips: cardiac MRI has real left-right anatomy
# (LV vs RV, situs) — an L-R flip would relabel an RV-abnormal case. Vertical
# flip breaks anatomy too.
AUG_ROTATION_DEG = 10
AUG_INTENSITY_JITTER = 0.15   # brightness/contrast style, post-normalisation
AUG_SCALE = 0.1               # +/-10% zoom
AUG_FLIP = False

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
D_MRI = 1024               # fusion contract: MriEncoder.forward -> [B, 1024]
CNN_BACKBONE = "resnet18"
CNN_FEATURE_DIM = 512
RNN_HIDDEN = 512           # bidirectional -> 1024 = D_MRI
RNN_LAYERS = 1
RNN_BIDIRECTIONAL = True
RNN_DROPOUT = 0.3          # only applies with >1 layer; kept for reference
HEAD_DROPOUT = 0.3         # dropout before the classifier head (overfitting guard)

# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # stratified by Group, one row per patient
SEED = 42

# --------------------------------------------------------------------------- #
# Training  (tiny dataset -> aggressive overfitting mitigation)
# --------------------------------------------------------------------------- #
# CNN stays FROZEN the whole run: 70 training patients cannot fine-tune 11M
# ResNet params. Only the BiLSTM + head (~4M) train. Best-checkpoint on val
# macro AUROC acts as early stopping.
FREEZE_CNN = True
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-3
AMP = True
BATCH_SIZE = 8             # profiled: frozen peak 0.22 GB / 3.63; VRAM is a non-issue, GPU is idle-bound
NUM_WORKERS = 4

USE_CLASS_WEIGHTS = False  # ACDC is 20/20/20/20/20 by design; verified in explore
CKPT_METRIC = "macro_auroc"

QUICK_TEST_TRAIN = 12
QUICK_TEST_VAL = 8
QUICK_TEST_EPOCHS = 1


@dataclass
class MriRunConfig:
    n_slices: int = N_SLICES
    slice_hw: int = SLICE_HW
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    seed: int = SEED
    amp: bool = AMP

    epochs: int = EPOCHS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    freeze_cnn: bool = FREEZE_CNN

    d_mri: int = D_MRI
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    quick_test: bool = False

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def apply_quick_test(self) -> "MriRunConfig":
        self.quick_test = True
        self.epochs = QUICK_TEST_EPOCHS
        self.num_workers = min(self.num_workers, 2)
        return self
