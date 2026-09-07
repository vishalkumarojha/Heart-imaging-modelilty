"""Phase 4 (fusion) configuration.

Ties together the three frozen encoders from Phases 1-3. Import-light.

=============================================================================
 THERE IS NO REAL TRI-MODAL PATIENT DATA.
 NIH ChestX-ray14, EchoNet-Dynamic and ACDC are three different institutions
 with non-overlapping patients. The fusion layer is therefore:
   * TRAINED / VALIDATED only in single-modality-present mode, on each
     modality's own real test set (the other two marked missing);
   * DEMONSTRATED on synthetic cross-dataset embedding combinations that are
     labelled "SYNTHETIC" everywhere and are NOT real multi-modal performance.
=============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- frozen encoder checkpoints (Phases 1-3) --------------------------------
XRAY_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "densenet121_best.pt"
ECHO_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "echo" / "echo_cnn_lstm_best.pt"
MRI_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "mri" / "mri_resnet18_bilstm_best.pt"

MODALITIES: List[str] = ["xray", "echo", "mri"]        # fixed order == mask/concat order
EMBED_DIM = 1024                                       # every encoder -> [B, 1024]

# raw input shapes (per sample, no batch dim) — used by the contract check + demo
INPUT_SHAPES: Dict[str, Tuple[int, ...]] = {
    "xray": (3, 224, 224),
    "echo": (16, 3, 112, 112),
    "mri": (10, 3, 128, 128),
}

# --- outputs ---------------------------------------------------------------
FUSION_CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "fusion"
FUSION_LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "fusion"
METRICS_CSV = FUSION_LOG_DIR / "metrics.csv"
CKPT_BEST = FUSION_CKPT_DIR / "fusion_best.pt"
CKPT_LAST = FUSION_CKPT_DIR / "fusion_last.pt"
SINGLE_MODALITY_RESULTS_CSV = FUSION_LOG_DIR / "single_modality_validation.csv"
SYNTHETIC_DEMO_CSV = FUSION_LOG_DIR / "SYNTHETIC_demo_predictions.csv"

for _d in (FUSION_CKPT_DIR, FUSION_LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --- unified output space --------------------------------------------------
# CONFIRMED: scheme (b) — task-specific heads on a SHARED fused representation.
# Each dataset keeps its own real label space (nothing invented). Single-modality
# validation is then apples-to-apples with Phases 1-3. A "risk level" readout is
# only DERIVED post-hoc from the head outputs in the demo (heuristic, not trained).
UNIFIED_SCHEME: str = "multi_head"

TASK_SPECS: Dict[str, dict] = {
    "xray": {"type": "multilabel", "n_out": 2, "loss": "bce",
             "classes": ["Cardiomegaly", "Effusion"],
             "primary": "mean_auroc", "standalone_test": 0.878},
    "echo": {"type": "multiclass", "n_out": 3, "loss": "ce",
             "classes": ["Reduced", "Mildly Reduced", "Normal"],
             "primary": "macro_auroc", "standalone_test": 0.802},
    "mri": {"type": "multiclass", "n_out": 5, "loss": "ce",
            "classes": ["DCM", "HCM", "MINF", "NOR", "RV"],
            "primary": "macro_auroc", "standalone_test": 0.706},
}

# --- embedding cache ------------------------------------------------------
EMB_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XRAY_TRAIN_CAP = 20_000   # cap X-ray TRAIN embeddings (76,977 full -> ~20 min DenseNet pass)
XRAY_VAL_CAP = 8_000      # cap X-ray VAL too (signal only); TEST is always full & uncapped

# --- fusion hyperparameters ---------------------------------------------
FUSION_HIDDEN = 512
FUSION_OUT_DIM = 512      # width of the shared fused representation the heads read
FUSION_DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
BATCH_SIZE = 256          # embeddings are tiny [1024] vectors
AMP = False              # fusion head is trivial; fp32 is fine and simpler


@dataclass
class FusionRunConfig:
    seed: int = SEED
    embed_dim: int = EMBED_DIM
    modalities: List[str] = field(default_factory=lambda: list(MODALITIES))
    hidden: int = FUSION_HIDDEN
    dropout: float = FUSION_DROPOUT
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    amp: bool = AMP
    quick_test: bool = False

    def apply_quick_test(self) -> "FusionRunConfig":
        self.quick_test = True
        self.epochs = 1
        return self
