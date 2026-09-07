# Phase 1 — Chest X-ray (Cardiomegaly / Effusion) — Frozen Reference

Status: **complete, accepted.** Test mean AUROC 0.878 (> val 0.868), best checkpoint
shows no overfitting. Use this doc when building the ECHO encoder + fusion layer.

---

## 1. Saved artifacts

| Path | What it is |
|---|---|
| `outputs/checkpoints/densenet121_best.pt` | **THE checkpoint.** Epoch 8, highest mean val AUROC (0.86780). |
| `outputs/checkpoints/densenet121_last.pt` | Epoch 10 (final). For `--resume` only — overfit, do not use for inference. |
| `outputs/logs/metrics.csv` | Per-epoch train_loss / val_loss / per-label + mean val AUROC (10 rows). |
| `outputs/logs/test_predictions.csv` | 15,884 rows: `Image Index, Patient ID, split, true_*, score_*, pred_*` per label. |
| `outputs/logs/train_run.log` | Full training stdout (tqdm-interleaved; `metrics.csv` is the clean source). |
| `outputs/logs/evaluate_run.log` | Test-set evaluation stdout + results table. |
| `data/processed/split_index.csv` | **Frozen split.** `Image Index, Patient ID, Finding Labels, Cardiomegaly, Effusion, path, split`. 109,312 rows. Regenerated only if deleted. |
| `data/processed/image_path_index.csv` | filename → absolute path lookup (cache). |

### `densenet121_best.pt` payload (dict)

```
epoch            : 8
global_epoch     : 8
phase            : "phase2-unfrozen"
phase_offset     : 3
best_auroc       : 0.8677973275491944          # mean val AUROC at save time
val_auroc        : {"Cardiomegaly": 0.87507, "Effusion": 0.86052, "mean": 0.86780}
model_state      : state_dict, 727 tensors     # MultiLabelClassifier (encoder.* + classifier.*)
optimizer_state  : Adam state (resume only)
scaler_state     : AMP GradScaler state (resume only)
target_labels    : ["Cardiomegaly", "Effusion"]   # index order == output logit order
image_size       : 224
arch             : "densenet121"
phase1_epochs    : 5
phase2_epochs    : 5
```

Load for inference:
```python
from src.model import build_model
import torch
ckpt = torch.load("outputs/checkpoints/densenet121_best.pt", map_location=dev)
model = build_model(num_classes=len(ckpt["target_labels"]), pretrained=False)
model.load_state_dict(ckpt["model_state"]); model.eval()
# logits -> torch.sigmoid() -> P(Cardiomegaly), P(Effusion)
```

---

## 2. Results

### Validation (best epoch 8, from `metrics.csv`)
| | mean | Cardiomegaly | Effusion |
|---|---|---|---|
| val AUROC | 0.8678 | 0.8751 | 0.8605 |

### Test — 15,884 held-out images, threshold 0.50 (`evaluate_run.log`)
| Label | AUROC | Sensitivity | Specificity | Precision | F1 | pos / neg |
|---|---|---|---|---|---|---|
| Cardiomegaly | **0.8972** | 0.6867 | 0.9153 | 0.1787 | 0.2836 | 415 / 15,469 |
| Effusion | **0.8587** | 0.7902 | 0.7711 | 0.3317 | 0.4673 | 1,997 / 13,887 |
| **mean AUROC** | **0.8779** | | | | | |

Low precision/F1 is a threshold-0.5 + `pos_weight` + class-imbalance artifact, not a
model defect. AUROC (threshold-free) is the accepted metric. Tune per-label thresholds
from `test_predictions.csv` if a deployable operating point is needed.

Phase 2 (full fine-tune) added only +0.005 AUROC over Phase 1's frozen best and epochs
9–10 overfit (val_loss 0.93 → 1.36). For any re-run: Phase 2 = 2–3 epochs, add
`weight_decay ≈ 1e-4`, or early-stop on mean val AUROC.

---

## 3. Data & split (must match for ECHO so patients align in fusion)

- Source: NIH ChestX-ray14, `data/raw/`. Images nested at `images_XXX/images/*.png`.
- 112,120 CSV rows; **109,312** images on disk (`images_012` partial download, 2,808
  missing — a contiguous tail block; patient-level split unaffected).
- **No official NIH split files present** → patient-level split created:
  - column: `Patient ID`; fractions **0.70 / 0.15 / 0.15**; `numpy.random.default_rng(42)`, shuffle patients.
  - **train 76,977 img / 20,797 patients · val 16,451 / 4,468 · test 15,884 / 4,455**
  - verified: **no Patient ID spans two splits**.
- Frozen in `data/processed/split_index.csv`. **The ECHO branch must reuse the same
  `Patient ID` → split mapping** (join on Patient ID) so a patient is never train in one
  modality and test in another once modalities are fused.

### Label extraction
`Finding Labels` (pipe-separated) → binary columns. `label_tensor = [Cardiomegaly, Effusion]`
as `float32`, order fixed by `config.TARGET_LABELS`. "No Finding" → both 0.

### Class balance (train)
| label | prevalence | `pos_weight` used (neg/pos) |
|---|---|---|
| Cardiomegaly | 2.44 % | **39.945** |
| Effusion | 11.69 % | **7.552** |

---

## 4. Preprocessing spec (exact — `src/dataset.py::build_transforms`)

Backend: **albumentations 2.0.8** + `ToTensorV2`. Image read with PIL, `.convert("RGB")`
(grayscale replicated to 3 channels), uint8 `H×W×3`.

| Step | Train | Val / Test |
|---|---|---|
| Resize | 224 × 224 | 224 × 224 |
| Rotate | `limit=±10°, border_mode=0 (constant), p=0.7` | — |
| RandomBrightnessContrast | `brightness_limit=0.15, contrast_limit=0.15, p=0.7` | — |
| **Horizontal flip** | **NEVER** (laterality is diagnostic) | **NEVER** |
| Normalize | mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`, `max_pixel_value=255` | same |
| ToTensorV2 | → `float32 [3,224,224]`, CHW | same |

ImageNet stats because the backbone is ImageNet-pretrained. **A non-RGB ECHO/MRI encoder
should compute its own channel stats — do not reuse these.**

---

## 5. Model contract for fusion (`src/model.py`)

```
XrayEncoder(pretrained=True)
  torchvision densenet121 (DenseNet121_Weights.IMAGENET1K_V1)
  .features  ->  ReLU  ->  AdaptiveAvgPool2d(1)  ->  flatten
  forward(x: [B,3,224,224]) -> [B, 1024]           # NO classifier, NO activation
  .feature_dim = 1024                              # attribute the wrapper reads

MultiLabelClassifier(encoder, num_classes, feature_dim=None)   # modality-agnostic
  = encoder  ->  nn.Linear(encoder.feature_dim, num_classes)   # logits, no sigmoid

freeze_backbone(model, train_last_block=True)  # encoder frozen except
                                               # features.denseblock4 + features.norm5;
                                               # classifier always trainable
unfreeze_all(model)
build_model(num_classes=2, pretrained=True) -> MultiLabelClassifier
```

**When you build `EchoEncoder`, match this contract:**
1. `forward(x) -> [B, D_echo]`, no head, no activation.
2. expose `.feature_dim = D_echo`.
3. keep all echo-specific loading/aug in an echo dataset module — `MultiLabelImageDataset`
   in `src/dataset.py` is already modality-agnostic (`path` column + label columns +
   an albumentations transform); reuse or mirror it, don't fork X-ray logic into it.
4. Fusion later: `concat([x_feat, echo_feat, mri_feat], dim=1) -> shared Linear/MLP head`.
   With DenseNet121's 1024-d, plan the fusion head input as `1024 + D_echo + D_mri`.

`src/utils.py` (seeding, `multilabel_auroc`, `binary_rates_from_confusion`,
`log_label_distribution`) is fully modality-agnostic — reuse as-is for ECHO.

---

## 6. Training recipe (`src/train.py`, `src/config.py`)

| | Phase 1 | Phase 2 |
|---|---|---|
| epochs | 5 | 5 |
| backbone | frozen (denseblock4 + norm5 + classifier trainable; 2,162,178 params) | fully unfrozen (6,955,906 params) |
| optimizer | Adam, lr **1e-4**, weight_decay 0 | Adam, lr **1e-5**, weight_decay 0 |
| loss | `BCEWithLogitsLoss(pos_weight=[39.945, 7.552])` | same |
| precision | AMP (`torch.amp`, `GradScaler("cuda")`) | same |
| batch | 32 | 32 |
| checkpoint | save `best.pt` when **mean val AUROC** improves (any epoch, any phase); `last.pt` every epoch | same |
| seed | 42 (`src/utils.py::set_seed`, cudnn.benchmark on) | same |

`python -m src.train` · `--resume` (from `last.pt`) · `--quick_test` (500/100, 1 epoch) ·
`--batch_size N` · `--num_workers N` · `--no_amp`.

---

## 7. Environment

- GPU: NVIDIA GeForce GTX 1650, **3.63 GB usable VRAM**, CC 7.5. Peak use: ~1.4 GB
  (unfrozen, batch 32, AMP) — batch 32 is safe.
- Throughput: ~85 img/s frozen, ~23 img/s unfrozen. Full run (10 epochs) ≈ 5 h 20 m.
- `.venv/` (Python 3.14.7): torch 2.14.0+cu130, torchvision 0.29.0+cu130,
  albumentations 2.0.8, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.3.
- Python 3.14 defaults DataLoader multiprocessing to **forkserver** — entrypoints must
  stay under `if __name__ == "__main__":` (a helper without the guard will re-spawn).
