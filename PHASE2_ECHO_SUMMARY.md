# Phase 2 — Echocardiogram (EF category) — Frozen Reference

Status: **complete, accepted.** Test macro OvR AUROC **0.802** (val 0.820), encoder
verified fusion-ready. Read alongside `PHASE1_XRAY_SUMMARY.md`. Use both when
building the Phase 3 (MRI) encoder + the fusion model.

Task: classify **Ejection Fraction category** from an EchoNet-Dynamic apical-4-chamber
video. 3-class, single-label (not multi-label like Phase 1).

---

## 1. Saved artifacts

| Path | What it is |
|---|---|
| `outputs/checkpoints/echo/echo_cnn_lstm_best.pt` | **THE checkpoint.** Epoch 8 (Phase 2 e5), highest val macro AUROC (0.8195). 185 MB. |
| `outputs/checkpoints/echo/echo_cnn_lstm_last.pt` | Epoch 9 (final). `--resume` only. |
| `outputs/logs/echo/metrics.csv` | Per-epoch train/val loss + macro AUROC + accuracy + balanced acc + per-class AUROC (9 rows). |
| `outputs/logs/echo/test_predictions.csv` | 1,277 rows: `FileName, split, EF, true_class, pred_class, prob_0_Reduced, prob_1_MildlyReduced, prob_2_Normal`. |
| `outputs/logs/echo/train_run.log` | Training stdout (tqdm-interleaved; `metrics.csv` is the clean source). |
| `data/processed/echo/norm_stats.json` | Channel mean/std computed from 250 TRAIN videos (**not** ImageNet). |
| `data/processed/echo/echo_index.csv` | Master table: `FileName, path, EF, ef_bucket, Split, FPS, NumberOfFrames` (10,030 rows). |

### `echo_cnn_lstm_best.pt` payload (dict)

```
epoch / global_epoch : 8
phase                : "phase2-unfrozen"
phase_offset         : 5
best_metric          : 0.8194617...            # val macro OvR AUROC at save time
ckpt_metric          : "macro_auroc"
val_metrics          : {macro_auroc, per_class_auroc:{Reduced,Mildly Reduced,Normal,mean},
                        accuracy, balanced_accuracy, loss}
model_state          : EchoClassifier state_dict  (encoder.* + head.*)
optimizer_state      : Adam state (resume only)
scaler_state         : AMP GradScaler state (resume only)
class_names          : ["Reduced", "Mildly Reduced", "Normal"]   # index == class id
frames_per_clip      : 16
frame_size           : 112
d_echo               : 1024
arch                 : "resnet18_bilstm"
norm_mean            : [0.12876, 0.12891, 0.12935]
norm_std             : [0.19706, 0.19706, 0.19739]
phase1_epochs / phase2_epochs : 3 / 6
```

Extract the reusable encoder (what Phase 3 / fusion calls):
```python
from src.echo_model import load_encoder_from_checkpoint
enc, ckpt = load_encoder_from_checkpoint("outputs/checkpoints/echo/echo_cnn_lstm_best.pt")
enc.eval()
# enc(video[B,16,3,112,112]) -> [B, 1024]   (no head, no activation)
```

---

## 2. Results

### Validation (best epoch 8, from `metrics.csv`)
| | macro | Reduced | Mildly Reduced | Normal |
|---|---|---|---|---|
| val AUROC (OvR) | **0.8195** | 0.9220 | 0.6997 | 0.8368 |
| val accuracy 0.738 · balanced accuracy 0.578 | | | | |

### Test — 1,277 held-out videos (`echo_evaluate.py`)
| Class | AUROC (OvR) | F1 | Support |
|---|---|---|---|
| Reduced (EF < 40) | **0.9062** | 0.552 | 160 |
| Mildly Reduced (40–54) | **0.6781** | 0.326 | 241 |
| Normal (EF ≥ 55) | **0.8222** | 0.841 | 876 |
| **macro** | **0.8022** | 0.573 | 1,277 |

overall accuracy 0.713 · balanced accuracy 0.549

Confusion matrix (rows = true, cols = pred):
```
                 → Reduced   → Mildly   → Normal
Reduced               72        60         28
Mildly Reduced        21        80        140
Normal                 8       110        758
```

- **Reduced** ranks well (AUROC 0.91); argmax recall is 45% but only 28/160 severe
  cases are called fully Normal — the adjacent-bucket error dominates.
- **Mildly Reduced** is the weak class (0.68) — EF 40–54 is a 15-pt band and
  echo-derived EF has ~±5% measurement noise, so boundary cases are genuinely
  ambiguous. Bleeds mostly into Normal.
- Mild overfitting began by epoch 9 (train_loss 0.63 vs val_loss flat ~0.80);
  val macro AUROC plateaued ~0.81–0.82 over the last 3 epochs. The 3+6 schedule
  (trimmed from 3+12) was about right. Best-checkpoint kept epoch 8.

---

## 3. Data & split

- Source: EchoNet-Dynamic, `data/raw/echonet/EchoNet-Dynamic/` (`Videos/*.avi` +
  `FileList.csv`). 7.4 GB, fits in page cache after epoch 1.
- **10,030 videos == 10,030 FileList.csv rows**, 1:1, zero missing/orphan.
  Decoded frame counts match `NumberOfFrames`. All frames decode to 112×112×3
  (6 CSV rows have wrong dim metadata; actual frames fine; loader resizes anyway).
- **Official `Split` column used directly** — one video per patient, so no leakage
  risk and no custom split (unlike Phase 1's patient-level split).

| Split | n | Reduced | Mildly Reduced | Normal |
|---|---|---|---|---|
| TRAIN | 7,465 | 948 (12.7%) | 1,333 (17.9%) | 5,184 (69.4%) |
| VAL | 1,288 | 156 (12.1%) | 231 (17.9%) | 901 (70.0%) |
| TEST | 1,277 | 160 (12.5%) | 241 (18.9%) | 876 (68.6%) |

EF: min 6.9, max 97.0, mean 55.7, no NaN. Imbalance ~5.5:1 (Normal:Reduced) —
far milder than Phase 1's 40:1, near-identical across splits.

### Label
`src/echo_config.py::bucket_ef(ef)` is the single source of truth:
`EF < 40 → 0 Reduced` · `40 ≤ EF < 55 → 1 Mildly Reduced` · `EF ≥ 55 → 2 Normal`.
Ascending index = ascending heart function.

### Class-imbalance handling
`CrossEntropyLoss(weight = inverse train frequency)` = **[2.625, 1.867, 0.48]** for
[Reduced, Mildly, Normal]. (Phase 1 used per-label `pos_weight` on BCE; different
because this task is single-label.)

---

## 4. Preprocessing spec (exact — `src/echo_dataset.py`)

Video decode: **`cv2.VideoCapture`** (`torchvision.io.read_video` was removed in
torchvision 0.29). `cv2.setNumThreads(0)` — DataLoader workers provide parallelism.

| Step | Detail |
|---|---|
| Frame sampling | **16** evenly-spaced indices: `round(linspace(0, n_total-1, 16))`. Sequential decode, stop after the last needed index. Gaps (short/failed decode) filled with nearest grabbed frame. |
| Color | cv2 BGR → RGB |
| Resize | 112 × 112 (defensive; videos are natively 112×112) |
| Augmentation (train only) | **clip-consistent** via `A.ReplayCompose` — one random parametrisation per clip, replayed on all 16 frames: `Rotate(±10°, border_mode=0, p=0.7)`, `RandomBrightnessContrast(0.15, 0.15, p=0.7)`. |
| **Horizontal flip** | **OFF** (`AUG_HFLIP=False`). Apical-4-chamber echo has a conventional acquisition orientation; a mirror image is a view never clinically acquired. Toggle in `echo_config.py` if we want to A/B it. |
| Normalize | mean `[0.12876, 0.12891, 0.12935]`, std `[0.19706, 0.19706, 0.19739]`, `max_pixel_value=255`. **Computed from 250 TRAIN videos — NOT ImageNet** (per Phase 1 summary's directive). Cached to `norm_stats.json`; checkpoint also carries them. |
| Output | `(video_tensor [T=16, C=3, H=112, W=112] float32, label int ∈ {0,1,2})` |

Unreadable videos are skipped + logged (return next valid sample), same policy as Phase 1.

---

## 5. Model contract for fusion (`src/echo_model.py`)

```
EchoEncoder(pretrained=True)
  resnet18 (ResNet18_Weights.IMAGENET1K_V1), fc -> Identity     # per frame -> 512-d
  1-layer BiLSTM(input=512, hidden=512, bidirectional)          # -> [B, T, 1024]
  temporal MEAN pool over T                                     # -> [B, 1024]
  forward(x: [B, T, C, H, W]) -> [B, 1024]        # NO classifier, NO activation
  .feature_dim = 1024

EchoClassifier(encoder, num_classes=3)   # THROW-AWAY training wrapper
  = encoder -> nn.Linear(1024, 3)        # logits, no softmax

freeze_cnn(model)        # freeze encoder.cnn; encoder.rnn + head stay trainable
set_frozen_bn_eval(model)  # call after model.train() in the frozen phase (BN stats)
unfreeze_all(model)
build_echo_model(num_classes=3, pretrained=True) -> EchoClassifier
load_encoder_from_checkpoint(path) -> (EchoEncoder, ckpt_dict)   # strips "encoder." prefix
```

Params: 15.38M total, **4.21M** trainable when CNN-frozen.

Temporal aggregation is **mean-pool over the BiLSTM outputs**, not the last hidden
state — robust to where in the cardiac cycle the sampled clip starts.

### For Phase 3 (MRI) + the fusion model
1. Build `MriEncoder` to the same contract: `forward(x) -> [B, D_mri]`, no head, no
   activation, expose `.feature_dim`.
2. **Each encoder owns its own input shape** — X-ray `[B,3,224,224]`, echo
   `[B,16,3,112,112]`, MRI whatever fits. Fusion only ever sees the `[B, 1024]` /
   `[B, D]` outputs.
3. Fusion head input width = `1024 (xray) + 1024 (echo) + D_mri`, then a shared
   Linear/MLP → target head(s). `concat([...], dim=1)`.
4. **NIH ChestX-ray14 and EchoNet-Dynamic are disjoint patient populations** —
   there is no shared-patient alignment between Phase 1 and Phase 2. The "reuse the
   Patient ID → split mapping" note in the Phase 1 summary applies only *within* a
   single dataset. True multimodal fusion needs a cohort where the same patient has
   multiple modalities; until such data exists, encoders are trained independently
   and the fusion layer is validated on whatever paired cohort becomes available.
5. `src/utils.py` reused unchanged — `multilabel_auroc(one_hot(y), softmax_probs,
   class_names)` gives per-class OvR AUROC + macro mean.

---

## 6. Training recipe (`src/echo_train.py`, `src/echo_config.py`)

| | Phase 1 | Phase 2 |
|---|---|---|
| epochs | 3 | 6 (trimmed from 12 — EchoNet train set is 10× smaller than Phase 1's) |
| backbone | CNN frozen; BiLSTM + head trainable (4.21M params) | fully unfrozen (15.38M) |
| optimizer | Adam, lr **1e-4**, weight_decay **1e-4** | Adam, lr **1e-5**, weight_decay **1e-4** |
| loss | `CrossEntropyLoss(weight=[2.625, 1.867, 0.48])` | same |
| precision | AMP (`torch.amp`, `GradScaler("cuda")`) | same |
| batch / frames | 8 / 16 | 8 / 16 |
| checkpoint | `best.pt` on val **macro OvR AUROC**; `last.pt` every epoch | same |
| seed | 42 (`src/utils.py::set_seed`) | same |

`python -m src.echo_train` · `--resume` · `--quick_test` (60/24, 1 epoch) ·
`--batch_size N` · `--frames N` · `--num_workers N` · `--no_amp`.

Checkpoint-selection metric is **macro OvR AUROC**, not accuracy: 69% Normal means
"always predict Normal" scores 0.69 accuracy, which rewards ignoring the two
clinically important minority classes. Macro AUROC weights all 3 equally and is
threshold-free (Phase 1's AUROC-first philosophy). Accuracy + balanced accuracy +
3×3 confusion matrix are still reported.

---

## 7. Environment / profiling

- Same `.venv` as Phase 1 (Python 3.14.7, torch 2.14.0+cu130). No new install —
  `cv2` (opencv-python-headless 5.0.0) was already pulled in by albumentations.
- GPU: GTX 1650, 3.63 GB. **Compute-bound, not memory-bound**: peak 0.67 GB at
  batch 8 / 16 frames unfrozen (1.88 GB even at batch 16 / 32 frames).
- Throughput: ~56 clip/s frozen, ~17.6 clip/s unfrozen. Full 3+6 run ≈ **47 min**.
- More frames or larger batches only slow it down on this GPU (13.4 clip/s at
  batch 16 vs 17.6 at batch 8). 16 frames @ 50 fps median spans ~3+ heartbeats.
