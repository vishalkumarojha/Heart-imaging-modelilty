# Phase 3 — Cardiac MRI (ACDC diagnosis) — Frozen Reference

Status: **complete, accepted as a fusion encoder.** Standalone test macro OvR
AUROC **0.706** / accuracy **0.40** — weak, and expected: 70 training patients is
too small for a robust standalone 5-class classifier (flagged before training).
The `[B, 1024]` encoder still carries real signal (DCM/NOR/RV) and meets the
fusion contract. Read alongside `PHASE1_XRAY_SUMMARY.md` and
`PHASE2_ECHO_SUMMARY.md`. This is the **last encoder before the Phase 4 fusion layer**.

Task: classify one of 5 cardiac diagnoses from an ACDC short-axis cine MRI
(ED + ES frames). 5-class, single-label.

---

## 1. Saved artifacts

| Path | What it is |
|---|---|
| `outputs/checkpoints/mri/mri_resnet18_bilstm_best.pt` | **THE checkpoint.** Epoch 25, highest val macro AUROC (0.8222). 95 MB. |
| `outputs/checkpoints/mri/mri_resnet18_bilstm_last.pt` | Epoch 25 (same epoch — best landed on the last, a symptom of the noisy 15-patient val set). |
| `outputs/logs/mri/metrics.csv` | Per-epoch train/val loss + macro AUROC + accuracy + balanced acc + per-class AUROC (25 rows). |
| `outputs/logs/mri/test_predictions.csv` | 15 rows: `patient_id, group, true_class, split, pred_class, prob_0_DCM … prob_4_RV`. |
| `outputs/logs/mri/train_run.log` | Training stdout. |
| `data/processed/mri/split_index.csv` | **Frozen split** — `patient_id, group, label, split, ed_frame, es_frame, ed_path, es_path, dir` (100 rows). |
| `data/processed/mri/norm_stats.json` | Residual channel mean/std after per-volume normalisation (from 70 train patients). |

### `mri_resnet18_bilstm_best.pt` payload (dict)

```
epoch / global_epoch : 25
best_metric          : 0.82222...          # val macro OvR AUROC at save time
ckpt_metric          : "macro_auroc"
val_metrics          : {macro_auroc, per_class_auroc:{DCM,HCM,MINF,NOR,RV,mean},
                        accuracy, balanced_accuracy, loss}
model_state          : MriClassifier state_dict  (encoder.* + dropout + head.*)
optimizer_state      : Adam state (resume only)
scaler_state         : AMP GradScaler state (resume only)
class_names          : ["DCM", "HCM", "MINF", "NOR", "RV"]   # index == class id
n_slices             : 10
slice_hw             : 128
d_mri                : 1024
arch                 : "resnet18_bilstm"
norm_mean            : [-0.00462, -0.00444, -0.00018]   # channels (ED, ES, ED-ES)
norm_std             : [0.98547, 0.98492, 0.26493]
epochs               : 25
freeze_cnn           : true
```

Extract the reusable encoder (what Phase 4 / fusion calls):
```python
from src.mri_model import load_encoder_from_checkpoint
enc, ckpt = load_encoder_from_checkpoint("outputs/checkpoints/mri/mri_resnet18_bilstm_best.pt")
enc.eval()
# enc(vol[B,10,3,128,128]) -> [B, 1024]   (no head, no dropout, no activation)
```

---

## 2. Results

### Validation (best epoch 25, from `metrics.csv`)
| | macro | DCM | HCM | MINF | NOR | RV |
|---|---|---|---|---|---|---|
| val AUROC (OvR) | **0.8222** | 0.722 | 0.944 | 0.833 | 0.861 | 0.750 |
| val accuracy 0.533 · balanced accuracy 0.533 | | | | | | |

### Test — 15 held-out patients (`mri_evaluate.py`)
| Class | AUROC (OvR) | F1 | recall |
|---|---|---|---|
| DCM (dilated cardiomyopathy) | **0.889** | 0.571 | 2/3 |
| HCM (hypertrophic) | 0.667 | **0.000** | **0/3** |
| MINF (myocardial infarction) | **0.528** | **0.000** | **0/3** |
| NOR (normal) | 0.778 | 0.667 | 2/3 |
| RV (abnormal right ventricle) | 0.667 | 0.444 | 2/3 |
| **macro** | **0.7056** | 0.337 | |

overall accuracy **0.400** (6/15) · balanced accuracy 0.400

Confusion matrix (rows = true, cols = pred) `[DCM, HCM, MINF, NOR, RV]`:
```
DCM   [ 2  0  1  0  0 ]
HCM   [ 2  0  0  0  1 ]
MINF  [ 0  1  0  0  2 ]
NOR   [ 0  0  0  2  1 ]
RV    [ 0  0  0  1  2 ]
```

### Honest read — overfitting on 70 patients

`metrics.csv`: `train_loss` collapsed **1.74 → 0.20** while `val_loss` never
improved past ~epoch 10 (stuck 1.5–1.9). Even with the CNN frozen, the 4.2M-param
BiLSTM+head memorised the training set. Val macro AUROC 0.822 is optimistic and
noisy — 15 val patients (3/class) move it in ~0.05 steps, and "best" landing on
the final epoch is a tell. **Test macro AUROC 0.706 is the honest number**; the
val→test gap (0.82 → 0.71) confirms it.

- **DCM / NOR / RV** classify at 2/3 — dilated ventricle, normal morphology, and
  RV shape are genuinely separable (DCM test AUROC 0.89).
- **HCM 0/3, MINF 0/3.** HCM is confused with DCM (both large-heart); MINF
  (regional wall-motion) is not learnable from 14 training cases without
  segmentation priors.

### If standalone MRI needs to be stronger (not done here)
1. **5-fold CV** over all 100 patients (80 train/fold) — uses all data, gives an
   honest ± instead of one lucky/unlucky 15-patient test draw.
2. **Add the `_gt` segmentation masks** as input channels or an auxiliary
   segmentation loss — this is how ACDC-leaderboard methods reach ~95 %.
3. Accept it as a **fusion-only encoder** (current choice).

---

## 3. Data & split

- Source: ACDC challenge `training/`, `data/raw/acdc/training/` — 100 patient
  folders, 1.6 GB. Per patient: `Info.cfg`, `patientXXX_4d.nii.gz`,
  `patientXXX_frameNN.nii.gz` (ED & ES) + `_gt` masks.
- File naming: `patientXXX_frame{ED|ES:02d}.nii.gz` where ED/ES come from
  `Info.cfg` (`ED: 1` for 99 patients, `4` for one; `ES: 6…16`). Zero missing files.
- **Diagnosis distribution: 20 / 20 / 20 / 20 / 20** (DCM, HCM, MINF, NOR, RV) —
  balanced by design → **no class weights** (`USE_CLASS_WEIGHTS=False`).
- **No official split file** → created here: **stratified by `Group`, one patient
  per row, seed 42, 70/15/15**. Result: train 70 (14/class), val 15 (3/class),
  test 15 (3/class). Frozen in `split_index.csv`. (This is a genuine random
  stratified split — unlike Phase 1's patient-level grouping, which solved a
  multi-image-per-patient problem that does not exist here.)

### Label
`src/mri_config.py::GROUP_TO_IDX` is the single source of truth — alphabetical, fixed:
`0 DCM · 1 HCM · 2 MINF · 3 NOR · 4 RV`.

### Volume variability (handled)
| Axis | Range | Median |
|---|---|---|
| X (rows) | 154–428 | 216 |
| Y (cols) | 154–512 | 256 |
| Z (slices) | 6–18 | 9 |

Voxel spacing: in-plane 0.7–1.92 mm, through-plane 10 mm (86 patients) / 5 mm (12).
Raw intensity per-patient max ranges **184 → 4025** (≈20×) → per-volume
normalisation is mandatory (global stats are meaningless).

---

## 4. Preprocessing spec (exact — `src/mri_dataset.py`)

Volume IO: **`nibabel`**. Resampling: **`scipy.ndimage`** (`zoom`, `rotate`).

| Step | Detail |
|---|---|
| Load | ED volume + ES volume (`.nii.gz`) → float32 `(X, Y, Z)` |
| Per-volume normalise | clip to `[0.5, 99.5]` percentile, then z-score (`(v - mean) / (std + 1e-6)`) — done **per volume**, per patient |
| Slice sampling | `round(linspace(0, Z-1, 10))` → 10 evenly-spaced short-axis slices |
| In-plane resize | each slice → **128 × 128** (`scipy.ndimage.zoom`, order 1) |
| Channels | 3 = **`[ED_slice, ES_slice, ED_slice − ES_slice]`** — the difference channel encodes contraction (separates DCM / HCM / MINF / RV) |
| Augmentation (train only) | **shared geometry** across ED, ES and all slices: rotation ±10°, scale ±10% (zoom + centre crop/pad); intensity `v·(1±0.15) + (±0.15)`. **NO flips** — cardiac L-R anatomy (LV vs RV, situs); an L-R flip would relabel an RV-abnormal case. |
| Dataset residual norm | subtract `norm_mean` / divide `norm_std` from `norm_stats.json` (channels ≈ 0/0.99/0.26). **Not ImageNet / EchoNet stats.** |
| Output | `(tensor [10, 3, 128, 128] float32, label int ∈ 0..4)` |

Unreadable patients are skipped + logged (return next valid sample). `_predict` /
`evaluate_split` also guard against stray non-finite probs (→ uniform + warning)
so a GPU fluke can't crash metric computation.

---

## 5. Model contract for fusion (`src/mri_model.py`)

```
MriEncoder(pretrained=True)
  resnet18 (ResNet18_Weights.IMAGENET1K_V1), fc -> Identity   # per slice -> 512-d
  1-layer BiLSTM(input=512, hidden=512, bidirectional)        # over the SLICE axis -> [B, 10, 1024]
  mean pool over slices                                       # -> [B, 1024]
  forward(x: [B, N_SLICES, C, H, W]) -> [B, 1024]   # NO head, NO dropout, NO activation
  .feature_dim = 1024

MriClassifier(encoder, num_classes=5, dropout=0.3)   # THROW-AWAY training wrapper
  = encoder -> Dropout(0.3) -> Linear(1024, 5)       # logits, no softmax

freeze_cnn(model)          # freeze encoder.cnn; encoder.rnn + head train (used ALL run)
set_frozen_bn_eval(model)  # BN stays in eval during the frozen run
unfreeze_all(model)        # only via --unfreeze_cnn (not recommended for 70 patients)
build_mri_model(num_classes=5, pretrained=True) -> MriClassifier
load_encoder_from_checkpoint(path) -> (MriEncoder, ckpt_dict)   # strips "encoder." prefix
```

Params: 15.38M total, **4.21M** trainable (CNN frozen). Structurally identical to
`EchoEncoder` — the sequence axis is slice index instead of video time.

### Phase 4 — fusion (all three encoders are now built)

| Encoder | Input shape | Output | Standalone test |
|---|---|---|---|
| `XrayEncoder` (DenseNet121) | `[B, 3, 224, 224]` | `[B, 1024]` | mean AUROC 0.878 |
| `EchoEncoder` (ResNet18+BiLSTM) | `[B, 16, 3, 112, 112]` | `[B, 1024]` | macro AUROC 0.802 |
| `MriEncoder` (ResNet18+BiLSTM) | `[B, 10, 3, 128, 128]` | `[B, 1024]` | macro AUROC 0.706 |

1. Each encoder owns its own input shape; the fusion model only ever sees the
   three `[B, 1024]` vectors.
2. Fusion head input width = **1024 + 1024 + 1024 = 3072** →
   `concat([x_feat, echo_feat, mri_feat], dim=1)` → shared Linear/MLP → target head.
3. **The three datasets (NIH ChestX-ray14, EchoNet-Dynamic, ACDC) are mutually
   disjoint patient populations.** There is no shared-patient alignment anywhere
   in Phases 1–3. Real fusion needs a cohort where the same patient has multiple
   modalities; until then, the encoders are pre-trained independently and the
   fusion layer is trained/validated on whatever paired data becomes available
   (e.g. late-fusion on a small paired set, or each encoder frozen + a trainable
   fusion head).
4. `src/utils.py` reused unchanged across all three phases.

---

## 6. Training recipe (`src/mri_train.py`, `src/mri_config.py`)

Single phase — **CNN frozen the entire run** (70 training patients cannot
fine-tune 11M ResNet params). Overfitting mitigation:

| Lever | Setting |
|---|---|
| backbone | ResNet18 **frozen** (BN in eval); only BiLSTM + head train (4.21M) |
| optimizer | Adam, lr **3e-4**, **weight_decay 1e-3** (10× Phase 2) |
| head dropout | **0.3** |
| loss | `CrossEntropyLoss` (no weights — balanced dataset) |
| augmentation | rotation ±10°, scale ±10%, intensity jitter ±0.15, **no flips** |
| epochs | **25**, best-checkpoint on val macro OvR AUROC = de-facto early stopping |
| precision / batch | AMP, batch **8** (profiled: 0.22 GB peak — VRAM irrelevant) |
| seed | 42 |

`python -m src.mri_train` · `--resume` · `--quick_test` (12/8, 1 epoch) ·
`--batch_size N` · `--epochs N` · `--num_workers N` · `--no_amp` ·
`--unfreeze_cnn` (not recommended).

Full run: **25 epochs in ~75 s** on the GTX 1650.

---

## 7. Environment / profiling

- Same `.venv` as Phases 1–2, **+ `nibabel` 5.4.2** (only new dependency;
  `scipy` 1.18.1 already present provides `ndimage.zoom` / `rotate`).
- GPU: GTX 1650, 3.63 GB. Frozen model peak **0.22 GB** at batch 8 (0.31 GB at
  batch 16). Neither VRAM nor GPU compute is the bottleneck — CPU-side volume
  loading is. Throughput ~44 vol/s frozen.
- 25-epoch run ≈ 75 s; first epoch cold-reads 1.6 GB (fits in RAM after).
