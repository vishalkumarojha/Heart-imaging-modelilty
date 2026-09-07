# Multi-Modality Heart-Condition Detection

Independently trained per-modality encoders — **chest X-ray**, **echocardiogram
video**, **cardiac MRI** — each producing a common **1024-d embedding**, built so
they can be composed into a single late-fusion cardiac-diagnosis model.

> **Status:** Phases 1–3 (the three encoders) complete and evaluated. Phase 4
> (fusion) not started. Per-phase deep-dive references:
> [`PHASE1_XRAY_SUMMARY.md`](PHASE1_XRAY_SUMMARY.md) ·
> [`PHASE2_ECHO_SUMMARY.md`](PHASE2_ECHO_SUMMARY.md) ·
> [`PHASE3_MRI_SUMMARY.md`](PHASE3_MRI_SUMMARY.md)

---

## 1. The idea

Cardiac disease is assessed with several imaging modalities that each see
something different:

| Modality | Sees | Cheap / available? |
|---|---|---|
| Chest X-ray | gross cardiac silhouette, pleural fluid | very |
| Echocardiogram | wall motion, chamber volumes, ejection fraction | moderately |
| Cardiac MRI | tissue characterisation, precise volumes, RV | least |

A model that fuses all three should be more robust than any single one — and
degrade gracefully when a modality is missing. The blocker is **data**: there is
no large public cohort where the *same patient* has all three. So this project
takes the tractable route:

1. **Train one encoder per modality** on that modality's best available public
   dataset, as an ordinary supervised classifier.
2. Enforce a **shared output contract** — every encoder maps its input to
   `[B, 1024]`, no classifier head, no final activation — so the encoders are
   drop-in interchangeable.
3. **Phase 4:** discard the per-modality heads, concatenate the three 1024-d
   embeddings (`→ 3072`), and train a small fusion head on whatever paired data
   is available (or with the encoders frozen).

This repo is Phases 1–3. Each phase followed the **same staged process**
(§5) so the write-up is reproducible.

---

## 2. Results at a glance

All numbers are on each phase's **held-out test split**, from `python -m
src.<phase>_evaluate`. **Primary metric is AUROC** (per-label for X-ray;
macro one-vs-rest for the multi-class phases) — accuracy is reported but is
misleading under class imbalance.

| Phase | Modality / dataset | Task | Test set | **Test AUROC** | Accuracy | Notes |
|---|---|---|---|---|---|---|
| **1** | Chest X-ray — NIH ChestX-ray14 | 2-label (Cardiomegaly, Effusion) | 15,884 images | **0.878** mean | — | strong, generalises (test > val) |
| **2** | Echo video — EchoNet-Dynamic | 3-class EF category | 1,277 videos | **0.802** macro | 0.713 | middle EF class is hard (0.68) |
| **3** | Cardiac MRI — ACDC | 5-class diagnosis | 15 patients | **0.706** macro | 0.400 | small-data limited — see §4.3 |

### Per-class detail

**Phase 1 — X-ray** (threshold 0.5)

| Label | AUROC | Sensitivity | Specificity | Precision | F1 | Prevalence (test) |
|---|---|---|---|---|---|---|
| Cardiomegaly | 0.897 | 0.687 | 0.915 | 0.179 | 0.284 | 2.6 % |
| Effusion | 0.859 | 0.790 | 0.771 | 0.332 | 0.467 | 12.6 % |
| **mean** | **0.878** | | | | | |

Low precision is a threshold-0.5 + `pos_weight` + rare-positive artefact; AUROC is the accepted metric.

**Phase 2 — Echo** (macro F1 0.573 · balanced acc 0.549)

| Class (EF) | AUROC | F1 | Confusion (row = true) → Reduced / Mildly / Normal |
|---|---|---|---|
| Reduced (< 40) | 0.906 | 0.552 | 72 / 60 / 28 |
| Mildly Reduced (40–54) | 0.678 | 0.326 | 21 / 80 / 140 |
| Normal (≥ 55) | 0.822 | 0.841 | 8 / 110 / 758 |

Only 28 / 160 severe cases are called fully Normal — the dangerous error is rare; most confusion is with the adjacent bucket.

**Phase 3 — MRI** (macro F1 0.337 · balanced acc 0.400)

| Class | AUROC | F1 | Test recall |
|---|---|---|---|
| DCM (dilated) | 0.889 | 0.571 | 2/3 |
| HCM (hypertrophic) | 0.667 | 0.000 | 0/3 |
| MINF (infarction) | 0.528 | 0.000 | 0/3 |
| NOR (normal) | 0.778 | 0.667 | 2/3 |
| RV (RV abnormal) | 0.667 | 0.444 | 2/3 |

Confusion matrix (rows = true, cols = pred `[DCM, HCM, MINF, NOR, RV]`):
```
DCM  [ 2 0 1 0 0 ]    HCM  [ 2 0 0 0 1 ]    MINF [ 0 1 0 0 2 ]
NOR  [ 0 0 0 2 1 ]    RV   [ 0 0 0 1 2 ]
```

---

## 3. The three encoders

Every encoder satisfies: **`forward(x) → [B, 1024]`, no head, no activation,
`.feature_dim = 1024`**, and exposes `load_encoder_from_checkpoint(path)` that
rebuilds *just* the encoder (strips the training head).

| | Phase 1 — `XrayEncoder` | Phase 2 — `EchoEncoder` | Phase 3 — `MriEncoder` |
|---|---|---|---|
| Backbone | DenseNet121 (ImageNet) | ResNet18 (ImageNet) per frame | ResNet18 (ImageNet) per slice |
| Temporal / depth agg | — (single image) | 1-layer BiLSTM over 16 frames → mean-pool | 1-layer BiLSTM over 10 slices → mean-pool |
| Input tensor | `[B, 3, 224, 224]` | `[B, 16, 3, 112, 112]` | `[B, 10, 3, 128, 128]` |
| Input channels | RGB (grey ×3) | RGB frame | `[ED, ES, ED−ES]` per slice |
| Output | `[B, 1024]` | `[B, 1024]` | `[B, 1024]` |
| Params (total / trainable) | 6.96 M / 2.16 M frozen | 15.4 M / 4.2 M frozen | 15.4 M / 4.2 M frozen |
| Checkpoint | `outputs/checkpoints/densenet121_best.pt` | `outputs/checkpoints/echo/echo_cnn_lstm_best.pt` | `outputs/checkpoints/mri/mri_resnet18_bilstm_best.pt` |

**Phase 4 fusion (planned):** `concat([x_feat, echo_feat, mri_feat], dim=1)` →
`Linear(3072, …)` → shared diagnosis head. Because the three source datasets are
**disjoint patient populations** (§4.1), fusion will use either a small paired
cohort or frozen encoders + a trainable head only.

---

## 4. Datasets, methods, and honest limitations

### 4.1 Datasets

| Phase | Dataset | Size | Split | Label source |
|---|---|---|---|---|
| 1 | NIH ChestX-ray14 | 112,120 rows / **109,312 images on disk** (`images_012` partial, 2,808 missing ≈ 2.5 %) | **patient-level** 70/15/15, seed 42 (no official split files present) — no `Patient ID` spans splits | `Finding Labels` → binary Cardiomegaly / Effusion |
| 2 | EchoNet-Dynamic | 10,030 videos (1 per patient) | **official** `Split` column: 7,465 / 1,288 / 1,277 | `EF` (%) → 3 buckets at 40 and 55 |
| 3 | ACDC (training) | 100 patients, **20 per class** | **stratified by diagnosis**, seed 42, 70/15/15 → 70 / 15 / 15 (no official split) | `Info.cfg` `Group:` → 5 classes |

> **The three datasets share no patients.** There is no cross-phase patient
> alignment; "reuse the split" only ever applies *within* a phase. This is the
> central constraint on Phase 4.

Every split is frozen to a CSV (`data/processed/{split_index.csv,
echo/echo_index.csv, mri/split_index.csv}`) and reproducible from seed 42.

### 4.2 Method choices common to all phases

- **AUROC-first.** Multi-class phases select the checkpoint on **macro
  one-vs-rest AUROC**, not accuracy — a "always predict the majority class"
  baseline scores 0.69 (echo) / 0.20 (MRI) accuracy while being useless.
- **No flips, any modality.** X-ray laterality, echo apical-4-chamber
  orientation, and cardiac situs (LV vs RV) are all diagnostic — an L-R flip can
  relabel a case. Augmentation is limited to rotation ±10°, mild
  intensity/brightness–contrast jitter, and (MRI) ±10 % scale.
- **Per-modality normalisation.** ImageNet stats for the ImageNet-pretrained
  X-ray backbone; **computed** channel stats for echo; **per-volume** robust
  normalisation (clip [0.5, 99.5] pct, z-score) for MRI because raw MRI intensity
  varies ~20× between patients. No modality reuses another's stats.
- **Class imbalance** handled at the loss: `BCEWithLogitsLoss(pos_weight=…)`
  (X-ray, ratios 40× / 7.6×), `CrossEntropyLoss(weight=inverse-freq)` (echo,
  ~5.5× / 3.7× / 1×), none for MRI (balanced by design).
- **Two-tier checkpointing.** `*_best.pt` (metric-gated) + `*_last.pt` (every
  epoch, atomic write) + `--resume` (restores model + optimizer + AMP scaler +
  epoch counter; `metrics.csv` appended).
- **Mixed precision** (`torch.amp`) throughout. `src/utils.py` (seeding, logging,
  `multilabel_auroc`, confusion-rate helpers) is shared **unchanged** across all
  three phases.

### 4.3 Limitations (for the paper's "Threats to validity")

1. **MRI standalone is weak** (test macro AUROC 0.706, accuracy 6/15). 70
   training patients cannot support a robust 5-class classifier; `metrics.csv`
   shows clear overfitting (train loss 1.74 → 0.20, val loss flat after ~epoch
   10). HCM and MINF get 0 test recall. The encoder still carries real signal
   (DCM test AUROC 0.89) and is kept as a **fusion-only** component. Stronger
   options not pursued: 5-fold CV over all 100 patients, or adding the ACDC
   segmentation masks (`_gt`) as input / auxiliary target.
2. **Single split per phase, no cross-validation.** All test numbers are point
   estimates. Error bars are largest for MRI (15-patient test — one patient =
   ±6.7 pp accuracy) and smallest for X-ray (15,884 images).
3. **Echo "Mildly Reduced" (EF 40–54)** is a 15-point band and echo-derived EF
   has ~±5 % measurement noise, so ~40 % of that class's ambiguity is
   intrinsic to the labels.
4. **Phase 4 has no paired cohort.** Fusion cannot yet be validated on same-patient
   multi-modal data; the plan is frozen encoders + a head trained on a small
   paired set when one is obtained.
5. **X-ray `images_012` is a partial download** (2.5 % of images missing, a
   contiguous tail block). Prevalence and the patient-level split are unaffected.

---

## 5. The staged process (repeat this per modality)

Each phase was executed in the same fixed order — this is the methodology
section for the paper:

1. **Scaffold + dependencies.** Create `src/<phase>_{config,dataset,model,train,evaluate}.py`,
   `requirements_<phase>.txt`; install only what's new (`nibabel` for MRI, nothing
   for echo — `opencv` came with Phase 1).
2. **Data exploration — shown and reviewed before any model code.**
   `python -m src.<phase>_explore`: file-count vs metadata reconciliation, label
   distribution per split, input-shape variability, a decode/preprocess preview,
   the split it creates, and computed normalisation stats. Nothing proceeds until
   this is clean.
3. **VRAM / throughput profiling before choosing batch size.** Forward+backward
   at several batch sizes (and frame/slice counts) on synthetic tensors →
   peak VRAM + clips-per-second. On the GTX 1650 every phase turned out
   **compute-bound, not memory-bound**, so batch size was chosen for
   optimisation quality, not memory.
4. **`--quick_test`** — tiny subset, 1 epoch, end-to-end. Catches wiring bugs
   (shape mismatches, NaN guards, checkpoint IO) in seconds.
5. **Time estimate + schedule decision.** Extrapolate epoch time from the profile;
   agree epoch counts before committing (Phase 2 was trimmed 12 → 6 unfrozen
   epochs; Phase 3 uses a frozen backbone throughout).
6. **Full run, detached**, `metrics.csv` per epoch, best/last checkpoints.
7. **Test evaluation** (`<phase>_evaluate.py`) → per-class AUROC / F1 / confusion
   matrix + a predictions CSV; write `PHASE<n>_<mod>_SUMMARY.md`; commit the
   baseline.

### Per-phase training recipe

| | Phase 1 — X-ray | Phase 2 — Echo | Phase 3 — MRI |
|---|---|---|---|
| Schedule | 5 ep frozen (`lr 1e-4`) + 5 ep unfrozen (`lr 1e-5`) | 3 ep CNN-frozen (`1e-4`) + 6 ep unfrozen (`1e-5`) | 25 ep, **CNN frozen throughout**, `lr 3e-4` |
| Optimizer | Adam, wd 0 | Adam, wd 1e-4 | Adam, wd **1e-3** + head dropout 0.3 |
| Loss | `BCEWithLogitsLoss(pos_weight)` | `CrossEntropyLoss(weight)` | `CrossEntropyLoss` |
| Batch / precision | 32 / AMP | 8 / AMP | 8 / AMP |
| Best epoch | 8 | 8 | 25 |
| Val (primary) | mean AUROC **0.868** | macro AUROC **0.820** | macro AUROC **0.822** (noisy) |
| Wall-clock (GTX 1650) | ~5 h 20 m | ~47 min | ~75 s |

---

## 6. Repository layout

```
Capstone/
├── data/
│   ├── raw/{<nih images>, echonet/, acdc/}          external, git-ignored
│   └── processed/{split_index.csv, echo/, mri/}     frozen splits + norm stats
├── src/
│   ├── utils.py                     shared, modality-agnostic (all 3 phases)
│   ├── config.py  dataset.py  model.py  train.py  evaluate.py  explore.py      # Phase 1
│   ├── echo_config.py  echo_dataset.py  echo_model.py  echo_train.py  echo_evaluate.py  echo_explore.py
│   └── mri_config.py   mri_dataset.py   mri_model.py   mri_train.py   mri_evaluate.py   mri_explore.py
├── outputs/
│   ├── checkpoints/{*.pt, echo/*.pt, mri/*.pt}      git-ignored (large)
│   └── logs/{, echo/, mri/}                         metrics.csv + test_predictions.csv tracked
├── requirements.txt  requirements_echo.txt  requirements_mri.txt
├── PHASE1_XRAY_SUMMARY.md  PHASE2_ECHO_SUMMARY.md  PHASE3_MRI_SUMMARY.md
└── README.md
```

---

## 7. Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # Phase 1
pip install -r requirements_echo.txt     # + Phase 2
pip install -r requirements_mri.txt      # + Phase 3   (adds nibabel)
```

Datasets go under `data/raw/` (`<images_001..012>/`, `echonet/EchoNet-Dynamic/`,
`acdc/training/`). Then, per phase (`<p>` ∈ `∅` for X-ray / `echo` / `mri`):

```bash
python -m src.<p>_explore                 # 1. data report + frozen split  (X-ray: src.explore)
python -m src.<p>_train --quick_test      # 2. smoke test
python -m src.<p>_train                   # 3. full run  (add --resume to continue)
python -m src.<p>_evaluate                # 4. test metrics + predictions CSV
```

Environment used: single **NVIDIA GTX 1650 (3.63 GB usable)**, Python 3.14,
PyTorch 2.14 + CUDA 13, `.venv`. Every phase fits in < 2.1 GB VRAM.

---

## 8. Next — Phase 4 (fusion)

- Build `FusionModel` = three frozen encoders (loaded via
  `load_encoder_from_checkpoint`) → `concat → [B, 3072]` → MLP head.
- Obtain / assemble a **paired** multi-modal cardiac cohort (the open problem).
- Compare against each single-modality encoder + head, and against
  modality-dropout at inference (robustness to a missing modality).
