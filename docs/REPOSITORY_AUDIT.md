# Repository Audit (Phase 1)

**Date:** 2026-10-07
**Branch:** `research/xray-calibration`
**Scope:** complete read-only inspection of the repository before any migration work.
**Status of application code at audit time:** unchanged (no source edits were made in Phase 1).

---

## 1. Actual repository tree

```
Capstone/
├── README.md                     18 KB — multimodal (4-phase) project overview
├── PHASE1_XRAY_SUMMARY.md        frozen X-ray reference (results + checkpoint payload)
├── PHASE2_ECHO_SUMMARY.md        EchoNet reference
├── PHASE3_MRI_SUMMARY.md         ACDC reference
├── FUSION_SUMMARY.md             Phase 4 fusion reference
├── DEMO_GUIDE.md                 Gradio demo instructions (2 uncommitted edits)
├── demo_app.py                   Gradio app, 4 tabs X-ray/ECHO/MRI/Fusion (2 uncommitted edits)
├── demo_samples/                 3 X-ray + 3 echo + 6 MRI curated sample files
├── requirements.txt              Phase 1 (X-ray) deps — unpinned
├── requirements_echo.txt         + opencv-python-headless
├── requirements_mri.txt          + nibabel
├── requirements_demo.txt         + gradio
├── src/                          4,692 LOC total (flat package, `python -m src.<mod>`)
│   ├── __init__.py               docstring only
│   ├── config.py  dataset.py  model.py  train.py  evaluate.py  explore.py  utils.py   ← X-ray (1,363 LOC)
│   ├── echo_config/dataset/model/train/evaluate/explore.py                             ← ECHO  (≈1,300 LOC)
│   ├── mri_config/dataset/model/train/evaluate/explore.py                              ← MRI   (≈1,000 LOC)
│   ├── fusion_config.py  fusion_model.py  fusion_train.py  fusion_demo.py  load_encoders.py  ← fusion (≈1,000 LOC)
│   └── __pycache__/              (tracked? no — git-ignored)
├── data/
│   ├── raw/                      git-ignored, 59 GB
│   │   ├── images_001..012/      NIH ChestX-ray14 PNGs (012 partial)
│   │   ├── Data_Entry_2017.csv   112,120 rows
│   │   ├── BBox_List_2017.csv, *.pdf (dataset docs), LICENSE_TERMS.md, MANDATORY_CITATION.md
│   │   ├── echonet/EchoNet-Dynamic/   videos + CSVs
│   │   └── acdc/training/             ACDC MRI cohort
│   └── processed/                git-ignored
│       ├── split_index.csv       FROZEN X-ray split (109,312 rows)
│       ├── image_path_index.csv  filename → path cache
│       ├── echo/, mri/, fusion/  per-phase caches
├── outputs/
│   ├── checkpoints/
│   │   ├── densenet121_best.pt   84 MB — epoch 8, best val AUROC 0.86780  ← THE X-ray checkpoint
│   │   ├── densenet121_last.pt   84 MB — epoch 10 (for --resume only)
│   │   ├── echo/  mri/  fusion/  per-phase checkpoints
│   │   ├── baseline/             byte-identical copies of the two X-ray checkpoints (Phase 2 preservation)
│   │   └── research/             empty, reserved
│   └── logs/
│       ├── metrics.csv           10 training epochs (X-ray)
│       ├── test_predictions.csv  15,884 rows (X-ray test)
│       ├── evaluate_run.log      test results table
│       ├── train_run.log         full training stdout
│       └── echo/ mri/ fusion/    per-phase logs
├── docs/                         (created by this migration; did not exist before)
├── .gitignore                    ignores data/, *.pt, checkpoints, verbose logs
├── .venv/                        Python 3.14.7 virtualenv
└── .git/                         branch `main` + new branch `research/xray-calibration`
```

**Not present:** `tests/`, `notebooks/`, `scripts/`, `docs/`, CI config, Docker, FastAPI,
experiment tracker (wandb/mlflow), LICENSE file, pinned dependency versions.

**Git status at audit:** clean on `main` except two uncommitted files
(`demo_app.py`, `DEMO_GUIDE.md` — a `GRADIO_SERVER_PORT` env-var override).
History: 7 commits, phases 1→4 then two demo commits.

---

## 2. X-ray implementation (the future primary pipeline)

Everything below was read from source, not from the handoff notes.

| Field | Actual value | Source |
|---|---|---|
| Architecture | `torchvision.models.densenet121(weights=IMAGENET1K_V1)` → `features` → ReLU → `AdaptiveAvgPool2d(1)` → flatten 1024-d → `Linear(1024, 2)`, **logits out** | `src/model.py` |
| Encoder/head split | `XrayEncoder` (no head) + `MultiLabelClassifier` (head) — designed for fusion reuse | `src/model.py:21,47` |
| Dataset | NIH ChestX-ray14, `Data_Entry_2017.csv` (112,120 rows) | `src/config.py` |
| Images on disk | 109,312 of 112,120 (`images_012` partial: 2,808 missing, contiguous tail) | `PHASE1_XRAY_SUMMARY.md` + verified |
| Labels | `Cardiomegaly`, `Effusion` (order = logit order), from `Finding Labels` split on `\|` | `src/config.py:43`, `src/dataset.py:79` |
| Input | 224×224, RGB (grayscale replicated ×3), ImageNet mean/std | `src/config.py:49-51` |
| Augmentation (train) | `Rotate ±10°` p=0.7 (border_mode=0), `RandomBrightnessContrast ±0.15` p=0.7, resize+normalize; **no horizontal flip** | `src/dataset.py:237-254` |
| Eval transform | resize + normalize only | `src/dataset.py:254` |
| Split | **patient-level** random 70/15/15, seed 42; official NIH split files NOT present | `src/dataset.py:109-163` |
| Split artifact | `data/processed/split_index.csv` — 109,312 rows / 29,720 patients | verified |
| Split counts | train 76,977 (20,797 pt) · val 16,451 (4,468 pt) · test 15,884 (4,455 pt) | verified |
| Leakage check | `groupby(Patient ID).split.nunique() > 1` → **0 patients overlap** (logged, warning only) | `src/dataset.py:155-162` |
| Loss | `BCEWithLogitsLoss(pos_weight=…)` computed from **train** prevalence → `[39.945, 7.552]` | `src/train.py:258-268` |
| Optimizer | Adam, weight_decay = 0, **no scheduler** (constant LR per phase) | `src/train.py:188` |
| Schedule | 5 epochs frozen (backbone frozen except `denseblock4` + `norm5`, lr 1e-4) + 5 epochs unfrozen (lr 1e-5) | `src/config.py:71-75` |
| Batch size | 32 (`num_workers=4`, `drop_last=True` on train) | `src/config.py:67` |
| Seed | 42 (python/numPy/torch); `cudnn.benchmark=True`, deterministic **off** | `src/utils.py:20-39` |
| AMP | enabled (`torch.amp.GradScaler` + `autocast`) | `src/config.py:78` |
| Checkpointing | `densenet121_best.pt` = max mean val AUROC (any epoch/phase); `densenet121_last.pt` = every epoch (atomic write) + `--resume` | `src/train.py:123-149,231-242` |
| Best epoch | **8**, mean val AUROC **0.867797** | `outputs/logs/metrics.csv` |
| Evaluation | `python -m src.evaluate` — sigmoid → threshold 0.5 → per-label AUROC / sens / spec / precision / F1 + `test_predictions.csv` | `src/evaluate.py` |
| Metrics NOT computed today | **AUPRC, ECE, Brier, confusion-matrix file, calibration, thresholds** | — |

**Verdict vs. handoff:** the implementation matches the handoff baseline exactly
(DenseNet121, ImageNet init, 224², Cardiomegaly+Effusion, weighted BCE,
patient-level 70/15/15, seed 42, 5+5 epochs, Adam, batch 32, AMP, best-checkpoint).
One handoff detail that the code adds: **there is no LR scheduler** (handoff is silent).

---

## 3. Historical results — located and verified

The handoff's numbers are **real and traceable** to repository artifacts:

| Claim | Artifact | Verified value |
|---|---|---|
| val mean AUROC ≈ 0.868 | `outputs/logs/metrics.csv` epoch 8 | **0.867797** (Cardio 0.875071, Effusion 0.860523) |
| test mean AUROC ≈ 0.878 | `outputs/logs/evaluate_run.log` line 11 | **0.8779** |
| Cardiomegaly AUROC 0.897 / sens 0.687 / spec 0.915 / prec 0.179 / F1 0.284 | same log, line 9 | 0.8972 / 0.6867 / 0.9153 / 0.1787 / 0.2836 (415 pos) |
| Effusion AUROC 0.859 / sens 0.790 / spec 0.771 / prec 0.332 / F1 0.467 | same log, line 10 | 0.8587 / 0.7902 / 0.7711 / 0.3317 / 0.4673 (1,997 pos recorded) |
| Checkpoint = epoch 8 | `torch.load(...)` payload | epoch 8, `best_auroc 0.8677973275491944` |
| Predictions file | `outputs/logs/test_predictions.csv` | 15,884 rows, test split, IDs match `split_index.csv` exactly |

Nothing in the handoff was fabricated; nothing needs to be recomputed to trust it.

### 2a. Data-integrity defects found during the audit

A full structural scan of all **109,312** PNGs (PNG `IEND` trailer check) plus a
decode check of every suspect file found **exactly 5 defective images**, all in the
partial `images_012` folder:

| File | Size | Defect | Split |
|---|---|---|---|
| `00029705_000.png` | 143 KB | truncated (`OSError: image file is truncated`) | **test** |
| `00029717_001.png` | 110 KB | truncated | train |
| `00029718_000.png` | 0 B | empty file | train |
| `00029719_000.png` | 0 B | empty file | train |
| `00029720_000.png` | 0 B | empty file | train |

Consequences, established by cross-checking `test_predictions.csv` against
`split_index.csv` and `Data_Entry_2017.csv`:

1. **`MultiLabelImageDataset.__getitem__` silently substitutes** the *next readable
   sample's image and label* when a file fails to load
   (`src/dataset.py:295-315`). So for the test-set defect:
   - row `00029705_000.png` in the historical `test_predictions.csv` carries
     `true_Effusion=1` (inherited from `00029705_001.png`; its score 0.878906 is
     byte-identical to the next row's — the fingerprint of the substitution),
     although the ground truth is **0 / "No Finding"**.
   - Test Effusion support is therefore recorded as **1,997 instead of 1,996**.
   - Impact ≈ 1 row of 15,884 (0.006 %); it cannot explain the observed
     precision gap. The behaviour is *deterministic*, so a re-run reproduces it:
     it is documented, not silently "fixed", because the baseline must stay frozen.
2. The four defective **train** rows substitute a neighbour's image+label during
   training (4 of 76,977 rows) — negligible, but recorded.
3. **Duplicate images exist in NIH** — 3 pixel-identical pairs confirmed inside the
   test set (`00027864_001`/`00028591_002`, `00016568_052`/`00016568_053`,
   `00008727_014`/`00008727_015`). None of the detected pairs cross splits, and the
   patient-level split prevents same-patient leakage, but *cross-patient* image
   duplicates were **not audited exhaustively** (59 GB read) — recorded as a limitation.
4. **AMP fp16 score quantization** — 19 score pairs in `test_predictions.csv` are
   bit-identical because logits are computed under `autocast` in fp16
   (≈11-bit mantissa). Harmless for metrics, but it means stored scores are not
   high-precision.

---

## 4. ECHO implementation (legacy candidate)

`src/echo_*.py` — EchoNet-Dynamic, 3-class EF bucket (<40 / 40–54 / ≥55).
ResNet18-per-frame + 1-layer BiLSTM over 16 frames → 1024-d `EchoEncoder`
(fusion contract) + throw-away `Linear(1024,3)` head. Batch 8, AMP, official
`Split` column (7,465 / 1,288 / 1,277), inverse-frequency-weighted CE,
3 frozen + 6 unfrozen epochs, best macro AUROC **0.802** (test, 1,277 videos).
Artifacts: `outputs/checkpoints/echo/echo_cnn_lstm_{best,last}.pt`,
`outputs/logs/echo/*`. Deps: `requirements_echo.txt` (opencv already present).
**State: complete and working. Independent of the X-ray pipeline** (imports only
`src.utils`). Safe to retain as legacy.

## 5. MRI implementation (legacy candidate)

`src/mri_*.py` — ACDC, 5-class diagnosis (DCM/HCM/MINF/NOR/RV).
ResNet18-per-slice ([ED, ES, ED−ES] 3-channel) + BiLSTM over 10 slices → 1024-d
`MriEncoder` + Dropout/Linear head. 100 patients, stratified 70/15/15 (70/15/15
patients), 25 epochs **CNN frozen throughout**, lr 3e-4, batch 8.
Test macro AUROC **0.706**, accuracy 6/15 — explicitly small-data limited.
Artifacts: `outputs/checkpoints/mri/*`, `outputs/logs/mri/*`, `data/processed/mri/`.
Deps: `requirements_mri.txt` (nibabel).
**State: complete but weak; independent of X-ray** (imports `src.utils` only).

## 6. Fusion implementation (legacy candidate)

`src/fusion_{config,model,train,demo}.py` + `src/load_encoders.py`.
Frozen Phases 1–3 encoders → per-modality LayerNorm → learned missing-modality
token → concat 3072 → MLP(512) → three task heads. **There is no paired
tri-modal cohort**, so it is trained/validated *single-modality-present only* and
demonstrated on explicitly-labelled *synthetic* embedding combinations
(`fusion_demo.py`, `SYNTHETIC_demo_predictions.csv`).
Validated results: X-ray pathway 0.865 vs 0.878 standalone, Echo 0.806 vs 0.802,
MRI 0.628 vs 0.706 (`outputs/logs/fusion/single_modality_validation.csv`).

**Dependency that constrains the migration:** `fusion_config.XRAY_CKPT` and
`demo_app.py:CKPT["xray"]` both point at
**`outputs/checkpoints/densenet121_best.pt` (root path)**. Moving or renaming that
file breaks legacy fusion + demo. → The root checkpoint must stay in place; the
baseline freeze must be a *copy*, not a move.

---

## 7. Current results inventory

| Component | Metric | Value | Artifact |
|---|---|---|---|
| X-ray val (epoch 8) | mean AUROC | 0.867797 | `outputs/logs/metrics.csv` |
| X-ray test | mean AUROC | 0.8779 | `outputs/logs/evaluate_run.log` |
| X-ray test Cardiomegaly | AUROC / F1 @0.5 | 0.8972 / 0.2836 | idem |
| X-ray test Effusion | AUROC / F1 @0.5 | 0.8587 / 0.4673 | idem |
| ECHO test | macro AUROC | 0.802 | `PHASE2_ECHO_SUMMARY.md` |
| MRI test | macro AUROC | 0.706 | `PHASE3_MRI_SUMMARY.md` |
| Fusion (single-mod present) | X-ray 0.865 / Echo 0.806 / MRI 0.628 | — | `outputs/logs/fusion/single_modality_validation.csv` |
| AUPRC / ECE / Brier / calibration / thresholds / external | **not computed anywhere** | — | — |

## 8. Actual dependencies (detected, not assumed)

| Package | Version in `.venv` | Used by |
|---|---|---|
| Python | 3.14.7 | all |
| torch | 2.14.0+cu130 | all |
| torchvision | 0.29.0+cu130 | X-ray, ECHO, MRI |
| CUDA runtime | 13.0 (GTX 1650, 4 GB) | training/inference |
| numpy | 2.5.3 | all |
| pandas | 3.0.5 | all |
| scikit-learn | 1.9.0 | metrics (AUROC only today) |
| albumentations | 2.0.8 | X-ray (+echo/mri) transforms |
| matplotlib | 3.11.1 | installed; **no plotting code exists yet** |
| Pillow | present | image IO |
| tqdm | present | progress bars |
| opencv-python-headless | present (via albumentations) | echo video |
| nibabel | installed (mri req) | MRI |
| gradio | in `requirements_demo.txt` | demo app |
| **pytest** | **NOT installed** | tests do not exist |
| seaborn | not installed | — |

## 9. Technical risks

| # | Risk | Severity | Evidence |
|---|---|---|---|
| R1 | Truncated PNG silently substitutes another sample's **image and label** | High (correctness) | `src/dataset.py:295-315`, §2a.1 |
| R2 | No tests, no CI — any refactor is unguarded | High | no `tests/` |
| R3 | Fixed output paths: `evaluate.py` overwrites `test_predictions.csv`; `metrics.csv` appends across runs | Medium | `src/config.py:30,90` |
| R4 | Fusion/demo hard-code the root checkpoint path | Medium (blocks moving baseline) | `fusion_config.py`, `demo_app.py` |
| R5 | Threshold 0.5 + `pos_weight≈40` → Cardiomegaly precision 0.179 | Medium (the research motivation) | `evaluate_run.log` |
| R6 | No calibration / ECE / Brier / AUPRC anywhere | Medium (research gap) | audit |
| R7 | No external dataset (CheXpert) present, no download tooling | Medium (blocks Exp 5) | `data/raw/` listing |
| R8 | `data/raw` + `data/processed` git-ignored → fresh clone must rebuild caches | Low | `.gitignore` |
| R9 | Cross-patient duplicate images not exhaustively audited | Low | §2a.2 |
| R10 | Single split, no cross-validation → point estimates only | Low | `metrics.csv` |
| R11 | AMP fp16 score quantization | Low | §2a.3 |
| R12 | Uncommitted changes on `main` (demo port override) | Low | `git status` |
| R13 | Unpinned `requirements.txt` → future installs may drift | Low | file contents |

## 10. Missing information

- AUPRC / ECE / Brier / confusion-matrix baselines — never computed.
- CheXpert (or any external cohort) — absent; no fetch script, no label map.
- Official NIH `train_val_list.txt` / `test_list.txt` — absent, so the split is
  *our* patient-level random split, not the NIH-official one (documented, but it
  means results are not comparable to papers using the official split).
- Dependency versions are unpinned in `requirements*.txt`.
- No wall-clock timing artifacts for the evaluation step.
- No licence file for the project code itself (dataset licences are present under
  `data/raw/`).

## 11. Recommended migration plan

1. **Phase 2 — freeze baseline:** copy the two X-ray checkpoints to
   `outputs/checkpoints/baseline/` (root copies stay for fusion/demo), record
   sha256 + config + verified metrics, extend evaluation with AUPRC and confusion
   matrices **without touching the model or the split**, and prove the historical
   numbers reproduce.
2. **Phase 3 — scope change:** retitle/describe the project as X-ray research;
   ECHO/MRI/fusion stay at their current paths (no mass move — R4) and are
   documented as legacy in `docs/LEGACY_MULTIMODAL.md`.
3. **Phase 4 — research infrastructure:** shared metric layer (AUROC, AUPRC, ECE,
   Brier, confusion), cached logit/probability extraction, temperature scaling
   fitted **on validation only**, five validation-derived threshold policies,
   prediction storage (raw + calibrated + thresholded).
4. **Phase 5 — experiments 1–4** as self-describing JSON/CSV/plot outputs under
   `outputs/{metrics,predictions,plots}/…`.
5. **Phase 6 — external validation:** CheXpert loader + frozen-policy evaluator;
   expect `BLOCKED — EXTERNAL DATASET NOT AVAILABLE` until the cohort is fetched.
6. **Phase 7 — error analysis + Grad-CAM** (analysis tool, not a claim).
7. **Phases 8–13 — documentation, reproducibility, QA, status report.**
8. **Tests first where cheap:** metrics, calibration, thresholds, split leakage,
   model/checkpoint round-trip — all runnable without retraining.
