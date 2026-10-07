# Baseline — frozen experimental control

**Status: VERIFIED.** Every number below was produced by running
`python -m src.baseline_eval` against the frozen checkpoint on this machine
(Phase 2). Nothing here is copied from a report, and nothing is estimated.

The baseline is the *control*: all calibration / threshold / external-evaluation
work in this project is measured **against this artifact**. It is therefore
immutable — do not retrain it, move it, or "improve" it. Its hash is recorded
below and in `outputs/checkpoints/baseline/baseline_manifest.json`.

---

## 1. Frozen artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| Best checkpoint (by val mean AUROC) | `outputs/checkpoints/baseline/densenet121_best.pt` | `35965f610c8b578b3ad52e9d7d06a1b3054948df6df52e80aa02f2b81803d68c` |
| Last checkpoint | `outputs/checkpoints/baseline/densenet121_last.pt` | `e5f6360d834a998863f60e4f86a43d5d94e3bd28162c8a1ef7543f077e995afd` |
| Freeze record | `outputs/checkpoints/baseline/baseline_manifest.json` | written once, never overwritten |

The byte-identical originals also remain at `outputs/checkpoints/densenet121_{best,last}.pt`
because `src/fusion_config.py` and `demo_app.py` (legacy fusion / demo) hard-code
those root paths. **The root files are the live training targets; the
`baseline/` copies are the immutable control.** *(RISK R1 in
`docs/REPOSITORY_AUDIT.md`.)*

Checkpoint payload (loaded by `src/inference.load_model`): `model_state`,
`optimizer_state`, `epoch`, `val_auroc`, `target_labels`
(`["Cardiomegaly", "Effusion"]`), plus training metadata.

## 2. What the model is *(verified by reading `src/model.py`, `src/config.py`, `src/train.py`)*

| Item | Value |
|---|---|
| Architecture | `torchvision.models.densenet121(weights=ImageNet1K_V1)`, classifier replaced by `Linear(1024, 2)`; logits out, sigmoid applied for metrics |
| Input | 224 × 224 RGB, ImageNet normalisation (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) |
| Augmentation (train only) | rotate ±10°, brightness ±20 %, contrast ±20 %; **no horizontal flip** (X-ray laterality) |
| Loss | `BCEWithLogitsLoss(pos_weight = [39.945, 7.552])` computed from train prevalence |
| Optimiser | Adam, weight decay 0, **no LR scheduler** |
| Schedule | 5 epochs frozen backbone @ `1e-4`, then 5 epochs full fine-tune @ `1e-5` |
| Batch / AMP | 32, mixed precision on (CUDA) |
| Seed | 42 (`src.utils.set_seed` → python, numpy, torch, cudnn deterministic off but seed set) |
| Selection | best = highest **mean validation AUROC**, epoch 8 (0.8677973) |
| Decision threshold | **0.50**, fixed, per label — the control's operating point |

## 3. Data *(verified from `data/processed/split_index.csv`, recomputed at every eval)*

| | rows | patients | Cardiomegaly+ | Effusion+ |
|---|---|---|---|---|
| train | 76,977 | 20,797 | 1,880 | 9,001 |
| val | 16,451 | 4,468 | 392 | 1,966 |
| test | 15,884 | 4,455 | 415 | 1,997 |
| **total** | **109,312** | **29,720** | **2,687** | **12,963** |

* Split = patient-level 70/15/15, seed 42. **Patient overlap train∩val, train∩test,
  val∩test = 0 / 0 / 0** (checked programmatically, stored in the metrics JSON).
* NIH source: `data/raw/images_001…012` (the `images_012` folder is partial —
  2,808 of 112,120 images are absent, so the usable frame is 109,312).
* Test Effusion support is recorded as **1,997 although the ground truth is
  1,996**: one row (`00029705_000.png`) is a truncated PNG and
  `MultiLabelImageDataset` substitutes the next readable sample's label. This is
  a pre-existing, deterministic data defect — see `docs/REPOSITORY_AUDIT.md` §2a
  and the Phase 12 data-QA report. It is **documented, not silently fixed**,
  because the control must stay byte-for-byte reproducible.

## 4. Results — test split (n = 15,884), threshold 0.50 *(COMPLETED / VERIFIED)*

| Label | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cardiomegaly | 0.8972 | 0.3043 | 0.6867 | 0.9153 | 0.1787 | 0.2836 | 0.9093 | 285 | 1310 | 130 | 14159 |
| Effusion | 0.8587 | 0.4700 | 0.7902 | 0.7714 | 0.3320 | 0.4676 | 0.7737 | 1578 | 3175 | 419 | 10712 |
| **macro** | **0.8780** | **0.3871** | | | | | | | | | |

## 5. Results — validation split (n = 16,451), threshold 0.50 *(COMPLETED / VERIFIED)*

| Label | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cardiomegaly | 0.8751 | 0.2307 | 0.5867 | 0.9113 | 0.1391 | 0.2248 | 0.9036 | 230 | 1424 | 162 | 14635 |
| Effusion | 0.8605 | 0.4751 | 0.8077 | 0.7667 | 0.3196 | 0.4580 | 0.7716 | 1588 | 3380 | 378 | 11105 |
| **macro** | **0.8678** | **0.3529** | | | | | | | | | |

The validation macro AUROC (0.8678) equals the value stored inside the
checkpoint (0.8677973) — an independent confirmation that the frozen weights are
the ones that were selected.

## 6. Reproduction of the historical Phase 1 evaluation *(VERIFIED by comparison)*

The original run is preserved in `outputs/logs/evaluate_run.log` /
`outputs/logs/test_predictions.csv`. Re-running through the new inference layer:

| Quantity | Historical log | This re-run | Agreement |
|---|---|---|---|
| macro AUROC (test) | 0.8779 | 0.8780 | Δ ≤ 1e-4 |
| Cardiomegaly AUROC / sens / spec / prec / F1 | 0.8972 / 0.6867 / 0.9153 / 0.1787 / 0.2836 | identical to 4 dp | exact |
| Effusion AUROC | 0.8587 | 0.8587 | exact |
| Effusion spec / prec / F1 | 0.7711 / 0.3317 / 0.4673 | 0.7714 / 0.3320 / 0.4676 | 4 rows |

**Why the 4-row difference:** inference runs under `autocast` (fp16). Four
Effusion rows sit on the fp16 quantum nearest 0.5
(`00004832_033`, `00012603_012`, `00020106_002`, `00027199_004`: historical
0.500000–0.500977 vs 0.499863–0.499950 now) and flip side across runs; every
other score agrees to ≤ 1.6e-3. Ranking metrics are unaffected
(AUROC changes by ≤ 5e-6). This is *hardware/AMP non-determinism of ±4 rows at
one threshold*, not a logic change.

**Consequence for this project:** all experiments consume **one cached
prediction table** (`outputs/predictions/raw/*`), so every calibration and
threshold result is internally consistent; the ±4-row caveat applies only when
comparing a *fresh* re-computation against the historical log.

## 7. What the baseline is *not*

* **No calibration** — probabilities are raw sigmoid outputs (ECE/Brier are
  computed for the first time in Phase 5; they are `UNKNOWN` for the control
  until then).
* **No tuned thresholds** — 0.50 everywhere; per-label operating points are the
  subject of Phase 5.
* **No external validation** — no CheXpert (or any other) data is present in
  `data/raw/`; the external phase is `BLOCKED — EXTERNAL DATASET NOT AVAILABLE`.
* **No AUPRC in the original run** — AUPRC in the tables above is added by
  Phase 2's metric layer (it is a well-defined, threshold-free measure; it does
  not alter any original number).
* **No cross-validation** — one split, one seed, one training run.

## 8. How to regenerate

```bash
# 1) logits + probabilities for both splits (cached, ~3 min each on GTX 1650)
.venv/bin/python -m src.inference --split test
.venv/bin/python -m src.inference --split val

# 2) metrics at threshold 0.5 (cache hits, seconds)
.venv/bin/python -m src.baseline_eval --split test
.venv/bin/python -m src.baseline_eval --split val
```

Outputs: `outputs/metrics/baseline/baseline_metrics_{test,val}.{json,csv}`,
aggregate `outputs/metrics/baseline/baseline_metrics.json`, predictions
`outputs/predictions/baseline/baseline_{test,val}_predictions.csv`.

Add `--refresh` only if you deliberately want to recompute logits — note the
±4-row AMP caveat in §6.

## 9. Artifact map

```
outputs/
├── checkpoints/
│   ├── baseline/                 FROZEN: best + last + baseline_manifest.json
│   ├── research/                 reserved for future training runs (empty)
│   └── densenet121_{best,last}.pt  live training targets (referenced by fusion/demo)
├── metrics/baseline/
│   ├── baseline_metrics.json         aggregate (all splits)
│   ├── baseline_metrics_test.{json,csv}
│   └── baseline_metrics_val.{json,csv}
├── predictions/
│   ├── raw/                          split__ckpt__sha12.csv  (logits + probs)
│   └── baseline/                     baseline_{test,val}_predictions.csv
└── logs/                             historical Phase 1 logs (unchanged)
```
