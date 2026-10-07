# Chest X-ray calibration & threshold selection

Primary project: **chest X-ray classification research** — a reproducible
study of **calibration-aware, class-specific decision policies** (the research
contribution) applied to a **frozen DenseNet121** multi-label chest-X-ray
classifier. ECHO/MRI material in this repository is historical/preliminary
multimodal work and a future extension, outside this study's scope.

From a fixed baseline checkpoint to calibrated probabilities, validation-frozen
decision thresholds, an experiment matrix, error analysis and Grad-CAM — every
number computed from one cached prediction table, every parameter fitted on
**validation only**.

> **Status legend** (used by every document in `docs/`):
> `IMPLEMENTED` code exists · `VERIFIED` code exists **and** its output was
> recomputed/checked · `PLANNED` not written · `BLOCKED` cannot proceed
> (stated reason) · `UNKNOWN` not measured · `LEGACY` kept, not part of the
> research scope.
>
> Current phase-by-phase state: [`docs/PHASE_REPORTS.md`](docs/PHASE_REPORTS.md) ·
> scope and non-goals: [`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md) ·
> status: [`docs/STATUS.md`](docs/STATUS.md) · full roll-up:
> [`docs/FINAL_PROJECT_STATUS.md`](docs/FINAL_PROJECT_STATUS.md)
>
> **Project status:** CORE RESEARCH COMPLETE · BASELINE FROZEN · CALIBRATION /
> THRESHOLD / ABLATION / ERROR ANALYSIS / EXPLAINABILITY COMPLETE ·
> **EXTERNAL VALIDATION PENDING** (no external cohort available) ·
> **PAPER PREPARATION COMPLETE / READY FOR AUTHOR REVIEW**
> (`outputs/paper/`). This is **not** a clinically validated system.

---

## 1. The question

A classifier that is good at ranking (AUROC) is not automatically good at
deciding. Two things break in practice:

1. **Calibration** — with `pos_weight ≈ 40` for Cardiomegaly, a predicted
   probability of 0.9 does not mean 90% chance of disease.
2. **Thresholds** — the default 0.5 is arbitrary; different clinical uses
   (rule-out vs rule-in) need different operating points.

This project measures both, on one frozen model, with the discipline that
**calibration parameters and thresholds are fitted on validation and only
evaluated** on test (and, when data exists, on an external cohort).

## 2. The model (frozen control)

| | |
|---|---|
| Architecture | DenseNet121 (ImageNet V1), 224×224, 2-label head |
| Labels | `Cardiomegaly`, `Effusion` (NIH ChestX-ray14, `Finding Labels`) |
| Loss | `BCEWithLogitsLoss`, `pos_weight=[39.945, 7.552]` |
| Split | patient-level 70/15/15, seed 42 → 86,608 / 16,451 / 15,884 images |
| Schedule | 5 epochs frozen (lr 1e-4) + 5 unfrozen (lr 1e-5), Adam, AMP |
| Best epoch | 8 — val mean AUROC **0.8678** |
| Checkpoint | `outputs/checkpoints/baseline/densenet121_best.pt`<br>`sha256 35965f610c8b…` — **never retrained, never overwritten** |

### Baseline test results (frozen, threshold 0.5)

| Label | AUROC | AUPRC | Sensitivity | Specificity | Precision | F1 | Prevalence |
|---|---|---|---|---|---|---|---|
| Cardiomegaly | **0.8972** | 0.3043 | 0.6867 | 0.9153 | 0.1787 | 0.2836 | 2.6 % (415/15,884) |
| Effusion | **0.8587** | 0.4700 | 0.7902 | 0.7714 | 0.3320 | 0.4676 | 12.6 % (1,996/15,884) |
| **macro** | **0.8780** | 0.3871 | | | | | |

Source: `outputs/metrics/baseline/baseline_metrics.json` (recomputed from the
cached table; reproduces the historical log to ≤5e-6 AUROC — the 4-row fp16
delta is explained in [`docs/BASELINE.md`](docs/BASELINE.md) §6).

## 3. What was measured

### 3.1 Calibration (Experiment 1)

Temperature scaling fitted on **validation** only (scipy, bounds [0.05, 50]):

| Label | T | NLL val → | Brier test → | ECE test → |
|---|---|---|---|---|
| Cardiomegaly | **1.187** | 0.2337 → 0.2302 | 0.0681 → 0.0678 | 0.1061 → 0.1169 ↑ |
| Effusion | **1.398** | 0.4889 → 0.4721 | 0.1561 → 0.1533 | 0.2077 → 0.2292 ↑ |

Honest findings, not rounded away:

* Temperature scaling **improves NLL and Brier** (its objective) but
  **increases binned ECE** — NLL ≠ ECE; a monotone rescaling can move every
  bin's mass without fixing the binwise gap.
* **ECE is essentially bin-invariant here** (5/10/15/20 bins give the same
  value): every confidence bin is over-confident, so
  `ECE ≈ mean(p) − prevalence` (0.132 − 0.026 = 0.106 for Cardiomegaly;
  0.333 − 0.126 = 0.207 for Effusion). Root cause: `pos_weight ≈ 40` pushes
  scores up while prevalence stays 2.6 %.
* AUROC/AUPRC are exactly invariant to temperature (asserted by a test).

### 3.2 Thresholds (Experiments 2 & 3)

Five policies, all fitted on validation:

| Policy | Cardio τ | Effusion τ | Test F1 (raw) | Δ F1 vs 0.5 |
|---|---|---|---|---|
| fixed@0.5 | 0.500 | 0.500 | 0.284 / 0.468 | — |
| F1-optimal | 0.902 | 0.770 | **0.354 / 0.487** | **+0.070 / +0.019** |
| Youden J | 0.183 | 0.449 | 0.185 / 0.455 | −0.099 / −0.013 |
| Sensitivity ≥ 0.90 | 0.052 | 0.304 | 0.128 / 0.412 | −0.156 / −0.055 |
| Precision ≥ 0.50 | 0.982 | 0.848 | 0.256 / 0.467 | −0.028 / −0.001 |

* The constrained policies trade F1 for their constraint, and they meet it on
  the split they were fitted to (val recall 0.901 / 0.900 for
  sensitivity@0.90; val precision 0.500 / 0.500 for precision@0.50); on test
  they transfer: precision@0.50 → precision **0.530 / 0.516**,
  sensitivity@0.90 → recall **0.921 / 0.891**.
* **Experiment 3 (calibrated) reproduces Experiment 2's confusion matrices
  exactly** — `sigmoid(logit/T) ≥ 0.5 ⟺ sigmoid(logit) ≥ 0.5`, and policies
  refit in a monotone space land on the same operating points; only the numeric
  τ differ. This is asserted by a test, not asserted by hand.

### 3.3 Error analysis (Phase 7)

At the F1-optimal policy on test:

| | Cardiomegaly | Effusion |
|---|---|---|
| Error rate | 3.54 % (563) | 14.28 % (2,268) |
| **High-confidence errors** (certainty ≥ 0.90) | **274 = 48.7 % of errors** | 193 = 8.5 % of errors |
| Errors wrong on *both* labels | 585 images (Jaccard 0.131 of error sets) | |
| Worst patient | 60 error images (repeat-offender → label-noise signal) | |

Errors live in the **high**-confidence bins (bin 5/10: 90.8 % error rate for
Cardiomegaly) — the model fails by confidently calling negatives positive.

### 3.4 Grad-CAM (Phase 7b)

Deterministic case list (k=2 per stratum × 2 labels × 4 strata = 16 overlays,
selected without any RNG) in `outputs/gradcam/overlays/`. Quantitative check on
the **fixed cohort of all 43 test images that carry a NIH radiologist bounding
box** for the boxed label (24 Cardiomegaly + 19 Effusion, no selection):

| Metric | Value |
|---|---|
| Pointing game (argmax inside box) | 21/43 = **48.8 %** (Cardio 19/24, Effusion 2/19) |
| Heat-mass concentration ratio (mass in box ÷ area share) | mean **2.60**, median 2.87, **38/43 > 1** |

Interpretation as reported: heat concentrates in the box ~2.6× more than area
chance for both labels, but the Effusion CAM's *peak* often falls outside the
box (2/19 pointing hits) — the pointing game understates a diffuse signal. Both
numbers are published as measured.

### 3.5 External validation — external cohort **PENDING**

**Project status: `CORE_RESEARCH: COMPLETE` · `EXTERNAL_VALIDATION: PENDING`.**

The pipeline (`src/external_eval.py`: CheXpert loader, uncertainty policy
`u_zeroes`, frozen model + temperature + thresholds, no refit; env overrides
`$CHEXPERT_ROOT` / `$CHEXPERT_DATA_DIR` plus `/mnt/data/chexpert` and
`/data/chexpert` are probed) is `IMPLEMENTED`. `data/raw/chexpert/` does not
exist in this repository, so the phase writes only
`outputs/metrics/external/status.json` with `no_metrics_were_computed: true`
and **no metric, prediction or plot file**. Tests enforce this. External
validity is the *single* remaining open question — see
`outputs/final_results/research_summary.json` and `docs/STATUS.md`.

### 3.6 Paper-ready evidence (`outputs/final_results/`)

Everything the paper needs is derived from the frozen artifacts, not re-measured:

| Artifact | Contents |
|---|---|
| `master_results.csv` | 24-row canonical table, arms A (baseline) / B (calibration) / C (threshold) / D (combined) × 5 policies × 2 labels |
| `confidence_intervals.csv` | 112 bootstrap CIs (2 000 resamples, seed 42; AUROC/AUPRC/F1/sens/spec/prec/acc/ECE/Brier) |
| `tables/table_1…8_*.csv` | dataset / config / main results / ablation / thresholds / errors / explainability / external status |
| `figures/fig_1…8_*.png` | ROC, PR, reliability, threshold sweep, confusion, training curves, error analysis, Grad-CAM |
| `error_analysis.csv` | class × error type × confidence bin × count × percentage |
| `region_analysis.json` | cautious pointing-game + concentration record (fixed 43-image cohort) |
| `gradcam/` | exported overlays + `gradcam_cases.csv` (sample_id, class, ground truth, prediction, confidence, error type) |
| `research_summary.json` | single evidence document (findings + limitations) |
| `verification_report.json` | docs↔artifact cross-check (currently **25/25 PASS**) |
| `outputs/research_snapshot.json` | the freeze record (model, config, T, τ, git state, test status) |

Rebuild with
`.venv/bin/python -m src.master_results && .venv/bin/python -m src.uncertainty && \
.venv/bin/python -m src.paper_artifacts`.

### 3.7 Paper package (`outputs/paper/` — READY FOR AUTHOR REVIEW)

Publication-facing deliverable, derived **only** from the artifacts above:

| Artifact | Contents |
|---|---|
| `paper_evidence_map.json` | every paper claim → source artifact (69 claims, 69 verified) |
| `tables/table_01…09_*.csv` | dataset / baseline / main / ablation / thresholds / calibration / error / explainability / external-status |
| `figures/figure_01…09_*.png` | architecture + dataset distribution schematics; research figures copied from `final_results/` |
| `claim_audit.csv` | claim × section × source × verified (all `true`) |
| `reproducibility_manifest.json` | model/checkpoint/dataset/split/seed/preprocessing/training/T/τ/env/git/test count |
| `*.md` | research question, contribution, outline, draft results, discussion, limitations, 10 ranked titles, abstract, conclusion |

The prose documents are run through a **number-hygiene check**: every numeric
token in `outputs/paper/*.md` must match a verified artifact value or a
documented constant, otherwise `src.paper_build` reports it. Table 9 states
`EXTERNAL VALIDATION PENDING — DATASET UNAVAILABLE`; no external numbers exist.

## 4. Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Data: NIH ChestX-ray14 under `data/raw/images_*/` + `Data_Entry_2017.csv`
(external, git-ignored). Then:

```bash
# 0. verify the frozen artifacts (safe, read-only)
.venv/bin/python -m unittest discover -s tests        # 193 tests

# 1. recompute the baseline from the frozen checkpoint
.venv/bin/python -m src.baseline_eval --split test
.venv/bin/python -m src.baseline_eval --split val

# 2. cached logits/probabilities (the single source for every experiment)
.venv/bin/python -m src.inference --split test

# 3. fit calibration + thresholds on VAL ONLY, then freeze
.venv/bin/python -m src.fit_parameters                # writes the frozen JSONs

# 4. experiments 1-4 (calibration, policies, ablations, robustness)
.venv/bin/python -m src.experiments --exp all

# 5. error analysis + Grad-CAM
.venv/bin/python -m src.error_analysis --split test
.venv/bin/python -m src.gradcam

# 6. external validation (reports PENDING until CheXpert is downloaded)
.venv/bin/python -m src.external_eval --status

# 7. paper-ready evidence (derived, deterministic, read-only over artifacts)
.venv/bin/python -m src.research_freeze --no-tests   # outputs/research_snapshot.json
.venv/bin/python -m src.master_results               # master table + verification report
.venv/bin/python -m src.uncertainty                  # bootstrap confidence intervals
.venv/bin/python -m src.paper_artifacts              # tables, figures, exports, summary

# 8. paper package (evidence map, tables, figures, claim audit, manifest)
.venv/bin/python -m src.paper_build                  # outputs/paper/
```

Environment actually used: Python 3.14.7, PyTorch 2.14.0+cu130 / CUDA 13,
single GTX 1650, `.venv` interpreter (system `python3` has no torch). Tests use
`unittest` — pytest is not installed; seaborn is not installed.

## 5. Repository layout

```
Capstone/
├── docs/
│   ├── PROJECT_SCOPE.md            in/out of scope, phase map, status legend
│   ├── PHASE_REPORTS.md            status / changes / why / tests / blockers
│   ├── REPOSITORY_AUDIT.md         Phase 1 audit + data-defect table
│   ├── BASELINE.md                 frozen baseline: hashes, reproduction, AMP delta
│   ├── LEGACY_MODALITIES.md        ECHO / MRI / fusion (LEGACY, kept)
│   ├── EXPERIMENTS.md              exp 1-4 results (calibration, thresholds, robustness)
│   ├── RESULTS.md                  headline results + how every number was produced
│   ├── STATUS.md                   project status: CORE COMPLETE, external PENDING
│   └── FINAL_PROJECT_STATUS.md     full phase roll-up
├── src/
│   ├── config.py                   single source of paths + research settings
│   ├── reproducibility.py          hashing, atomic JSON, env capture
│   ├── metrics.py  inference.py    AUROC/AUPRC/ECE… · cached logit extraction
│   ├── baseline_eval.py            frozen baseline evaluation
│   ├── calibration.py  thresholds.py  temperature scaling · 5 policies
│   ├── fit_parameters.py           fit on val, freeze, verify
│   ├── experiments.py  plots.py    exp 1-4 runner · figures
│   ├── error_analysis.py  gradcam.py   Phase 7
│   ├── external_eval.py            Phase 6 (PENDING on data)
│   ├── research_freeze.py          outputs/research_snapshot.json
│   ├── master_results.py           master_results.csv + verification report
│   ├── uncertainty.py              bootstrap confidence intervals
│   ├── paper_artifacts.py          tables/figures/exports/research_summary
│   ├── paper_build.py              outputs/paper/ evidence+package builder
│   └── <legacy modality modules>   echo_* mri_* fusion_* demo_app.py (LEGACY)
├── tests/                          193 unittest tests (all green)
├── data/raw/                       NIH images + CSV (git-ignored)
├── data/processed/split_index.csv  frozen split cache
└── outputs/
    ├── checkpoints/baseline/       frozen control + manifest
    ├── research_snapshot.json      the freeze record
    ├── predictions/{raw,baseline,calibrated,thresholded}/
    ├── metrics/{baseline,calibration,thresholds,experiments,
    │            error_analysis,external}/
    ├── plots/{calibration,threshold,error_analysis}/
    ├── gradcam/{case_list,overlays,gradcam_summary,bbox_localization}
    ├── final_results/              master/CI/tables/figures/exports/summary
    └── paper/                      evidence map/tables/figures/audit/abstract
```

## 6. Hard rules this project does not break

1. **No refitting on test/external data** — enforced at runtime (`ValueError`)
   and by tests, not by convention.
2. **No new architecture** — no CBAM/SE/attention/transformers/focal loss, no
   retraining the baseline, no hyper-parameter search.
3. **No invented numbers** — a missing dataset produces a `BLOCKED` status
   file, never an estimate.
4. **The frozen control is not touched** — baseline checkpoints, the frozen
   split and the historical logs are read-only; the 5 defective PNGs are
   documented (`docs/REPOSITORY_AUDIT.md` §2a), not silently fixed.
5. **One cached prediction table per split** — all experiments share the same
   logits, so AMP fp16 noise cannot compound between results.

## 7. Legacy: multi-modality (kept, out of scope)

The repository also contains the earlier multi-modality work — ECHO (EchoNet),
cardiac MRI (ACDC), late-fusion layer and the Gradio demo. It is **LEGACY**:
kept and documented, never part of the X-ray research claims, and it hard-depends
on the root `outputs/checkpoints/densenet121_best.pt` (do not delete it).
See [`docs/LEGACY_MODALITIES.md`](docs/LEGACY_MODALITIES.md),
[`PHASE2_ECHO_SUMMARY.md`](PHASE2_ECHO_SUMMARY.md),
[`PHASE3_MRI_SUMMARY.md`](PHASE3_MRI_SUMMARY.md),
[`FUSION_SUMMARY.md`](FUSION_SUMMARY.md),
[`DEMO_GUIDE.md`](DEMO_GUIDE.md).

## 8. Known limitations

* Single split, single seed, **one bootstrapped CI pass** (2000 resamples,
  resampling unit = test image, not patient; patient-level clustering is noted
  in `docs/RESULTS.md`).
* Our split is patient-level 70/15/15, **not** the NIH official lists (the
  official split files are absent), so absolute numbers are not comparable with
  published NIH-split results.
* Only 2 of 14 NIH findings are modelled (pre-existing label choice).
* AMP fp16 gives ±4 rows of threshold-level non-determinism between runs
  (≤1.6e-3 score, ≤5e-6 AUROC) — documented in `docs/BASELINE.md` §6.
* External generalisation is **unmeasured** (`EXTERNAL_VALIDATION: PENDING`)
  until a cohort is downloaded (Phase 6). This is the only open research item.
* Binned ECE *rises* after temperature scaling (documented, reproduced,
  explained as the mean-confidence shift); NLL and Brier both improve.
