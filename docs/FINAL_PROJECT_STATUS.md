# Final project status

**Scope:** reproducible Chest X-ray research on a frozen DenseNet121
(Cardio­megaly / Effusion) baseline — calibration, validation-frozen thresholds,
experiment matrix, error analysis, Grad-CAM, external pipeline, paper-ready
evidence pack, **paper package (ready for author review)**.
**Date of this snapshot:** 2026-10-07.
**Test suite:** 193 tests, all green, ~15 s (`.venv/bin/python -m unittest
discover -s tests`).
**Project status:** **`CORE_RESEARCH: COMPLETE` · `BASELINE: FROZEN` ·
`CALIBRATION: COMPLETE` · `THRESHOLD ANALYSIS: COMPLETE` · `ABLATION: COMPLETE`
· `ERROR ANALYSIS: COMPLETE` · `EXPLAINABILITY: COMPLETE` ·
`EXTERNAL_VALIDATION: PENDING` · `PAPER_PREPARATION: COMPLETE / READY FOR
AUTHOR REVIEW`.**

Legend: `IMPLEMENTED` (code exists) · `VERIFIED` (code + checked output) ·
`PLANNED` · `BLOCKED` (reason stated) · `UNKNOWN` (not measured) · `LEGACY`
(kept, out of scope).

---

## 1. Phase-by-phase

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Repository audit | **VERIFIED** | `docs/REPOSITORY_AUDIT.md`; PNG `IEND` scan of 109,312 files → 5 documented defects |
| 2 | Baseline freeze + reproduction | **VERIFIED** | `docs/BASELINE.md`; `outputs/checkpoints/baseline/` (sha `35965f610c8b`); test AUROC 0.8972 / 0.8587 reproduced ≤5e-6 |
| 3 | Scope + legacy documentation | **IMPLEMENTED** | `docs/PROJECT_SCOPE.md`, `docs/LEGACY_MODALITIES.md` |
| 4 | Calibration & threshold infrastructure | **VERIFIED** | `src/calibration.py`, `src/thresholds.py`, `src/fit_parameters.py`; val-only fit guards; frozen params `outputs/metrics/{calibration,thresholds}/` |
| 5 | Experiments 1–4 | **IMPLEMENTED** | `src/experiments.py`; `outputs/metrics/experiments/experiments_summary.csv` (332 rows) + plots |
| 6 | External validation pipeline | **BLOCKED — EXTERNAL DATASET NOT AVAILABLE**; pipeline **IMPLEMENTED** | `src/external_eval.py`; `outputs/metrics/external/status.json` (`no_metrics_were_computed: true`) |
| 7 | Error analysis + Grad-CAM | **IMPLEMENTED** | `src/error_analysis.py`, `src/gradcam.py`; `outputs/metrics/error_analysis/`, `outputs/gradcam/` (16 overlays + bbox cohort) |
| 8 | README + documentation suite | **IMPLEMENTED** | `README.md` rewritten around this project; docs tree below |
| 9 | Implementation log | **IMPLEMENTED** | `docs/IMPLEMENTATION_LOG.md` |
| 10 | Tech stack | **IMPLEMENTED** | `docs/TECH_STACK.md` |
| 11 | Reproducibility document | **IMPLEMENTED** | `docs/REPRODUCIBILITY.md` |
| 12 | QA + data-integrity report | **IMPLEMENTED** | `docs/QA_REPORT.md` |
| 13 | Final status report | **IMPLEMENTED** | this file |
| 14 | Paper-ready evidence (freeze, master table, CIs, tables, figures, exports, summary) | **VERIFIED** | `outputs/research_snapshot.json`, `outputs/final_results/` (tables 1–8, figures 1–8, `master_results.csv`, `confidence_intervals.csv`, `error_analysis.csv`, `region_analysis.json`, `gradcam/`, `research_summary.json`, `verification_report.json` 25/25 PASS) |
| 15 | Paper package (evidence map, tables 01–09, figures 01–09, claim audit, reproducibility manifest, prose drafts) | **VERIFIED** | `outputs/paper/` (`paper_evidence_map.json` 69/69 verified, `claim_audit.csv` all-`true`, `reproducibility_manifest.json`, tables/figures, research question/abstract/conclusion/etc.), `src/paper_build.py`, `tests/test_paper_package.py` |

Documentation set: `README.md`, `docs/{PROJECT_SCOPE,REPOSITORY_AUDIT,BASELINE,LEGACY_MODALITIES,PHASE_REPORTS,IMPLEMENTATION_LOG,TECH_STACK,REPRODUCIBILITY,QA_REPORT,FINAL_PROJECT_STATUS}.md`.

## 2. Headline results (all from frozen artifacts)

**Baseline (frozen, τ = 0.5, test n = 15,884)**

| Label | AUROC | AUPRC | Sens | Spec | Precision | F1 | Prev |
|---|---|---|---|---|---|---|---|
| Cardiomegaly | 0.8972 | 0.3043 | 0.6867 | 0.9153 | 0.1787 | 0.2836 | 2.6 % |
| Effusion | 0.8587 | 0.4700 | 0.7902 | 0.7714 | 0.3320 | 0.4676 | 12.6 % |
| macro | **0.8780** | 0.3871 | | | | | |

**Calibration (temperature fitted on val, T = 1.187 / 1.398)**

| | NLL | Brier | ECE (15 bins) |
|---|---|---|---|
| Cardio, test | 0.2297 → **0.2262** ✓ | 0.0681 → **0.0678** ✓ | 0.1061 → 0.1169 ✗ |
| Effusion, test | 0.4819 → **0.4690** ✓ | 0.1561 → **0.1533** ✓ | 0.2077 → 0.2292 ✗ |

* Finding: temperature scaling optimises NLL, **not** binned ECE; with every
  bin over-confident, `ECE ≈ mean(p) − prevalence` and is bin-invariant
  (0.132 − 0.026 = 0.106 Cardio; 0.333 − 0.126 = 0.207 Effusion). AUROC/AUPRC
  are exactly invariant to T (test-asserted).

**Thresholds (all fitted on val, evaluated on test)**

| Policy | τ Cardio / Effusion | Test F1 | vs 0.5 |
|---|---|---|---|
| fixed | 0.500 / 0.500 | 0.284 / 0.468 | — |
| **F1-optimal** | 0.902 / 0.770 | **0.354 / 0.487** | **+0.070 / +0.019** |
| Youden J | 0.183 / 0.449 | 0.185 / 0.455 | −0.099 / −0.013 |
| Sensitivity ≥ 0.90 | 0.052 / 0.304 | 0.128 / 0.412 | constraints met on test: recall 0.921 / 0.891 |
| Precision ≥ 0.50 | 0.982 / 0.848 | 0.256 / 0.467 | constraints met on test: precision 0.530 / 0.516 |

* Experiment 3 (calibrated thresholds) reproduces Experiment 2's confusion
  matrices exactly — monotone-equivalence, test-asserted, reported not hidden.

**Error analysis (test, F1-optimal policy)**

| | Cardiomegaly | Effusion |
|---|---|---|
| Error rate | 3.54 % (563) | 14.28 % (2,268) |
| High-confidence errors (≥0.90) | **274 = 48.7 % of errors** | 193 = 8.5 % |
| Wrong on both labels | 585 images (error-set Jaccard 0.131) | |
| Worst patient | 60 error images (label-noise signal) | |

**Grad-CAM (16 deterministic overlays + bbox cohort)**

| Metric | Value |
|---|---|
| Case list | 2 per stratum × 2 labels × 4 strata = 16, ordered by (certainty ↓, image index ↑), no RNG |
| Bbox cohort (fixed, all boxed test images for the boxed label) | n = **43** (24 Cardio + 19 Effusion) |
| Pointing game | 21/43 = **48.8 %** (Cardio 19/24; Effusion 2/19) |
| Concentration ratio (mass-in-box ÷ area share) | mean **2.60**, median 2.87, **38/43 > 1** |

**External validation:** project status **`CORE_RESEARCH: COMPLETE` ·
`EXTERNAL_VALIDATION: PENDING`** — the external cohort
(`data/raw/chexpert/`, `$CHEXPERT_ROOT[, $CHEXPERT_DATA_DIR]`,
`/mnt/data/chexpert`, `/data/chexpert`) is absent. The pipeline is implemented
and unit-tested; no metric, prediction or plot file exists;
`status.json` says `no_metrics_were_computed: true`. See `docs/STATUS.md`.
The paper package records this honestly (`table_09_external_validation_status.csv`:
**EXTERNAL VALIDATION PENDING — DATASET UNAVAILABLE**) and reports no external
numbers anywhere.

## 3. Guarantees kept

1. No calibration/threshold fit on test or external data — enforced at runtime
   (`ValueError`) and by tests.
2. No new architecture, no retraining, no hyper-parameter search; the frozen
   baseline checkpoints and split are untouched (hash-asserted).
3. No fabricated numbers: the missing external cohort produced a `BLOCKED`
   status file only; `UNKNOWN`/`BLOCKED` tags are used where data is absent.
4. One cached prediction table per split; AMP fp16 delta documented once
   (`docs/BASELINE.md` §6) and not allowed to compound.
5. Legacy ECHO/MRI/fusion/demo preserved and documented as `LEGACY`;
   root `outputs/checkpoints/densenet121_best.pt` kept for their hard dependency.
6. Data defects documented, not silently fixed (`docs/REPOSITORY_AUDIT.md` §2a).

## 4. What is still open

| Item | Status | Unblock |
|---|---|---|
| External generalisation (CheXpert) | **`EXTERNAL_VALIDATION: PENDING`** — no data | download official files to `data/raw/chexpert/{valid.csv,images/}` (or point `$CHEXPERT_ROOT` at a mount), run `.venv/bin/python -m src.external_eval --split valid` |
| Multi-seed runs | **PLANNED out of scope** (single split, no retraining allowed) | would require retraining → outside contract |
| Research checkpoint under `outputs/checkpoints/research/` | empty by design | reserved for any future training run |
| Committing this migration | **not committed** (modified/new paths in working tree) | commit only when explicitly requested |
| Lint/type checking | no ruff/mypy in the environment | see `docs/QA_REPORT.md` §6 |

Single-split statistics are now quantified with **bootstrap confidence
intervals** (`outputs/final_results/confidence_intervals.csv`; 2 000 resamples,
seed 42, image-level resampling with patient clustering documented as a
limitation).

## 5. How to verify this snapshot in one command

```bash
.venv/bin/python -m unittest discover -s tests     # 193 tests, ~15 s
```

A green run means: checkpoint hashes intact, split integrity intact, published
metrics recompute from stored CSVs, every paper table/figure/export matches its
source artifact, every paper claim is verified against its artifact, the
external phase is still honestly pending, and the
`verification_report.json` has zero `DIFF`s.

For the derived evidence pack (deterministic, read-only over artifacts):

```bash
.venv/bin/python -m src.research_freeze --no-tests
.venv/bin/python -m src.master_results
.venv/bin/python -m src.uncertainty
.venv/bin/python -m src.paper_artifacts
.venv/bin/python -m src.paper_build
```
