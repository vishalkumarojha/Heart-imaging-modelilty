# Project scope — Chest X-ray calibration & threshold research

**Status of this document:** the intended scope, with each item tagged by its
current status. Nothing in this document claims a result that has not been
produced; per-phase detail lives in `docs/PHASE_REPORTS.md`.

**Status legend** (used by every document in `docs/`):

| Tag | Meaning |
|---|---|
| **IMPLEMENTED** | code exists in the repository and is importable/runnable |
| **VERIFIED** | executed on this machine; outputs inspected and recorded |
| **PLANNED** | designed but not yet built in this migration |
| **BLOCKED** | designed, blocked by a missing external dependency (named explicitly) |
| **UNKNOWN** | not determinable from the repository |
| **LEGACY** | pre-existing work, kept working, out of scope for this research line |

---

## 1. One-line scope

Reproduce, then improve the *decision layer* of an existing, frozen
Chest X-ray classifier — probability calibration and disease-specific decision
thresholds — and evaluate the result honestly on held-out and (when available)
external data.

The **model is not the research object.** DenseNet121, its training recipe and
its checkpoint are frozen (see `docs/BASELINE.md`); what changes between
experiments is only how its probabilities are calibrated and thresholded.

**Answers (each traced to an artifact in `docs/FINAL_PROJECT_STATUS.md` §2):**
ranking is strong (macro AUROC 0.878) but probabilities are badly over-confident
(`ECE ≈ mean(p) − prevalence`); temperature scaling improves NLL/Brier yet
*raises* binned ECE; the F1-optimal validation threshold lifts test F1 by
+0.070 (Cardiomegaly) / +0.019 (Effusion); nearly half of Cardiomegaly's errors
are high-confidence; Grad-CAM heat sits in the radiologist box 2.6× above area
chance. External generalisation remains the single open item: the project is
**`CORE_RESEARCH: COMPLETE` · `EXTERNAL_VALIDATION: PENDING` ·
`PAPER_PREPARATION: COMPLETE / READY FOR AUTHOR REVIEW`** (see
`docs/STATUS.md`).

## 2. Research questions

1. **Are the raw sigmoid probabilities calibrated?** Measure discrimination
   (AUROC/AUPRC) and calibration (ECE, Brier, reliability curves) of the frozen
   model. *(VERIFIED — `outputs/metrics/experiments/exp1_calibration.json`.)*
2. **Does post-hoc calibration improve probabilistic quality without changing
   ranking?** Fit temperature scaling **on the validation split only**, then
   re-measure on test. AUROC/AUPRC must be invariant (temperature scaling is
   monotone) — asserted by a test, not by assertion of faith. *(VERIFIED —
   `tests/test_experiment_outputs.py`; NLL/Brier improve, binned ECE rises.)*
3. **What operating point should each disease use?** Derive per-label thresholds
   from validation predictions under five named policies (fixed 0.50, F1-optimal,
   Youden, sensitivity-constrained, precision-constrained) and report the
   resulting test operating characteristics. *(VERIFIED — `exp2/exp3_thresholds*.json`.)*
4. **How far does it generalise?** Apply the *frozen* calibrated model and the
   *frozen* thresholds to an external cohort (CheXpert) with no re-fitting.
   **`EXTERNAL_VALIDATION: PENDING`** — no CheXpert files exist under
   `data/raw/`, `$CHEXPERT_ROOT`/`$CHEXPERT_DATA_DIR`, `/mnt/data/chexpert` or
   `/data/chexpert`. The loader and evaluator are **IMPLEMENTED**
   (`src/external_eval.py`); the phase writes only a status file
   (`no_metrics_were_computed: true`) until the data is provided.)
5. **Where does it fail, and what does the model look at?** Error analysis
   (confusion strata, confidence/error relationship) and Grad-CAM on a fixed
   , deterministic case list. *(IMPLEMENTED — `outputs/metrics/error_analysis/`,
   `outputs/gradcam/`. Analysis output, not a performance claim.)*

## 3. In scope

| # | Item | Status |
|---|---|---|
| 1 | Repository audit (code, data, results, risks) | **VERIFIED** — `docs/REPOSITORY_AUDIT.md` |
| 2 | Frozen baseline: checkpoint copies + hashes + manifest | **VERIFIED** — `outputs/checkpoints/baseline/` |
| 3 | Baseline evaluation layer (AUROC, AUPRC, confusion, per-label table) | **IMPLEMENTED / VERIFIED** — `src/metrics.py`, `src/baseline_eval.py` |
| 4 | Cached logit/probability extraction (shared by every experiment) | **IMPLEMENTED / VERIFIED** — `src/inference.py`, `outputs/predictions/raw/` |
| 5 | Environment / split-statistics capture for results provenance | **IMPLEMENTED / VERIFIED** — `src/reproducibility.py` |
| 6 | Calibration metrics (ECE, Brier, reliability bins) | **IMPLEMENTED / VERIFIED** — `src/calibration.py` |
| 7 | Temperature scaling fitted on validation only | **IMPLEMENTED / VERIFIED** — T=1.187 / 1.398 fitted on val |
| 8 | Five validation-derived threshold policies | **IMPLEMENTED / VERIFIED** — `src/thresholds.py`, val-frozen |
| 9 | Experiment matrix: raw vs calibrated × policies | **IMPLEMENTED** — exp1–exp4, `outputs/metrics/experiments/` |
| 10 | External evaluation (CheXpert) with frozen parameters | **EXT. VALIDATION PENDING** — no data; pipeline **IMPLEMENTED** |
| 11 | Error analysis + Grad-CAM case study | **IMPLEMENTED** — `src/error_analysis.py`, `src/gradcam.py` |
| 12 | Automated tests (`unittest`; pytest is not installed) | **IMPLEMENTED** — 193 tests, all green |
| 13 | Reproducibility docs, README, status report | **IMPLEMENTED** — README + docs/{PHASE_REPORTS,IMPLEMENTATION_LOG,TECH_STACK,REPRODUCIBILITY,QA_REPORT,FINAL_PROJECT_STATUS} |
| 14 | Paper-ready evidence pack: freeze snapshot, master table, bootstrap CIs, tables 1–8, figures 1–8, error-analysis CSV, region analysis, Grad-CAM export, research summary, verification report | **VERIFIED** — `outputs/research_snapshot.json` + `outputs/final_results/` (verification 25/25 PASS) |
| 15 | Paper package: evidence map, tables 01–09, figures 01–09, claim audit, reproducibility manifest, prose drafts (research question, contribution, outline, results, discussion, limitations, titles, abstract, conclusion) | **VERIFIED** — `outputs/paper/` (evidence map 69/69 verified; claim audit all-`true`; table 09 states EXTERNAL VALIDATION PENDING — DATASET UNAVAILABLE) |

## 4. Out of scope (explicitly)

* **No new architecture.** No CBAM/SE/attention blocks, no transformers, no
  focal loss, no re-training of the baseline, no hyper-parameter search.
  If a future run trains a model at all, it lands in
  `outputs/checkpoints/research/` and is never compared against itself.
* **No touching the frozen split** (`data/processed/split_index.csv`), the
  frozen metrics (`outputs/logs/`) or the baseline checkpoints.
* **No fitting on test or external data.** Calibration parameters and thresholds
  come from validation predictions only; test/external are evaluation-only.
  Any code path that would violate this is a test failure, not a warning.
* **No multi-modality research.** ECHO, MRI and fusion are LEGACY — see
  `docs/LEGACY_MODALITIES.md`.
* **No claims about data we do not have.** External metrics stay PENDING
  (`EXTERNAL_VALIDATION: PENDING`, `no_metrics_were_computed: true`) until a
  cohort is actually present.
* **No fixing of the frozen control's data defects** (truncated PNG substitution,
  `docs/REPOSITORY_AUDIT.md` §2a). They are documented; re-downloading the five
  defective files would invalidate reproducibility of the control.

## 5. Non-goals / limitations accepted

* Single split, single seed → point estimates with bootstrap confidence
  intervals (2 000 resamples, image-level; patient clustering documented in
  `docs/RESULTS.md`). Cross-validation would require retraining → out of scope.
* The split is *our* patient-level 70/15/15, not the NIH official
  `train_val_list.txt`/`test_list.txt` (those files are absent), so absolute
  numbers are not comparable with published NIH-split results (risk R10/§10 of
  the audit).
* Binned ECE rises after temperature scaling (explained; reported as a finding).
* Only 2 of 14 NIH findings are modelled (pre-existing label choice).
* AMP fp16 inference gives ±4 rows of threshold-level non-determinism between
  runs (`docs/BASELINE.md` §6) — all experiments use one cached prediction table.

## 6. Planned phase map

| Phase | Deliverable | Expected status |
|---|---|---|
| 1 | Repository audit | `docs/REPOSITORY_AUDIT.md` — **VERIFIED** |
| 2 | Freeze + reproduce baseline | `docs/BASELINE.md` — **VERIFIED** |
| 3 | Scope + legacy documentation | `docs/PROJECT_SCOPE.md`, `docs/LEGACY_MODALITIES.md` — this file |
| 4 | Calibration & threshold infrastructure + tests | **IMPLEMENTED / VERIFIED** |
| 5 | Experiments 1–4 (calibration, policies, ablations) | **IMPLEMENTED** |
| 6 | External evaluation pipeline | **BLOCKED — EXTERNAL DATASET NOT AVAILABLE**; pipeline **IMPLEMENTED** |
| 7 | Error analysis + Grad-CAM | **IMPLEMENTED** |
| 8 | README + documentation suite | **IMPLEMENTED** |
| 9 | Implementation log / changelog | **IMPLEMENTED** |
| 10 | Tech stack document | **IMPLEMENTED** |
| 11 | Reproducibility document (env capture) | **IMPLEMENTED** |
| 12 | QA + data-integrity report | **IMPLEMENTED** (`docs/QA_REPORT.md`; data scan **VERIFIED** in Phase 1) |
| 13 | `docs/FINAL_PROJECT_STATUS.md` | **IMPLEMENTED** |
| 14 | Paper-ready evidence (freeze, master table, CIs, tables, figures, exports, summary) | **VERIFIED** — `outputs/final_results/`, `verification_report.json` 25/25 PASS |

Every phase ends with a report stating status, changes, why, tests and blockers
(`docs/PHASE_REPORTS.md`); phases are never marked complete on the basis of intent.

## 7. Where results live

```
outputs/
├── research_snapshot.json     the freeze record (model/config/T/τ/env/git/tests)
├── checkpoints/baseline/     frozen control (+ manifest)
├── checkpoints/research/     future training runs (empty; reserved)
├── metrics/
│   ├── baseline/             baseline_metrics{,_test,_val}.{json,csv}   VERIFIED
│   ├── calibration/          temperature_scalers.json, reports        IMPLEMENTED
│   ├── thresholds/           thresholds_{val,calibrated_val}.json    IMPLEMENTED
│   ├── experiments/          exp1–exp4.json, experiments_summary.csv IMPLEMENTED
│   ├── external/             status.json (PENDING, no metrics)       PENDING
│   └── error_analysis/       error_analysis_test.json + strata CSV   IMPLEMENTED
├── predictions/
│   ├── raw/                  split__ckpt__sha.csv  (logits + probs)     VERIFIED
│   ├── baseline/             baseline_<split>_predictions.csv            VERIFIED
│   ├── calibrated/ thresholded/ external/   IMPLEMENTED / PENDING (external)
├── plots/{calibration,threshold}                                       IMPLEMENTED
│   ├── error_analysis/                                                IMPLEMENTED
│   ├── threshold/ also holds ROC+PR curves with threshold marks         IMPLEMENTED
│   ├── external/                                                       BLOCKED (no data)
├── gradcam/              case_list, overlays, bbox_localization     IMPLEMENTED
└── final_results/        master_results.csv, confidence_intervals.csv,
                          tables/1-8, figures/1-8, error_analysis.csv,
                          region_analysis.json, gradcam/, research_summary.json,
                          verification_report.json (25/25 PASS)          VERIFIED
└── paper/                evidence map (69/69), tables/01-09, figures/01-09,
                          claim_audit.csv, reproducibility_manifest.json,
                          prose drafts (research question→conclusion)    VERIFIED
```

Machine-readable JSON is the source of truth for every reported number; CSVs are
tidy projections of the same payloads; plots are derived views.
