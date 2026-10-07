# Phase reports

Every phase ends here with status / changes / why / tests / blockers.
Status tags follow `docs/PROJECT_SCOPE.md` §4: IMPLEMENTED, VERIFIED, PLANNED,
BLOCKED, UNKNOWN, LEGACY. A phase is never marked complete on intent.

Legend of run commands (this repo uses the venv interpreter; `pytest` is not
installed):

```
.venv/bin/python -m unittest discover -s tests          # full suite
.venv/bin/python -m src.fit_parameters                  # refit (val only)
.venv/bin/python -m src.experiments --exp all           # experiments 1-4
.venv/bin/python -m src.external_eval --status          # external phase status
```

---

## Phase 1 — Repository audit — **VERIFIED**

* **Changes:** `docs/REPOSITORY_AUDIT.md` (inventory, modalities, hard
  dependency on root `outputs/checkpoints/densenet121_best.pt`, 5-defect PNG
  table, consequences: substitution / duplicates / AMP fp16).
* **Why:** nothing may be frozen before the repo's real contents are known.
* **Tests:** PNG `IEND`-trailer scan over all 109,312 files (structural check;
  5 defects found and documented, not fixed).
* **Blockers:** none.

## Phase 2 — Baseline freeze + reproduction — **VERIFIED**

* **Changes:** `src/reproducibility.py`, `src/metrics.py`, `src/inference.py`,
  `src/baseline_eval.py`, `src/config.py` (research output tree only);
  `docs/BASELINE.md`; `outputs/checkpoints/baseline/` (frozen copy + manifest,
  best `sha256 35965f610c8b…`); `outputs/metrics/baseline/baseline_metrics*`.
* **Why:** all later claims need one immutable checkpoint, one cached
  prediction table and a written reproduction of the historical numbers.
* **Tests:** metrics smoke vs sklearn; reproduction of test AUROC
  0.8972 / 0.8587 and val macro AUROC 0.8678. Difference vs the historical
  log is exactly **4 Effusion rows** at the 0.5 fp16 quantum → documented in
  `docs/BASELINE.md` §6 (score Δ ≤ 1.6e-3, AUROC Δ ≤ 5e-6); every experiment
  reads one cached table so the delta cannot compound.
* **Blockers:** none.

## Phase 3 — Scope + legacy documentation — **IMPLEMENTED**

* **Changes:** `docs/PROJECT_SCOPE.md`, `docs/LEGACY_MODALITIES.md`.
* **Why:** ECHO/MRI/fusion are out of scope but are hard dependencies of the
  demo and of `src/fusion_config.py`; they must be documented, not deleted.
* **Tests:** n/a (docs; figures re-checked against the fusion CSV).
* **Blockers:** none.

## Phase 4 — Calibration & threshold infrastructure — **IMPLEMENTED**

* **Changes:**
  * `src/calibration.py` — ECE/MCE/Brier/NLL, `reliability_bins`,
    `fit_temperature` (scipy bounded minimisation + golden-section fallback),
    `TemperatureScaler` (refuses `test`/`external` fit split at runtime),
    JSON save/load.
  * `src/thresholds.py` — 5 policies (`fixed`, `f1_optimal`, `youden`,
    `sensitivity_constrained`, `precision_constrained`),
    `fit_thresholds` (refuses eval splits), `apply_thresholds`.
  * `src/config.py` — Phase-4 settings: `CALIBRATION_FIT_SPLIT="val"`,
    `THRESHOLD_FIT_SPLIT="val"`, 15 equal-width bins, temperature bounds,
    targets (sens 0.90 / precision 0.50), artifact paths.
  * `src/fit_parameters.py` — CLI `--refit/--check`; freezes
    `temperature_scalers.json`, `thresholds_val.json`,
    `thresholds_calibrated_val.json` on **val only**.
  * Tests: `test_metrics`, `test_calibration`, `test_thresholds`,
    `test_fit_parameters`, `test_baseline_artifacts`.
* **Why:** calibration and threshold selection are the research contribution;
  every fit must be provably val-only.
* **Tests:** full suite green; the val-only guard raises `ValueError` for
  `test`/`external` (asserted).
* **Frozen parameters (val, n=16,451):** T = 1.18654 (Cardiomegaly),
  1.39792 (Effusion). Raw thresholds — fixed 0.5/0.5; f1_optimal 0.9021/0.7699;
  youden 0.1826/0.4488; sensitivity@0.90 0.0519/0.3044;
  precision@0.50 0.9822/0.8475.
* **Blockers:** none.

## Phase 5 — Experiments 1–4 — **IMPLEMENTED**

* **Changes:** `src/plots.py` (Agg; reliability, calibration bars,
  ROC/PR + threshold marks, sweeps), `src/experiments.py`
  (`exp1_calibration`, `exp2_thresholds` raw, `exp3` calibrated variant,
  `exp4_robustness`: binning sweep 5/10/15/20, target sweeps, 10× bootstrap
  threshold stability), tests `test_experiment_outputs.py`.
  Outputs: `outputs/metrics/experiments/experiments_summary.csv`
  (332 rows), `exp{1..4}*.json`, calibrated + thresholded prediction CSVs,
  plots under `outputs/plots/{calibration,threshold}/`.
* **Why:** one cached prediction table → all variants share exactly the same
  logits; raw vs calibrated differ only by the frozen temperature.
* **Verified findings:**
  * Temperature scaling improves **NLL and Brier** for both labels on val and
    test, but **increases binned ECE** (test Cardio 0.1061 → 0.1169,
    Effusion 0.2077 → 0.2292): NLL and ECE are different objectives.
  * **ECE is essentially bin-invariant here** (5/10/15/20 bins give the same
    value to 4 dp) because every confidence bin is over-confident, so
    `ECE ≈ mean(p) − prevalence` (0.132 − 0.026 = 0.106 Cardio;
    0.333 − 0.126 = 0.207 Effusion) — a consequence of `pos_weight ≈ 40`.
  * AUROC/AUPRC are invariant to temperature (asserted by a test).
  * F1-optimal vs fixed@0.5 on test: Cardio F1 0.2836 → 0.3536 (Δ +0.0700),
    Effusion 0.4676 → 0.4866 (Δ +0.0191). precision@0.50 → precision
    0.5303 / 0.5157; sensitivity@0.90 → recall 0.9205 / 0.8908.
  * **exp3 (calibrated) reproduces exp2's confusion matrices exactly** — the
    sigmoid/logit monotonicity means policies refit in a calibrated space land
    on the same operating points; only the numeric τ differ. Asserted by a
    test (not a bug).
* **Tests:** 108-test suite green at the end of Phase 5
  (`test_experiment_outputs` recomputes published ECE/Brier/confusion from the
  stored prediction CSVs).
* **Blockers:** none.

## Phase 6 — External validation (CheXpert) — **BLOCKED — EXTERNAL DATASET NOT AVAILABLE**

* **Changes:** `src/config.py` (CheXPERT paths, `CHEXPERT_LABEL_MAP`,
  `EXTERNAL_UNCERTAINTY_POLICY="u_zeroes"`, `EXTERNAL_STATUS_JSON`),
  `src/external_eval.py` (`check_external_data`, `apply_uncertainty_policy`,
  `load_external_frame`, `predict_external`, `evaluate_external`, `run`,
  CLI `--status/--split/--limit`), `tests/test_external_eval.py`.
* **Why:** the pipeline must exist so the block is a data problem, not an
  implementation gap — and a missing cohort must never yield a number.
* **Implemented:** CheXpert CSV loader with explicit uncertainty policy
  (−1 → 0; blank → 0), `Path` → `images/` resolution with existence check
  (missing rows raise `FileNotFoundError`), model inference with the frozen
  eval transform, metrics (AUROC/AUPRC/ECE/Brier + all five frozen val
  policies + curves) written only when data is present.
* **Verified behaviour without data:** `run()` returns
  `BLOCKED — EXTERNAL DATASET NOT AVAILABLE` and writes only
  `outputs/metrics/external/status.json` (`no_metrics_were_computed: true`);
  tests assert **no** `external_eval_*` JSON/CSV and no prediction CSV exist;
  `load_external_frame` raises `ExternalDataNotAvailable`.
* **Tests:** 13 new tests (availability, blocked-run honesty, uncertainty
  policy, synthetic CheXpert layout: label mapping/missing images/missing
  column). Suite: **108 tests OK**.
* **Blockers:** `data/raw/chexpert/{valid.csv,images/}` absent → status
  `BLOCKED — EXTERNAL DATASET NOT AVAILABLE` (confirmed 2026-10-07).
  To unblock: drop the official files in the expected layout and re-run
  `.venv/bin/python -m src.external_eval --split valid`.

---

## Phase 7 — Error analysis + Grad-CAM — **IMPLEMENTED**

* **Changes:**
  * `src/error_analysis.py` — confusion strata per policy, high-confidence
    errors, confidence→error bins, cross-label error overlap, per-patient error
    concentration, deterministic Grad-CAM case list (k=2 per stratum,
    ordered by certainty desc / Image Index asc — no RNG).
  * `src/gradcam.py` — Grad-CAM on `encoder.features.denseblock4`,
    deterministic overlays for the frozen case list, plus a **cohort-wide**
    localization check against NIH's `BBox_List_2017.csv`.
  * `src/plots.py` — `strata_bars`, `error_rate_by_confidence`.
  * Outputs: `outputs/metrics/error_analysis/error_analysis_{split}.json` +
    `error_strata_{split}.csv`, `outputs/gradcam/case_list.json`,
    `gradcam_summary.json`, `bbox_localization.json`, 16 overlays,
    `outputs/plots/error_analysis/`.
  * Tests: `tests/test_error_analysis.py`, `tests/test_gradcam.py`.
* **Why:** calibration/threshold numbers say nothing about *where* errors are
  concentrated; Grad-CAM must be attached to a fixed case list, not to
  hand-picked beauties, and must report a localization number only where a
  radiologist box exists.
* **Verified findings (test, calibrated variant, f1_optimal policy):**
  * Error rate: Cardiomegaly 3.54% (563/15,884), Effusion 14.28% (2,268).
  * **High-confidence errors** (certainty ≥ 0.90): Cardiomegaly **274 =
    48.7% of all its errors**; Effusion 193 = 8.5% of its errors. The
    cardiomegaly model's mistakes are mostly *confident* mistakes — consistent
    with the over-confidence documented in Phase 5.
  * Errors concentrate above p=0.5 (bin 5 of 10: 90.8% error rate for
    Cardiomegaly) — i.e. the model fails by confidently calling negatives
    positive, not by being unsure.
  * Cross-label: 585 images wrong on *both* labels; Jaccard of error sets
    0.131 (largely independent failure modes).
  * Patient concentration: 4,449 error images from 1,317 patients; worst
    patient 60 errors (repeat-offender = label-noise signal; consistent with
    the audit's duplicate/substitution findings).
  * Grad-CAM bbox cohort — **fixed set of all 43 test images that have a NIH
    box for the boxed label** (24 Cardiomegaly + 19 Effusion), no selection:
    pointing game 21/43 = 48.8%, mean concentration ratio **2.60**
    (38/43 > 1: heat lands in the box 2.6× more than area chance).
    Per label: Cardiomegaly 19/24 pointing, ratio 2.72; Effusion 2/19
    pointing, ratio 2.46 — the effusion CAM mass is in the box but its argmax
    usually is not, so the pointing game understates it; both numbers are
    reported as-is.
* **Tests:** 140 tests green (strata recomputed from the stored CSV, case list
  re-selection identical, all overlays exist, bbox statistics recomputed from
  stored cases).
* **Blockers:** none.

---

## Phase 8 — README + documentation suite — **IMPLEMENTED**

* **Changes:** `README.md` rewritten to describe *this* project (frozen
  baseline, calibration, thresholds, error analysis, Grad-CAM, BLOCKED
  external phase, reproduce ladder, hard rules); legacy multi-modality moved to
  §7 with pointers to `docs/LEGACY_MODALITIES.md` and the phase summaries.
* **Why:** the entry point must state what the repo now researches and what is
  legacy, with every quoted number traceable to a JSON artifact.
* **Tests:** all quoted values checked against
  `outputs/metrics/{baseline,experiments,error_analysis}` and
  `outputs/gradcam/bbox_localization.json` (one F1 table corrected against the
  CSV during writing).
* **Blockers:** none.

## Phase 9 — Implementation log — **IMPLEMENTED**

* **Changes:** `docs/IMPLEMENTATION_LOG.md` — 30 numbered entries
  (phase, change, why, verification) for Phases 1–13, plus a
  "decisions worth remembering" table.
* **Why:** an audit trail is part of the deliverable; each research-phase
  change is tied to the test that verifies it.
* **Tests:** n/a (doc).
* **Blockers:** none.

## Phase 10 — Tech stack — **IMPLEMENTED**

* **Changes:** `docs/TECH_STACK.md` — versions read from the running `.venv`
  (torch 2.14.0+cu130, CUDA 13.0, Python 3.14.7, sklearn 1.9.0, scipy 1.18.1,
  matplotlib 3.11.1, albumentations 2.0.8), what is used by which module, and
  what is deliberately absent (pytest, seaborn) or legacy-only (nibabel,
  gradio). `requirements.txt` gained `scipy` (Phase 4's temperature fit
  imports it; it was only present transitively).
* **Why:** a stack document written from memory drifts; this one is read off
  the environment.
* **Tests:** every listed version printed by an import check;
  `py_compile src/*.py tests/*.py` clean.
* **Blockers:** none.

## Phase 11 — Reproducibility document — **IMPLEMENTED**

* **Changes:** `docs/REPRODUCIBILITY.md` — immutable inputs (checkpoint/split
  hashes), split statistics (patient overlap 0), environment of record, a
  6-rung reproduction ladder with **expected values per rung**, the three known
  non-determinism sources and their mitigations, the provenance chain from a
  README number down to a checkpoint SHA, and an explicit "what cannot be
  reproduced" list.
* **Why:** reproducibility is a claim about procedure, so it is written as
  commands + expected outputs, not prose.
* **Tests:** expected values cross-checked against the JSON/CSV artifacts.
* **Blockers:** none.

## Phase 12 — QA + data-integrity report — **IMPLEMENTED**

* **Changes:** `docs/QA_REPORT.md` — per-file test breakdown (140 tests), data
  integrity table (109,312 files, 5 defects, 0 patient overlap), the guard-rail
  table (rule → enforcement point → test), published-number recomputation
  matrix, known open issues (ECE rise, exp2≡exp3, fp16 delta, Effusion
  pointing game, external BLOCKED), and static-check status (no ruff/mypy).
* **Why:** quality evidence must be checkable line by line.
* **Tests:** suite re-run for the counts quoted (20/21/18/17/13/7/14/12/18 = 140).
* **Blockers:** none.

## Phase 13 — Final project status — **IMPLEMENTED**

* **Changes:** `docs/FINAL_PROJECT_STATUS.md` — phase-by-phase status table
  with evidence paths, headline results (baseline / calibration / thresholds /
  error analysis / Grad-CAM / external), the six guarantees kept, what is still
  open (external data, commits, CI), and the one-command verification.
* **Why:** a single roll-up a reader can trust and re-check.
* **Tests:** numbers taken from the artifacts, not from earlier drafts.
* **Blockers:** commit pending explicit instruction; external validation
  remains `BLOCKED — EXTERNAL DATASET NOT AVAILABLE`.

## Phase 14 — Paper-ready evidence (freeze, master table, CIs, tables, figures, exports) — **VERIFIED**

* **Changes:**
  * `src/research_freeze.py` → `outputs/research_snapshot.json`: the freeze
    record (baseline id, config, checkpoint hashes, split statistics, frozen
    temperatures/thresholds, environment, git state, test status). Written once,
    read-only afterwards.
  * `src/master_results.py` → `outputs/final_results/master_results.csv`
    (24 canonical rows: arms A baseline / B calibration-only / C threshold-only /
    D combined × all 5 policies × 2 labels) + `verification_report.json`
    (25 automated docs↔artifact checks, **25/25 PASS** — incl. Cardio 48.7 %
    high-conf errors, 43/21/2.60 Grad-CAM figures, test F1 for every policy).
  * `src/uncertainty.py` → `outputs/final_results/confidence_intervals.csv`
    (112 rows: percentile bootstrap, 2 000 resamples, seed 42, image-level;
    AUROC/AUPRC/F1/sens/spec/prec/acc/ECE/Brier). Fixed bug: calibration
    metrics were bootstrapped against the mismatched labels (`p` vs `p[ix]`),
    which inflated their CIs — caught by the bracketing test.
  * `src/paper_artifacts.py` → `tables/table_1..8`, `figures/fig_1..8`,
    `error_analysis.csv`, `region_analysis.json`, `gradcam/` export +
    `gradcam_cases.csv`, `research_summary.json`. All derived deterministically
    from frozen artifacts; Table 8 is an honest *pending* row.
  * `src/external_eval.py`: candidate detection now probes `$CHEXPERT_ROOT` /
    `$CHEXPERT_DATA_DIR` and `/mnt/data/chexpert`, `/data/chexpert`; status JSON
    gains `project_status: {core_research: COMPLETE, external_validation:
    PENDING}` (top-level `status` unchanged → test compat kept).
  * `tests/test_consistency.py` (cross-artifact honesty: headline numbers,
    unique experiment IDs, val-only fitting guards, frozen checkpoint hashes,
    percentages↔counts, external honesty, verification PASS) and
    `tests/test_final_results.py` (evidence pack: master↔CI exact match, table
    integrity, 8 figures, error-analysis CSV, region analysis, Grad-CAM export,
    research summary, snapshot). **176 tests total, all green** (~15 s).
  * Docs: README §3.5/§3.6/§4/§5/§8 updated; `docs/STATUS.md`,
    `docs/EXPERIMENTS.md`, `docs/RESULTS.md` added; `PROJECT_SCOPE.md`,
    `FINAL_PROJECT_STATUS.md`, `IMPLEMENTATION_LOG.md` rolled up.
* **Why:** turn the completed implementation into an auditable, paper-ready
  study without moving the goalposts (no retraining, no new metrics, no
  external data fabricated). The freeze + consistency tests make "drift
  between doc and data" a test failure.
* **Tests:** 176 green; `test_final_results` proves every CI point estimate
  equals the master-table value to 4 dp; `test_consistency` proves selection
  honesty (val-only fitting) and checkpoint immutability.
* **Blockers:** commit still pending explicit instruction; external validation
  is now **`CORE_RESEARCH: COMPLETE` · `EXTERNAL_VALIDATION: PENDING`**.

---

## Cumulative test status

| Phase | Suite | Result |
|---|---|---|
| 1–5 | 95 tests | **OK** (~11 s) |
| 6 | 108 tests | **OK** (~12 s) |
| 7 | 140 tests | **OK** (~15 s) |
| 8–13 | 140 tests | **OK** (~15 s) |
| 14 | 176 tests | **OK** (~15 s) |
| 15 | 193 tests | **OK** (~15 s) |

---

## Phase 15 — Paper package (READY FOR AUTHOR REVIEW)

**Status:** `PAPER_PREPARATION: COMPLETE / READY FOR AUTHOR REVIEW` ·
`EXTERNAL_VALIDATION: PENDING`.

**Changes**
* `src/paper_build.py` builds `outputs/paper/`: evidence map (`paper_evidence_map.json`,
  65 claims / 65 verified), tables 01–09 (table 09 = `EXTERNAL VALIDATION PENDING —
  DATASET UNAVAILABLE`), figures 01–09 (2 new schematics + 7 copies of the
  deterministic research figures), claim audit (all `verified = true`),
  reproducibility manifest, and a prose number-hygiene check.
* Prose documents added: research question, contribution statement, 20-section outline,
  draft results, draft discussion, limitations, 10 ranked titles + selected title,
  abstract, conclusion.
* ECE scientific-correctness check: `TestECEImplementationIsStandard` recomputes raw and
  calibrated ECE by an independent code path (equal-width 15 bins; empty bins → 0) and
  matches the published values; the paper reports NLL↓ / Brier↓ / ECE↑ without hiding.
* `tests/test_paper_package.py` (14 tests): 9 tables + 9 figures + evidence map + claim
  audit + manifest + docs + number hygiene.
* README (§1 purpose, §3.7 paper package, §4 + command, §5 layout, 193 tests),
  `docs/STATUS.md`, `docs/PROJECT_SCOPE.md`, `docs/FINAL_PROJECT_STATUS.md`,
  `outputs/research_snapshot.json` (`project_status` block; tests 193 PASS).
* **No retraining, no new architecture, no new experiment.** Everything derived from
  frozen artifacts.

**Tests:** 193 green (~15 s) incl. new ECE recompute + paper package tests.

**Blockers:** commit still pending explicit instruction; external validation PENDING;
next action recommended = human review of the paper content (`outputs/paper/`).
