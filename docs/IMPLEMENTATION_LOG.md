# Implementation log (Phase 9)

Chronological record of the research migration (Phases 1–13), with the reason
for each change and what verified it. The legacy multi-modality work
(2026-09-07 commits, Phases 1–4 of the *old* README) predates this log and is
covered by `docs/LEGACY_MODALITIES.md`.

Commit history below starts at the state inherited from those commits; the
research migration itself is **not yet committed** — it is staged as working
tree changes (commit only when explicitly requested).

---

## 2026-10-06

| # | Phase | Change | Why | Verified by |
|---|---|---|---|---|
| 1 | **1** | `docs/REPOSITORY_AUDIT.md` — full inventory, hard-dependency map, risk register | nothing may be frozen before the repo's real contents are known | manual review + PNG scan |
| 2 | 1 | PNG structural scan (`IEND` trailer check) over all **109,312** files | the dataset's integrity had to be established, not assumed | 5 defects found and documented in §2a (`00029705_000` truncated test PNG, `00029717_001` truncated train PNG, `00029718/19/20_000` zero-byte train PNG) |
| 3 | 1 | Documented the *consequences* of those defects (row substitution at load, duplicate handling, fp16 delta) instead of fixing the data | re-downloading would invalidate reproducibility of the frozen control | `REPOSITORY_AUDIT.md` §2a |
| 4 | **2** | `src/reproducibility.py` (sha256, atomic JSON w/ NaN→null, env + split-stat capture) | results must be provably tied to a checkpoint and an environment | `tests/test_baseline_artifacts.py` |
| 5 | **2** | `src/metrics.py` — AUROC/AUPRC/binary/multilabel reports | one implementation used by every phase | smoke tests vs `sklearn` |
| 6 | **2** | `src/inference.py` — sha-keyed prediction cache `raw/{split}__{ckpt}__{sha12}.csv` | one cached table → all experiments share identical logits; fp16 noise cannot compound | cache-hit/miss tests, row-count guard |
| 7 | **2** | `src/baseline_eval.py` + frozen checkpoint copy + manifest | freeze the control before any experiment touches it | reproduction of historical test AUROC 0.8972/0.8587 (≤5e-6) |
| 8 | 2 | Explained the only reproduction delta: **exactly 4 Effusion rows** at the 0.5 fp16 quantum | an unexplained number is an unusable number | `docs/BASELINE.md` §6 (image IDs listed) |
| 9 | **3** | `docs/PROJECT_SCOPE.md` (research questions, in/out, phase map, status legend) | scope is a contract, not a vibe | — |
| 10 | **3** | `docs/LEGACY_MODALITIES.md` | ECHO/MRI/fusion stay, but flagged LEGACY | fusion CSV values corrected against the real file |

## 2026-10-06 (later)

| # | Phase | Change | Why | Verified by |
|---|---|---|---|---|
| 11 | **4** | `src/calibration.py` — ECE/MCE/Brier/NLL, reliability bins, `fit_temperature` (bounded scipy + golden-section fallback), `TemperatureScaler` | calibration is the first research contribution | `tests/test_calibration.py` |
| 12 | **4** | Runtime guard: `TemperatureScaler` **raises** on `test`/`external` fit split | "val only" must be enforced by code, not by convention | guard test |
| 13 | **4** | `src/thresholds.py` — 5 policies + `fit_thresholds` (refuses eval splits) | same reason as above | `tests/test_thresholds.py` |
| 14 | **4** | `src/fit_parameters.py` — fits and freezes `temperature_scalers.json`, `thresholds_val.json`, `thresholds_calibrated_val.json` on val | frozen parameters are inputs to every later experiment | `tests/test_fit_parameters.py`; T = 1.1865 / 1.3979 |
| 15 | **4** | Phase-4 settings added to `src/config.py` (bins, bounds, targets, paths) | one source of truth for experiment settings | import test |
| 16 | **5** | `src/plots.py` — Agg backend, reliability/bars/curves/sweeps | headless, batch-safe figures | every figure written by an experiment run |
| 17 | **5** | `src/experiments.py` — exp1 calibration, exp2 raw policies, exp3 calibrated variant, exp4 robustness (binning sweep, target sweeps, 10× bootstrap) | the experiment matrix the scope promised | `tests/test_experiment_outputs.py` (recomputes published numbers from the CSVs) |
| 18 | 5 | Findings recorded as measured: NLL/Brier improve but **binned ECE rises**; ECE ≈ mean(p) − prevalence (bin-invariant); AUROC invariant to T; **exp2 ≡ exp3 confusion matrices** | no rounding away of inconvenient results | assertions + `docs/PHASE_REPORTS.md` |

## 2026-10-07

| # | Phase | Change | Why | Verified by |
|---|---|---|---|---|
| 19 | **6** | `src/external_eval.py` — availability check, CheXpert loader, `u_zeroes` uncertainty policy, frozen-parameter evaluation, CLI | the block must be a *data* problem, never an implementation gap | `tests/test_external_eval.py` (13 tests) |
| 20 | 6 | Honest-blocking behaviour: `run()` writes **only** `status.json` (`no_metrics_were_computed: true`) and no metric/prediction/plot file | a missing cohort may never yield a number | tests assert `external_eval_*` files do not exist |
| 21 | 6 | Status: **BLOCKED — EXTERNAL DATASET NOT AVAILABLE** (`data/raw/chexpert/` absent) | stated, not papered over | `outputs/metrics/external/status.json` |
| 22 | 7 | `src/error_analysis.py` — strata per policy, high-confidence errors, confidence→error bins, cross-label overlap, patient concentration, deterministic case list | calibration/threshold numbers say nothing about *where* errors live | `tests/test_error_analysis.py` |
| 23 | 7 | `src/gradcam.py` — Grad-CAM on `encoder.features.denseblock4`, 16 overlays, plus a **cohort-wide** bbox check on all 43 boxed test images | qualitative method + the one quantitative check the data allows | `tests/test_gradcam.py` (math tests + published-artifact recomputation) |
| 24 | 7 | Findings: 48.7 % of Cardiomegaly errors are high-confidence; pointing game 21/43; concentration ratio 2.60 (38/43 > 1) | recorded as measured, including the Effusion pointing-game weakness | `outputs/gradcam/bbox_localization.json` |
| 25 | **8** | `README.md` rewritten around the X-ray research project; legacy kept in §7 and in its own docs | the README must describe *this* project first | — |
| 26 | **9** | this log | an audit trail is part of the deliverable | — |
| 27 | **10** | `docs/TECH_STACK.md` | state what runs, versions included | — |
| 28 | **11** | `docs/REPRODUCIBILITY.md` | every number must be re-derivable | — |
| 29 | **12** | `docs/QA_REPORT.md` (tests, data integrity, guards) | quality evidence, not intent | 140 tests green |
| 30 | **13** | `docs/FINAL_PROJECT_STATUS.md` | final roll-up | — |
| 31 | 14 | `src/research_freeze.py` → `outputs/research_snapshot.json` | the freeze record (config, T, τ, env, git, tests) | one-shot, read-only afterwards |
| 32 | 14 | `src/master_results.py` → `master_results.csv` + `verification_report.json` | 24-row canonical table; docs↔artifact audit (25/25 PASS) | 100 point-estimates match CIs exactly |
| 33 | 14 | `src/uncertainty.py` → `confidence_intervals.csv` | 2000-resample percentile bootstrap, seed 42 | every CI brackets its point estimate |
| 34 | 14 | `src/paper_artifacts.py` → tables 1–8, figures 1–8, `error_analysis.csv`, `region_analysis.json`, `gradcam/` export, `research_summary.json` | paper-ready evidence derived from frozen files | byte-deterministic |
| 35 | 14 | `tests/test_consistency.py` + `tests/test_final_results.py` | consistency + evidence-pack tests | 176 tests green |
| 36 | 15 | `src/paper_build.py` → `outputs/paper/` (evidence map, tables 01–09, figures 01–09, claim audit, reproducibility manifest, number-hygiene) | publication page derived from frozen artifacts; every claim verified | 69/69 evidence, audit all-true |
| 37 | 15 | `tests/test_consistency.py::TestECEImplementationIsStandard` | independent ECE recompute (raw + calibrated) — scientific-correctness check | matches published to 5 dp |
| 38 | 15 | `tests/test_paper_package.py` | 9 tables, 9 figures, evidence/audit/manifest/docs, number hygiene | 14 tests |
| 39 | 15 | prose drafts (`outputs/paper/*.md`) + README/STATUS/snapshot `project_status` | research question → contribution → outline → results → discussion → limitations → titles → abstract → conclusion | PAPER PREPARATION COMPLETE / READY FOR AUTHOR REVIEW |

---

## Decisions worth remembering

| Decision | Rationale |
|---|---|
| Cached logits, not cached metrics | temperature scaling and threshold policies are defined on logits/probabilities; caching final numbers would force re-inference for every variant |
| Guard `fit_split` at runtime + test | a test alone can be skipped; a `ValueError` cannot |
| exp3 refits policies in calibrated space even though the operating points coincide | the experiment is about *procedure*; that it collapses to exp2's confusion matrices is a finding, not a reason to skip it |
| Grad-CAM case list ordered by `(certainty desc, Image Index asc)` | reproducible selection with no seed; two runs agree byte for byte |
| Bbox localization reported on the **full boxed subset**, not on selected cases | 0 of 16 hand-picked cases had a box; a cherry-picked subset would be even worse |
| External phase writes `status.json` only | makes "no data" machine-distinguishable from "no result" |
| Project status is `CORE_RESEARCH: COMPLETE` / `EXTERNAL_VALIDATION: PENDING` | external validity is a *separate* open question from the completed + verified core work |
| Bootstrap CIs resample images, with patient clustering documented | metric unit = image; the patient-level heterogeneity is a stated limitation, not hidden |
| Even a "negative" calibration result (ECE↑) is published as a finding | temperature scaling sharpening moves `mean(p)` away from prevalence; masking it would misreport |
| Legacy ECHO/MRI/fusion untouched | `demo_app.py` and `src/fusion_config.py` hard-depend on the root checkpoint |
| Paper package is derived, never measured | every number in `outputs/paper/` re-reads the frozen artifacts; prose numbers are machine-audited (`check_prose_numbers`) so a drifting draft fails loudly |
