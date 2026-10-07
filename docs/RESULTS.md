# Results — how every number was produced

This document ties the headline results to the artifact that produced them and
the test that guards them. Every number below is machine-derived from the
frozen files listed; nothing is typed in by hand. The automated cross-check is
`tests/test_consistency.py` and `outputs/final_results/verification_report.json`
(currently **25/25 PASS**).

## Primary results (test split, n = 15 884)

Source: `outputs/metrics/baseline/baseline_metrics_test.json` (baseline),
`outputs/metrics/experiments/exp3_thresholds_calibrated.json` (combined arm).

| | Baseline (raw, τ=0.5) | Combined (calibrated, F1-opt τ) |
|---|---|---|
| **Cardiomegaly** AUROC / AUPRC | 0.8972 / 0.3043 | 0.8972 / 0.3043 (invariant) |
| Cardiomegaly F1 / precision / recall / specificity | 0.2836 / 0.1787 / 0.6867 / 0.9153 | **0.3536** / 0.3377 / 0.3711 / 0.9805 |
| **Effusion** AUROC / AUPRC | 0.8587 / 0.4700 | 0.8587 / 0.4700 (invariant) |
| Effusion F1 / precision / recall / specificity | 0.4676 / 0.3320 / 0.7902 / 0.7714 | **0.4866** / 0.4440 / 0.5383 / 0.9031 |

Bootstrap 95% CIs (2 000 resamples, seed 42): e.g. Cardio AUROC
**0.8972 [0.8846, 0.9096]**, Effusion AUROC **0.8587 [0.8518, 0.8655]**,
Cardio combined-arm F1 **0.3536 [0.3025, 0.3744]** — full table in
`outputs/final_results/confidence_intervals.csv`.

*Why CIs are image-based, not patient-based:* metrics are defined per image and
the bootstrap resamples the same unit; patient-level clustering of errors is
documented separately (error analysis) and noted as a limitation, not hidden.

## Calibration (post-hoc temperature scaling)

Source: `exp1_calibration.json`. Temperatures fit on validation NLL.

* NLL falls (Cardio 0.230 → 0.226; Effusion 0.482 → 0.469) and Brier falls.
* Binned ECE **rises** (Cardio 0.106 → 0.117; Effusion 0.208 → 0.229) — an
  honest, explained outcome: `ECE ≈ mean(p) − prevalence` here, and sharpening
  moves mean confidence away from prevalence. Reported, not masked.

## Error analysis (test, combined arm, F1-opt policy)

Source: `outputs/metrics/error_analysis/error_analysis_test.json`;
per-file CSV in `outputs/final_results/error_analysis.csv`.

* Cardiomegaly error rate 3.5 % (563 errors); **274 = 48.7 % are
  high-confidence (p ≥ 0.90)** — a confident-wrong failure mode.
  Strata: TP 154 · FP 302 · FN 261 · TN 15 167.
* Effusion error rate 14.3 % (2 268); only 8.5 % high-confidence —
  uncertainty is concentrated where the model is honestly unsure.
* Cross-label: 585 images wrong on both labels (Jaccard of error sets 0.131).
* Patient concentration: 1 317 patients hold all 4 449 error images; one
  patient accounts for 60 errors.

## Explainability (fixed 43-image cohort, frozen)

Source: `outputs/gradcam/{gradcam_summary,bbox_localization}.json`; exports in
`outputs/final_results/gradcam/` and `region_analysis.json`.

* 16 overlays (2 per stratum × 2 labels), target layer
  `encoder.features.denseblock4`; case selection is deterministic (certainty
  desc, image index asc — no RNG).
* Pointing game (argmax inside NIH box): **21/43 = 48.8 %** (Cardio 19/24,
  Effusion 2/19).
* Concentration ratio (heat in box ÷ area share): mean **2.60**, median 2.87,
  **38/43 > 1**.

Scope note (kept in `region_analysis.json`): this is a **localization sanity
check on a small fixed cohort**, not a diagnostic localization validation.

## Training curves

Source: `outputs/logs/metrics.csv` (10 epochs). Best validation mean AUROC
**0.8678 at epoch 8**; the checkpoint is hash-frozen
(`35965f610c8b…`) and never retrained.

## External cohort

**Not measured.** Source: `outputs/metrics/external/status.json`
(`no_metrics_were_computed: true`). The project status is
**`CORE_RESEARCH: COMPLETE` · `EXTERNAL_VALIDATION: PENDING`** — see
`docs/STATUS.md`.

## The single evidence document

`outputs/final_results/research_summary.json` assembles the research question,
baseline/proposed descriptions, the best observed effect sizes, the
calibration anomaly, the error-analysis finding, the external status, and the
full limitation list — all derived, none invented.