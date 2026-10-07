# Figure and Table Explanation Guide

All items below refer to the authoritative 14-item publication set in `outputs/paper/figures/` and `outputs/paper/tables/`. Their numerical basis is also represented in `outputs/final_results/`. For a figure's exact generated plotting code, see `src/paper_build.py` and `src/paper_artifacts.py`. Presentation language is deliberately limited to what each artifact measures.

## Figures

### Figure 1 — Pipeline / architecture
**File:** `outputs/paper/figures/figure_01_architecture.png`. **Shows:** NIH images and labels → fixed DenseNet121 → raw score → optional temperature transform → threshold policy → test evaluation. **Axes/legend:** schematic, no numeric axes. **Finding:** parameters are fitted on validation, then frozen for test. **Say:** “This diagram separates probability adjustment from the final operating decision; test data are used for evaluation only.” **Do not claim:** the calibration module is a new model or the flow is a clinical workflow.

### Figure 2 — Dataset distribution
**File:** `figure_02_dataset_distribution.png`; split counts from `table_01_dataset.csv`. **Shows:** image counts by patient-level split and positive counts for each label. **Axes:** split categories and counts. **Finding:** train 76,977; validation 16,451; test 15,884; total available 109,312. **Say:** “Patients, rather than individual images, are assigned to one split.” **Caveat:** test Effusion support discrepancy: table label sum 1,996, prediction support 1,997 due documented unreadable-image replacement behavior.

### Figure 3 — ROC curves
**File:** `figure_03_roc.png`; source `outputs/final_results/figures/fig_1_roc.png`. **Axes:** false-positive rate vs true-positive rate, with a threshold sweep. **Legend:** one curve per target; diagonal is chance reference if shown. **Finding:** test AUROC 0.8972 Cardiomegaly and 0.8587 Effusion; DeLong 95% CIs 0.8821–0.9124 and 0.8507–0.8667. **Say:** “This summarizes ranking across cutoffs, not a selected operating point.” **Do not claim:** calibration raises AUROC; the strictly monotone maps preserve ranking.

### Figure 4 — Precision–recall curves
**File:** `figure_04_precision_recall.png`; source `fig_2_pr.png`. **Axes:** recall vs precision; prevalence reference dashed as indicated in source title. **Finding:** test AUPRC 0.3043 Cardiomegaly, 0.4700 Effusion. **Say:** “The positive-class precision/recall tradeoff is useful under imbalance; the baseline level depends on prevalence.” **Do not compare** across datasets without prevalence/protocol context.

### Figure 5 — Reliability diagrams
**File:** `figure_05_calibration.png`; source `fig_3_reliability.png`. **Axes:** mean predicted confidence vs observed event fraction, raw and temperature-scaled. Diagonal is ideal reliability. **Finding:** equal-width ECE (15 bins) increases on test from 0.1061 to 0.1169 for Cardiomegaly and 0.2077 to 0.2292 for Effusion, while Brier/NLL improve. **Say:** “The calibration conclusion depends on metric; the reference ECE worsens.” **Do not claim** a uniformly improved calibration curve.

### Figure 6 — Threshold analysis
**File:** `figure_06_threshold_analysis.png`; source `fig_4_threshold_sweep.png`. **Axes:** operating metric vs threshold; class panels and frozen policy markers. **Finding:** changing threshold moves operating characteristics, with class-specific F1-optimum τ values. **Say:** “The threshold selects the operating point on fixed scores.” **Do not claim** the test curve selected τ; τ came from validation.

### Figure 7 — Confusion matrices
**File:** `figure_07_confusion_matrix.png`; source `fig_5_confusion.png`. **Axes:** actual vs predicted categories for test/arms as captioned. **Legend:** TP/FP/FN/TN. **Finding:** high cutoff reduces false positives and recall; the paired arms show equivalent predictions for monotone-calibrated + corresponding fitted threshold. **Say:** “This makes the precision/recall cost visible.”

### Figure 8 — Error analysis
**File:** `figure_08_error_analysis.png`; source `fig_7_error_analysis.png`. **Panels:** error rate by confidence and confusion strata, test F1-optimal policy. **Finding:** 274/563 Cardiomegaly errors are ≥0.9-confidence; 193/2,268 Effusion errors are ≥0.9. **Say:** “High score does not guarantee correctness, particularly for the rare Cardiomegaly label.” **Do not imply** calibration alone fixes representation or label errors.

### Figure 9 — Grad-CAM
**File:** `figure_09_gradcam.png`; source `fig_8_gradcam.png`. **Panels:** deterministic TP/FP/FN/TN cases, labels and heatmaps. **Finding:** companion 43-case boxed cohort pointing hits 21/43, mean concentration ratio 2.601. **Say:** “These are exploratory visualizations plus a small box-based sanity check.” **Do not claim** clinically validated localization, causal explanation, or proof of correct anatomy.

### Figure 10 — Calibration comparison
**File:** `figure_10_calibration_comparison.png`; artifact `table_13_calibration_comparison.csv`. **Shows:** raw, temperature, logistic calibration on validation and/or metric comparison, per caption. **Finding:** logistic calibration has very low validation ECE/NLL in-sample, whereas temperature's ECE rises. **Say:** “This is a separate validation-fitted extension and susceptible to fitting optimism.”

### Figure 11 — ECE sensitivity
**File:** `figure_11_ece_sensitivity.png`; source `outputs/metrics/ece_sensitivity/ece_sensitivity.csv`. **Axes:** ECE versus bin count / binning strategy. **Finding:** finite-bin ECE is sensitive to definition and occupancy; the declared reference is equal-width, 15 bins. **Say:** “We expose binning dependence rather than treating one ECE number as absolute truth.”

### Figure 12 — Threshold stability
**File:** `figure_12_threshold_stability.png`; source `outputs/metrics/threshold_stability/threshold_stability.csv`. **Shows:** validation patient-bootstrap distributions and intervals by policy/score type. **Finding:** calibrated F1 threshold intervals: Cardiomegaly 0.7860–0.8799, Effusion 0.6434–0.7444; both within ±10% but not ±5% flags relative to first fit. **Say:** “The selected cutoffs vary under patient resampling, although these two F1 cutoffs remain within the broader ±10% band.”

### Figure 13 — Decision policy comparison
**File:** `figure_13_decision_policy.png`; source `decision_policy_analysis.csv`. **Axes:** policy on x, precision/recall/specificity/F1 on y, by label. **Finding:** sensitivity-constrained raises recall and sacrifices precision/specificity; precision-constrained trades recall for precision; F1 is one particular objective. **Say:** “Policy choice encodes an operating preference.” **Do not generalize monotonic tradeoffs as mathematical guarantees for every dataset.

### Figure 14 — Prevalence sensitivity
**File:** `figure_14_prevalence_sensitivity.png`; source `outputs/metrics/prevalence/prevalence_shift.csv`. **Axes:** simulated prevalence vs metric, with frozen scores/threshold. **Finding:** precision/F1/AUPRC shift as negatives are subsampled; recall remains fixed because positives and their predictions are kept. Below-natural prevalence targets are skipped. **Say:** “This is a controlled prevalence-only simulation, not a new cohort or external validation.”

## Tables

### Table 1 — Dataset / split counts
**File:** `table_01_dataset.csv`. Train/val/test images and patients, target-positive counts, totals. Use counts above; describe split as patient-level. State the Effusion support inconsistency openly (1,996 split-table positives vs 1,997 prediction support) and point to `docs/BASELINE.md` and `docs/REPOSITORY_AUDIT.md`.

### Table 2 — Baseline metrics
**File:** `outputs/paper/tables/table_02_baseline.csv`. Rows are AUROC, AUPRC, F1, precision, recall, specificity, accuracy and prevalence for Cardiomegaly and Effusion, plus macro AUROC/AUPRC. It is **not** a model-configuration table. For example, baseline AUROC is 0.8972/0.8587, F1 is 0.2836/0.4676, and macro AUROC/AUPRC are 0.8780/0.3871. Model settings belong to the source and reproducibility manifest, not this numbered table. Current manifest/config augmentation is rotation 10°, brightness/contrast 0.15, while an older `docs/BASELINE.md` says ±20%; flag this doc/config mismatch rather than resolving it silently.

### Table 3 — Main test results
**File:** `table_03_main_results.csv`. Rows A, D and D−A for both labels; columns F1, precision, recall, specificity, AUROC, AUPRC. Cardiomegaly D−A F1 +0.0700; Effusion +0.0191. AUROC/AUPRC equal under monotone score transform. Recall falls at D.

### Table 4 — A/B/C/D ablation
**File:** `table_04_ablation.csv`. Rows for each arm/label; calibration and threshold factors, thresholded metrics and probability metrics. A/B F1 same at 0.5; C/D equal at F1-optimal thresholds; B/D calibration probability scores change ECE/Brier but classification decisions coincide. This makes factor attribution inspectable.

### Table 5 — Threshold policies
**File:** `table_05_thresholds.csv`. Validation thresholds raw/calibrated, test F1 and calibrated precision/recall/specificity. Raw F1-optimal τ 0.9021/0.7699; calibrated τ 0.8666/0.7035. Emphasize validation fit and frozen test application. Values are not clinical cutoffs.

### Table 6 — Calibration metrics
**File:** `table_06_calibration.csv`. Raw vs calibrated NLL, Brier, ECE, MCE, mean confidence. Test NLL/Brier improve but ECE rises. These disagree because they aggregate probability errors differently and ECE bins scores.

### Table 7 — Error analysis
**File:** `table_07_error_analysis.csv`. Per-label error count/rate and high-confidence count/share, plus joint errors and patient concentration. Cardiomegaly 563 / 274 high confidence; Effusion 2,268 / 193; 1,317 patients with errors; maximum per-patient count 60; Jaccard both-wrong 0.1315. Check explanatory column labels carefully: one row encodes “patients_with_errors” with values `4449,1317,60` across columns; read with source JSON for meaning.

### Table 8 — Explainability
**File:** `table_08_explainability.csv`. 16 Grad-CAM cases, target layer, 43-box cohort, 21 pointing hits, mean concentration 2.601, median 2.8726, 38/43 ratio >1, class counts 24 Cardiomegaly/19 Effusion. Exploratory only.

### Table 9 — External status
**File:** `table_09_external_validation_status.csv`. Pending/unavailable, no metrics, paths probed, evaluator implemented. Use as evidence of absence, not performance.

### Table 10 — Patient bootstrap CIs
**File:** `table_10_patient_bootstrap_ci.csv`. Test per-label, variant/policy metrics with 5,000 patient-cluster resamples, 95% intervals, seed 42. Distinguish single-metric intervals from paired arm-difference intervals in Table 11.

### Table 11 — Arm differences
**File:** `table_11_arm_differences.csv`. Paired D−A/other arm deltas and patient-bootstrap CIs. Primary F1 deltas: Cardiomegaly 0.070034 [0.028025, 0.106492], Effusion 0.019090 [0.005024, 0.032553]. Same-patient resampling preserves pairing.

### Table 12 — Threshold stability
**File:** `table_12_threshold_stability.csv`. Bootstrap threshold distribution summaries and ±5/±10% status. The ± flags are relative to the first fit and do not mean 95% of values are “clinically stable.”

### Table 13 — Calibration comparison
**File:** `table_13_calibration_comparison.csv`. Raw/temp/logistic ECE ranges and validation NLL before/after plus logistic parameters. Logistic calibration is a validation-only extension, not a primary A/B/C/D arm. Very low in-sample ECE does not establish generalization.

### Table 14 — Prevalence shift
**File:** `table_14_prevalence_shift.csv`. Target prevalence, feasibility, kept counts, frozen D threshold, confusion metrics and note. Six below-natural cells are infeasible; eight are simulated. Never call these external data.

## Source naming discrepancy

The current publication package has exactly 14 paper figures and 14 paper tables. Older `outputs/final_results/tables/` contains only eight numbered tables and the final-results figure folder has 13 plot images. Use `outputs/paper/figures/figure_01…14` and `outputs/paper/tables/table_01…14` as the current presentation set, with each metric traced to its underlying JSON/CSV in `outputs/metrics/` or `outputs/final_results/`.
