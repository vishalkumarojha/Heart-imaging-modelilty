# Draft — Results

All numbers below are canonical values from
`outputs/final_results/master_results.csv` and the frozen experiment artifacts;
each is cross-checked in `outputs/paper/paper_evidence_map.json` and
`outputs/paper/claim_audit.csv`. Thresholds were fit on the NIH validation
split only; every number in this section is measured on the held-out NIH test
split (15,884 images) unless stated otherwise.

## Baseline

A frozen DenseNet121 (ImageNet-1K pretrained) trained on the NIH train split
and thresholded at the default 0.50 produced the following test metrics:

| class        | AUROC | AUPRC | F1    | precision | recall | specificity |
|--------------|-------|-------|-------|-----------|--------|-------------|
| Cardiomegaly | 0.8972| 0.3043| 0.2836| 0.1787    | 0.6867 | 0.9153      |
| Effusion     | 0.8587| 0.4700| 0.4676| 0.3320    | 0.7902 | 0.7714      |

Macro AUROC = 0.8780; macro AUPRC = 0.3871
(`outputs/paper/tables/table_02_baseline.csv`).

## Calibration

Per-label temperature scaling was fit on the validation split
(Cardiomegaly T = 1.187, Effusion T = 1.398; full precision 1.1865 / 1.3979)
and applied to the frozen test logits. NLL and Brier improved for both labels;
ECE increased in this implementation.

| class        | NLL raw → cal | Brier raw → cal | ECE raw → cal |
|--------------|---------------|-----------------|---------------|
| Cardiomegaly | 0.2297 → 0.2262 | 0.0681 → 0.0678 | 0.1061 → 0.1169 |
| Effusion     | 0.4819 → 0.4690 | 0.1561 → 0.1533 | 0.2077 → 0.2292 |

ECE uses equal-width binning with 15 bins; empty bins contribute 0. AUROC and
AUPRC are invariant to temperature scaling. The ECE increase is reported
without modification (`outputs/paper/tables/table_06_calibration.csv`).

## Threshold Policies

All five policies were frozen on the validation split and re-applied to test.
F1-optimal validation thresholds: Cardiomegaly τ = 0.9021, Effusion τ = 0.7699
(raw probabilities). Applying the validation-frozen F1-optimal policy
(combined temperature-scaling + threshold arm, arm D) on the test split gave:

| class        | F1    | precision | recall | specificity |
|--------------|-------|-----------|--------|-------------|
| Cardiomegaly | 0.3536| 0.3377    | 0.3711 | 0.9805      |
| Effusion     | 0.4866| 0.4440    | 0.5383 | 0.9031      |

(`outputs/paper/tables/table_03_main_results.csv`,
`outputs/paper/tables/table_05_thresholds.csv`).)

## Ablation

Four arms isolate the contribution of each component (test split):

| arm | name                     | calibration | threshold        |
|-----|--------------------------|-------------|------------------|
| A   | baseline                 | raw         | fixed 0.50       |
| B   | calibration only         | calibrated  | fixed 0.50       |
| C   | threshold only           | raw         | F1-optimal       |
| D   | calibration + threshold  | calibrated  | F1-optimal       |

Because the sigmoid is monotone, temperature scaling (T > 1) never changes the
sign of the logit, so the fixed-0.50 confusion matrices of arm A and arm B are
identical; F1 is unchanged (Cardiomegaly 0.2836; Effusion 0.4676). Applying
the F1-optimal policy (arms C and D) raises test F1 to 0.3536 / 0.4866 and
precision to 0.3377 / 0.4440. The F1/precision gain is therefore attributable
primarily to threshold selection, not to temperature scaling.
(`outputs/paper/tables/table_04_ablation.csv`.)

## Error Analysis

At the F1-optimal operating point on the test split:

- Cardiomegaly: 563 errors, of which 274 carry confidence ≥ 0.9 — 48.7% of the
  label's errors are high-confidence.
- Effusion: 2,268 errors, of which 8.5% are high-confidence.
- Cross-label: error-set Jaccard = 0.131; 4,449 images are misclassified by at
  least one label across 1,317 patients; the worst single patient accumulates
  60 error images.

High-confidence errors (conf ≥ 0.9) are substantial for the rarer class,
Cardiomegaly, and are concentrated in relatively few patients
(`outputs/paper/tables/table_07_error_analysis.csv`,
`outputs/paper/figures/figure_08_error_analysis.png`).

## Explainability

16 deterministic Grad-CAM cases were selected across TP/FP/FN/TN strata at the
F1-optimal calibrated operating point (target layer `encoder.features.
denseblock4`). An exploratory bounding-box sanity cohort (43 chest X-rays) gave
a pointing-game accuracy of 21/43, a mean concentration ratio of 2.60, and
38/43 cases with ratio above 1 (Cardiomegaly cohort share 19/24, Effusion 2/19).
This is a descriptive sanity check on the frozen model's gradient-based
attention; it is **not** a clinical localization validation
(`outputs/paper/tables/table_08_explainability.csv`,
`outputs/paper/figures/figure_09_gradcam.png`).

## External Validation

External validation was not completed because the required external cohort was
unavailable.

No external performance metric, prediction file, or figure was generated
(`outputs/paper/tables/table_09_external_validation_status.csv`).