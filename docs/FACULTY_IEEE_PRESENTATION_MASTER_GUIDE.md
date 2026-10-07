# Faculty / IEEE Presentation Master Guide

**Purpose:** one rehearsal-ready account of the fixed NIH ChestX-ray14 calibration and decision-policy study. This is documentation, not a change to the frozen research. Numerical claims should be checked against the linked artifacts before a talk.

## Evidence labels

- **VERIFIED FROM REPOSITORY** — source code/configuration or checked-in documentation.
- **VERIFIED FROM ARTIFACT** — machine-generated frozen result or manifest.
- **VERIFIED FROM LITERATURE** — claim supported by cited paper.
- **INTERPRETATION** — bounded reading of the evidence.
- **PRESENTATION RECOMMENDATION** — suggested framing, not an experimental result.
- **UNKNOWN / NOT VERIFIED** — repository does not establish the fact.

The current machine-readable paper evidence map links 111 claims to sources (`outputs/paper/paper_evidence_map.json`); `outputs/paper/claim_audit.csv` records claim verification. These help trace the existing package but do not replace checking the source artifact.

## 1. Executive Summary

**VERIFIED FROM REPOSITORY/ARTIFACT:** A fixed ImageNet-initialized DenseNet121 with two independent sigmoid outputs was evaluated on an available NIH ChestX-ray14 image set. A patient-level train/validation/test partition was frozen using seed 42. Temperature parameters and per-label operating thresholds were fitted on validation only, then applied without refitting to cached held-out test scores.

The experiment asks how calibration and threshold choice affect probability and operating-point metrics. The test AUROC was 0.8972 for Cardiomegaly and 0.8587 for Effusion. At the selected F1 operating point, D−A F1 was +0.070034 (95% patient-cluster bootstrap CI 0.028025–0.106492; paired permutation p≈0.0002, Holm 0.0004) for Cardiomegaly and +0.019090 (0.005024–0.032553; p≈0.0262, Holm ≈0.0262) for Effusion. This gain has a visible tradeoff: recall falls from 0.6867 to 0.3711 and 0.7902 to 0.5383, respectively, while precision/specificity rise.

Temperature scaling improved test NLL/Brier but increased reference equal-width 15-bin ECE. Since the map is strictly monotonic, AUROC/AUPRC rankings are unchanged. The four-arm results show classification changes arise from the selected threshold, not the calibration map itself when validation thresholds are transformed consistently. This does not prove a general law or clinical benefit.

**External validation: PENDING/BLOCKED.** The CheXpert evaluator exists; no external cohort was available and no external metric was computed. This is a research demonstration, not clinical software.

## 2. Project Identity

**Current research identity (repository + frozen paper artifacts):** “Calibration-Aware Decision Policies for Chest X-Ray Classification.” The project is a retrospective, single-dataset, two-label evaluation around one frozen DenseNet121 checkpoint. The dashboard is an evidence viewer with a live inference demonstration.

**Repository-history conflict:** root README/phase summaries preserve an earlier multimodal project covering X-ray, echo, MRI, and fusion. Newer `docs/`, `outputs/paper/`, `outputs/final_results/`, and `research_demo_app.py` describe the final calibration-focused study. For this viva, use the final artifacts and current dashboard as the authoritative research scope; explain legacy modality code as retained history, not part of the final claim. See `docs/LEGACY_MODALITIES.md`, `docs/PROJECT_SCOPE.md`, and commit history.

## 3. Research Question

> Can calibration-aware, class-specific decision policies improve the operating characteristics of a fixed DenseNet121 multi-label chest X-ray classifier under class imbalance?

Operational answer is conditional: under this fixed NIH split/model, validation-frozen F1 thresholding improved test F1; temperature scaling had mixed probability-metric effects and did not independently alter the corresponding thresholded classes. It does not establish clinical utility, external generalization, or superiority for other models/datasets. Primary prose: `outputs/paper/research_question.md`.

## 4. Problem and Motivation

A model score has at least three separate evaluation dimensions:

1. **Discrimination/ranking:** are positives scored above negatives? AUROC/AUPRC summarize ranking across cutoffs.
2. **Probability quality:** do numeric probabilities align with event frequency? NLL, Brier, ECE, reliability diagrams address different aspects.
3. **Operating point:** after selecting a threshold, what are precision, recall, F1, specificity, and confusion counts?

A strong ranking score does not imply a useful F1 at threshold 0.50. Class imbalance affects precision and makes positive-class precision–recall behavior important. Threshold selection is an operating choice, not retraining. **Presentation recommendation:** explain this three-part distinction before showing A/B/C/D.

## 5. Dataset

**VERIFIED FROM ARTIFACT:** NIH ChestX-ray14 metadata and available radiographs, using labels Cardiomegaly and Effusion. `data/raw/Data_Entry_2017.csv` supplies `Image Index`, `Patient ID`, and `Finding Labels`; `src/dataset.py` converts pipe-delimited label names to binary indicators. The paper package records 112,120 metadata images but only 109,312 available on disk; 2,808 are missing from the partial image collection. Relevant sources: `src/config.py`, `src/dataset.py`, `data/raw/README_CHESTXRAY.pdf`, `outputs/paper/tables/table_01_dataset.csv`, `docs/REPOSITORY_AUDIT.md`.

| Split | Images | Patients | Cardiomegaly positive | Effusion positive in split table |
|---|---:|---:|---:|---:|
| Train | 76,977 | 20,797 | 1,880 | 9,001 |
| Validation | 16,451 | 4,468 | 392 | 1,966 |
| Test | 15,884 | 4,455 | 415 | 1,996 |
| Total | 109,312 | 29,720 | 2,687 | 12,963 |

**Inconsistency to disclose:** the split-table/raw target total shows 1,996 test Effusion positives, but test prediction/statistical artifacts show support 1,997 (and 13,887 negatives). `docs/BASELINE.md` documents a truncated PNG `00029705_000.png` and deterministic dataset-loader substitute behavior as the cause. Do not hide or silently “fix” this frozen-input defect; for model metrics cite the prediction artifact's support and explain the exception. This discrepancy is a known limitation in data handling.

**Why these two labels?** The target list in `src/config.py` is explicitly only Cardiomegaly/Effusion. The repository does not verify the scientific rationale for selecting them over the other 12 labels. State that this is the chosen project scope, not a proven optimal label set. No evidence shows an ablation against all 14 labels.

## 6. Data Split

**VERIFIED FROM REPOSITORY:** 70/15/15 patient-level, seed 42. `src/dataset.assign_splits` groups by `Patient ID`; it can use official NIH split lists if found, but local reproducibility documentation says the official train/test lists are absent and the frozen split is custom patient-level. The checked split has zero patient overlap across train/validation/test.

Faculty explanation: “A patient can have several X-rays. If individual images were randomized separately, one patient's anatomy or acquisition pattern might appear in both training and test data. The model could then be tested on a familiar person. We assigned all images for a patient to only one split and measured zero overlap.” This blocks this specific leakage route; it does not remove all sources of bias.

The 70/15/15 choice and seed are verified constants. Why 70/15/15 or why the seed value 42 specifically were selected is **not verified from repository**; present them as reproducibility design choices, not scientific discoveries. The dataset split has no documented stratification guarantee, so do not imply exact prevalence balancing.

## 7. Model Architecture

**Simple explanation:** DenseNet121 extracts visual features; a linear head turns the pooled features into two scores, one for each finding. Both findings can be positive independently, so this is multi-label binary classification rather than a mutually exclusive two-class choice.

**Technical source:** `src/model.py` uses torchvision `densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)`, global average pooling to 1,024 features, and a `Linear(1024,2)` head. It returns raw logits. `src/config.py` fixes label order `[Cardiomegaly, Effusion]`. The stored baseline checkpoint SHA-256 is `35965f610c8b578b3ad52e9d7d06a1b3054948df6df52e80aa02f2b81803d68c` (`outputs/checkpoints/baseline/baseline_manifest.json`).

Dense connectivity is the established DenseNet design (Huang et al., CVPR 2017), and DenseNet121 has prior chest-radiograph use (CheXNet, Rajpurkar et al.). The project does not prove this architecture is best. Why these two labels and this architecture were selected is not experimentally compared here.

## 8. Training

**VERIFIED FROM REPOSITORY:** 224×224 RGB, ImageNet mean/std; augmentations only in training: rotation ±10°, brightness 0.15, contrast 0.15, no horizontal flip (laterality concern in config comment). Loss `BCEWithLogitsLoss` with class `pos_weight` derived from train prevalence; Adam, learning rate 1e-4 for five epochs with earlier backbone frozen except final dense block/norm and classifier, then fully unfrozen at 1e-5 for five epochs; batch size 32; weight decay 0; AMP enabled where CUDA available; best checkpoint chosen by highest mean validation AUROC, epoch 8 per saved documentation. All configured epochs run; no early-stopping callback or learning-rate scheduler is implemented.

**Conflict:** `docs/BASELINE.md` describes brightness/contrast augmentation as ±20%, while current `src/config.py` says 0.15. The baseline manifest also records 0.15. Prefer the frozen manifest/config for technical description, flag the stale doc mismatch, and do not infer the exact historical transform beyond the manifest/source evidence. No training rerun was performed for this documentation audit.

The seed is 42 across the setup, but exact bitwise reproducibility across hardware is not guaranteed: `docs/BASELINE.md` documents small AMP threshold-boundary differences relative to a historical run. All experiments use the same cached prediction table, limiting internal inconsistency.

## 9. Baseline Results

Held-out test metrics, threshold 0.50 (`outputs/paper/tables/table_02_baseline.csv`, `outputs/metrics/baseline/baseline_metrics_test.json`):

| Label | AUROC | AUPRC | F1 | Precision | Recall | Specificity | Accuracy | Prevalence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 0.8972 | 0.3043 | 0.2836 | 0.1787 | 0.6867 | 0.9153 | 0.9093 | 0.0261 |
| Effusion | 0.8587 | 0.4700 | 0.4676 | 0.3320 | 0.7902 | 0.7714 | 0.7737 | 0.1257 |
| Macro | 0.8780 | 0.3871 | — | — | — | — | — | — |

AUROC: probability a randomly selected positive ranks above a randomly selected negative, with tie convention; it summarizes rank discrimination over cutoffs. AUPRC summarizes precision/recall and is prevalence-sensitive, useful for imbalanced positives. F1 is the harmonic mean of precision and recall at one chosen cutoff. Therefore good AUROC can coexist with modest F1/precision at 0.50; one measures ranking, the other a single operating point.

## 10. Calibration

Calibration asks whether probability values match observed frequency. A model can rank cases correctly but assign over/underconfident values. Calibration is not the same as classification accuracy or AUROC.

`src/calibration.py` fits one scalar per label by minimizing validation NLL over T∈[0.05,50]; `src/ui_data.py` applies `sigmoid(z/T)`. It is validation-only and test is untouched during fitting. Parameters: Cardiomegaly T=1.1865413532791096; Effusion T=1.3979213336039338 (`outputs/metrics/calibration/temperature_scalers.json`). T>1 softens logits. The transform is strictly monotone, so score ranking and AUROC/AUPRC are preserved; after validation thresholds are independently fit on each score scale, corresponding operating classifications also match.

**Actual test calibration metrics** (`outputs/paper/tables/table_06_calibration.csv`):

| Label | NLL raw→temp | Brier raw→temp | ECE raw→temp (15 equal-width bins) | MCE raw→temp |
|---|---|---|---|---|
| Cardiomegaly | 0.2297→0.2262 | 0.0681→0.0678 | 0.1061→0.1169 | 0.6926→0.6783 |
| Effusion | 0.4819→0.4690 | 0.1561→0.1533 | 0.2077→0.2292 | 0.5115→0.4357 |

Thus NLL/Brier improve, ECE worsens, MCE improves. NLL heavily penalizes confident errors; Brier is mean squared probability error; ECE is a binned weighted gap and changes with bins; MCE uses worst occupied-bin gap. Do not state “calibration improved” without naming the metric.

The **validation-fit/reference report** is a separate split (`outputs/metrics/calibration/calibration_report_val.json`): Cardiomegaly NLL 0.233691→0.230191, Brier 0.070498→0.069946, ECE 0.108645→0.119448; Effusion NLL 0.488918→0.472142, Brier 0.158548→0.154762, ECE 0.214534→0.235131.

## 11. Threshold Selection

Probability estimation produces a score. Operating-point selection chooses τ and maps score ≥τ to positive. The 0.50 baseline is a reference, not automatically optimal under class imbalance.

`src/thresholds.py` implements:

- **Fixed:** τ=0.50, no fit objective.
- **F1-optimal:** maximize validation F1, balancing precision and recall as a descriptive summary.
- **Youden:** maximize sensitivity+specificity−1, balancing those two rates by the criterion.
- **Sensitivity-constrained:** achieve target sensitivity 0.90 where feasible, choose the largest qualifying threshold (then specificity among such cutoffs); if infeasible, recorded fallback/constraint status must be consulted.
- **Precision-constrained:** achieve precision ≥0.50 where feasible and select highest recall among qualifying thresholds; if infeasible, fallback to maximum achievable precision.

Thresholding higher often reduces positive calls and can raise precision while lowering recall, but it is not a universal strict law for every finite sample. Policy objectives are not clinical utility functions. API refuses fitting on test/external splits. All selected thresholds are label-specific and validation-frozen.

## 12. A/B/C/D Experimental Design

| Arm | Probability | Threshold policy | Question |
|---|---|---|---|
| A | Raw | Fixed 0.50 | Baseline reference |
| B | Temperature-scaled | Fixed 0.50 | Calibration with same nominal cutoff |
| C | Raw | Validation F1-optimal | Threshold change without calibration |
| D | Temperature-scaled | Validation F1-optimal | Combined calibration + threshold policy |

Diagram: `A raw → .50` and `B logits/T → .50`; separately `C raw → τ_val,F1` and `D logits/T → τ_val,F1,cal`. Each label has its own τ. One frozen model and cached predictions are used; nothing is retrained.

A vs B probes calibration under fixed .50. C vs D compares score scales at their own fitted F1 thresholds. A vs C / B vs D probe threshold-policy changes. D vs A is the complete pipeline comparison and primary endpoint. Because the transform is monotone and thresholds are correspondingly refitted on validation, C and D classifications match; A and B at 0.50 match because logit sign does not change. This exact study cannot support a general causal claim beyond its controlled frozen setup.

## 13. Statistical Inference

Primary test endpoints are D−A F1. `outputs/final_results/statistical_report.json`, `arm_differences.csv`, and `outputs/metrics/patient_stats/paired_tests.json` record paired analyses. Bootstrap: 5,000 patient-cluster resamples, 95%, seed 42; arms are paired on the same patient sample. Paired permutation: 5,000 flips at patient level with validation thresholds frozen. Holm correction applies to two primary label-specific D−A F1 tests.

| Label | D−A F1 | Patient bootstrap 95% CI | Permutation p | Holm p |
|---|---:|---:|---:|---:|
| Cardiomegaly | +0.070034 | [0.028025, 0.106492] | 0.000200 | 0.0004 |
| Effusion | +0.019090 | [0.005024, 0.032553] | 0.026195 | 0.026195 |

Both intervals exclude zero, meaning the observed paired difference is positive over these resamples; it does not prove causality, clinical relevance, or population generalization. Statistical significance is not clinical significance. Patient resampling recognizes within-patient dependence; image-level resampling treats repeated images as independent and can understate uncertainty. Older `docs/` entries describe a 2,000-image bootstrap; the current primary endpoint upgrade is 5,000 patient-cluster bootstrap samples. For raw F1-optimal Cardiomegaly, image-bootstrap F1 width is 0.066443 [0.320589,0.387032], compared with patient-bootstrap width 0.136103 [0.281606,0.417709], wider by 0.069660. Effusion width is 0.030108 [0.470910,0.501018] versus 0.058972 [0.456294,0.515266], wider by 0.028864. These are unpaired per-arm F1 intervals; primary inference is the paired D−A interval in Table 11.

DeLong intervals are for test AUROC: Cardiomegaly 0.897229 [0.882078,0.912380], Effusion 0.858686 [0.850660,0.866712] (`outputs/metrics/patient_stats/delong_auroc.json`). Calibration pairwise AUC comparison is degenerate because monotone transforms produce identical ranking; NaN p-values are not missing evidence of improvement.

## 14. Threshold Stability

`outputs/metrics/threshold_stability/threshold_stability.json` and `outputs/paper/tables/table_12_threshold_stability.csv` use 2,000 validation patient-level bootstrap samples, seed 42. A vectorized reproduction/equality check is recorded by the artifact/test suite. For calibrated F1 policy, Cardiomegaly first fit 0.8666, 95% interval [0.7860,0.8799], width 0.0939; Effusion first fit 0.7035, interval [0.6434,0.7444], width 0.1009. Neither is within ±5%; both are within ±10% by artifact flags. This asks whether the selected operating point would be approximately similar under validation resampling; it does not claim perfect or clinical robustness.

## 15. Alternative Calibration

The separate logistic/Platt-style map is `p=σ(a·z+b)`, with constrained positive slope. Validation-only parameters: Cardiomegaly a=0.5441,b=−2.8079; Effusion a=0.7474,b=−1.9919. Validation NLL: 0.23369→0.08453 and 0.48892→0.26496; validation Brier 0.020428 and 0.079667, and report ECE 0.002390 and 0.008289. Because this is an extension analysis and fit/evaluated on the same validation split, the very low validation metrics are not held-out evidence. Positive slope makes it monotone and preserves ranking. It is explicitly **not part of primary A/B/C/D**. `table_13_calibration_comparison.csv` reports another 15-bin ECE reference for logistic (0.0029 / 0.0136) and grid ranges; cite a named file rather than reconciling these values.

## 16. ECE Sensitivity

ECE aggregates absolute accuracy-confidence gaps across bins with weights n_b/N. Repository config checks equal-width/equal-frequency and bin counts 10,15,20 (`outputs/metrics/ece_sensitivity/ece_sensitivity.csv/json`). Equal-width bins can leave sparse/empty bins for imbalanced probability distributions; equal-frequency bins partition by score quantiles. The result depends on binning and finite sample. Primary reference remains equal-width/15 bins; do not cherry-pick the smallest ECE.

## 17. Decision Policies

Five policies and their objectives are listed in §11. Test comparisons are in `outputs/final_results/decision_policy_analysis.csv` and Table 5. Examples: Cardiomegaly sensitivity-constrained calibrated policy has recall 0.9205 and precision 0.0689; precision-constrained has recall 0.1687, precision 0.5303. Effusion equivalents: 0.8908/0.2682 and 0.4271/0.5157. These quantify the chosen policy tradeoff, not a universally best operating policy.

## 18. Prevalence Sensitivity

`src/prevalence_shift.py` resamples only negative rows from frozen test predictions with deterministic seed 42, retaining all positives and frozen D probabilities/thresholds. The natural test prevalence is 2.61% Cardiomegaly and 12.57% Effusion. Targets below natural prevalence are infeasible because negatives-only subsampling can only increase prevalence; the table skips six such cells. Eight higher/equal target cells are simulated. At fixed positive cases/predictions, recall remains pinned; fewer retained negatives lower false-positive count, often increasing precision/F1/AUPRC. These are prevalence-shift simulations, **not external cohorts**; covariate shift is not modeled.

## 19. Error Analysis

At the calibrated F1-optimal test policy (`outputs/paper/tables/table_07_error_analysis.csv`): Cardiomegaly 563 errors, 274 high-confidence (≥0.9), 48.7% of errors; Effusion 2,268 errors, 193 high-confidence, 8.5%. Errors across either label occur on 4,449 images across 1,317 patients; worst patient has 60 error images; both-wrong Jaccard 0.1315. A high score does not mean correct. Calibration changes score scale; it cannot repair missing visual features, label noise, shortcut features, or ambiguous findings. Patient concentration is descriptive and does not establish cause.

## 20. Grad-CAM

`src/gradcam.py` computes gradient-weighted activation from `encoder.features.denseblock4`. A deterministic 16-case TP/FP/FN/TN set supports qualitative review. In a fixed cohort of 43 test images with NIH boxes, 21/43 pointing-game hits (0.4884), mean concentration ratio 2.601, median 2.8726, and 38/43 ratios >1 were reported. The ratio compares heat mass inside a box with its image-area fraction. This is exploratory qualitative analysis plus a descriptive boxed subset; it cannot prove causal feature use or clinically validated localization. A heatmap is not a diagnosis or an explanation guaranteed faithful in the clinical sense.

## 21. External Validation

Status artifact: `outputs/metrics/external/status.json`, Table 9. CheXpert loader/evaluator is implemented; expected `valid.csv`/images were unavailable in the repository and probed locations. `no_metrics_were_computed=true`. Answer to faculty: “We did not complete external validation because the CheXpert cohort was unavailable. We report pending status and no external metrics. The next step is an authorized independent cohort with prespecified label mapping and frozen model, calibration, and thresholds.” Do not describe prevalence simulation as external validation.

## 22. Literature Review

See the full comparison table and safe language in [Literature Positioning](LITERATURE_POSITIONING.md). Key roles:

- Wang et al. 2017: NIH ChestX-ray8 dataset and weakly supervised chest-X-ray context.
- Rajpurkar et al. 2017 CheXNet: DenseNet121 and ChestX-ray14 multi-label precedent.
- Huang et al. 2017: DenseNet architecture.
- Guo et al. 2017: temperature scaling methodology.
- Rajaraman, Ganesan & Antani 2022: closest prior work on calibration/threshold interaction in class-imbalanced medical images.
- Selvaraju et al. 2017: Grad-CAM.
- DeLong et al. 1988 / Holm 1979: statistical methods.

Included dataset documentation includes NIH PDF/readme/FAQ/log files. A repository-wide systematic bibliography search is not represented as a systematic review. External links are cited in `docs/LITERATURE_POSITIONING.md`.

## 23. Rajaraman et al. Comparison

PLOS full text: [DOI article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262838), [PMC text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8794113/). Their work examines chest X-rays and fundus images, multiple backbones and imbalance levels, calibration methods, and default 0.5 versus PR-optimal cutoffs; reports calibration-associated gains at 0.5 that are not significant at the PR-guided threshold. Exact study-specific details should be quoted from the linked full text, not reconstructed from secondary summaries.

**Overlap:** same broad question of calibration and operating threshold in imbalanced medical images, including chest X-rays. **Difference:** present repository fixes one DenseNet121, two NIH labels, one patient split and decomposes raw/calibrated × fixed/F1-optimal with patient-cluster inference, threshold stability, ECE sensitivity, policy and prevalence sensitivity, and artifact traceability. It is incremental, controlled and context-specific—not a new method or claim that calibration is ineffective universally.

## 24. Research Gap

**INTERPRETATION:** this repository provides a more explicit decomposition and patient-level uncertainty audit for a fixed NIH two-label classifier than a single “calibration improved performance” statement would provide. It does not establish that the research literature lacks such analyses; a systematic review is not verified. Safer framing is “a focused, reproducible empirical analysis” rather than “first.”

## 25. Defensible Contribution

**A — one sentence:** A controlled, retrospective study decomposes probability calibration and validation-frozen class-specific threshold effects on a fixed two-label DenseNet121 chest-X-ray model, reporting patient-level uncertainty and sensitivity analyses.

**B — three elements:** (1) frozen score model with validation-only calibration and cutoffs; (2) A/B/C/D decision decomposition and patient-paired inference; (3) transparent analysis of calibration metric dependence, threshold stability, operating policy, errors, and missing external evidence.

**C — 30 seconds:** “We held the DenseNet121 model fixed and asked how post-hoc temperature scaling and class-specific thresholds affect its probabilities and its test operating point. The key distinction is that ranking, calibration metrics, and thresholded F1 are not the same. On this NIH split, the validation-frozen F1 thresholds increased measured F1, while temperature scaling improved NLL and Brier but worsened the reference ECE. Patient-level uncertainty is reported, and external validation remains pending.”

**D — 60 seconds:** “The contribution is not a new network or calibration algorithm. We froze a two-label DenseNet121 on a patient-level NIH ChestX-ray14 split, fit temperature and thresholds on validation only, and evaluated the held-out test predictions. The four arms separate raw versus calibrated scores from fixed versus F1-optimal decisions. The D−A F1 change was positive for both labels, but recall dropped as precision and specificity rose. Since temperature scaling is monotone, it does not improve ranking; the decision gain is about the selected operating point. We add paired patient-cluster intervals/tests, threshold stability, ECE sensitivity and policy/prevalence analyses. The study is single-dataset and retrospective; no external or clinical validation is claimed.”

**E — paper paragraph:** This study provides a controlled, artifact-traceable evaluation of probability calibration and operating-threshold selection for a frozen DenseNet121 multi-label chest radiograph classifier targeting Cardiomegaly and Effusion. Per-label temperatures and decision thresholds are estimated using validation data only and evaluated on a held-out patient-level test split through a four-arm decomposition. Paired patient-cluster resampling and permutation inference quantify uncertainty in prespecified F1 differences. Additional analyses characterize threshold stability, calibration-metric/binning sensitivity, alternate calibration, operating-policy tradeoffs, prevalence-only resampling, errors, and exploratory Grad-CAM. Findings are conditional on this single dataset/split; external validation remains pending.

**F — explicitly not claimed:** novel architecture/algorithm/dataset; first calibration-threshold study; universal calibration or threshold law; state-of-the-art; causal clinical benefit; external validity; clinical diagnosis/use.

### Design decision defense (choice → reason/evidence → consequence → without it → viva line)

| Choice | Why / evidence | Technical consequence | Without it | Viva wording |
|---|---|---|---|---|
| NIH ChestX-ray14 | Dataset present and metadata/images available; a scientific preference is not documented. | One public, retrospective cohort. | A different cohort would change scope and require new data handling. | “We used the repository’s NIH cohort; this does not prove it is representative.” |
| Two labels | Frozen `TARGET_LABELS`; selection rationale beyond scope is unknown. | Two-output multi-label task. | More labels require additional analyses and frozen targets. | “These are the two prespecified project targets; selection optimality was not tested.” |
| DenseNet121 | Frozen baseline and established architecture. | All arms share same representation. | Changing model would confound policy analysis. | “We fixed it to isolate post-hoc effects.” |
| ImageNet initialization | Explicit torchvision weights in code/manifest. | Transfer-initialized weights. | Random initialization would be another experiment; not compared. | “This was the configured initialization; benefit is not ablated here.” |
| 224×224 | Configured model input/evaluation transform. | Resized tensor input. | Another resolution could alter scores and compute. | “It is the frozen input size, not an empirically optimized resolution.” |
| Patient split | Repeated images per patient; zero patient overlap verified. | Grouped split protects against patient overlap. | Image-level random split can expose same-patient images across train/test. | “All images from one patient stay together.” |
| 70/15/15 | Frozen config and split artifact; why this exact ratio is not verified. | Counts support train, fit and held-out evaluation. | Other proportions alter sample sizes. | “The ratio is a reproducibility choice, not a claimed optimum.” |
| Seed 42 | Frozen config/manifest. | Repeatable split/bootstrap RNG. | Different seed creates different partition. | “42 is a stored reproducibility constant; its numeric choice is not scientific.” |
| Validation-only fitting | Prevents test outcome leakage; fit guards enforce it. | T and τ frozen before test scoring. | Test-tuned parameters bias evaluation. | “The test set evaluates; it does not select.” |
| Temperature scaling | Simple one-parameter post-hoc method; established by Guo et al. | Monotone rescaling, ranking preserved. | No calibrated score arm to assess probability metrics. | “We chose a known compact transform; this is not a new algorithm.” |
| Per-label thresholds | Separate Bernoulli targets have separate score distributions and frozen policies. | One τ per output. | A shared cutoff would force a common operating rule. | “Each label's threshold is fitted separately on validation.” |
| Fixed 0.50 | Common baseline reference and arm A definition. | Reference operating point. | No standard comparator for A/B. | “It is a baseline, not an assumed optimum.” |
| F1 optimization | F1 is the prespecified study operating summary. | Balances precision/recall without clinical utility weights. | Other policy would answer a different question. | “F1 is descriptive here, not a clinical utility function.” |
| Patient bootstrap | Images cluster within patient; code samples patient contributions. | Wider, dependence-aware intervals. | Image bootstrap treats correlated images as independent. | “We resample the unit that can contribute multiple images.” |
| Paired permutation | Same cases evaluated by both arms; permutation keeps pairing and freezes τ. | Tests within-case/patient arm contrast. | Unpaired test wastes matching and misstates design. | “We compare paired decisions under fixed validation thresholds.” |
| Holm correction | Two primary label-specific D−A F1 hypotheses. | Family-wise adjustment over the declared pair. | Multiple primary tests inflate false-positive chance. | “Holm adjusts the two planned primary tests.” |
| DeLong | AUROC uncertainty via placement-value covariance. | Test-set AUC SE/CI; monotone pair test degenerate. | No named AUROC interval procedure. | “DeLong is for AUC uncertainty, not F1.” |
| Threshold stability | Threshold itself is a validation-sample estimate. | 2,000 patient bootstrap threshold distributions. | A single τ hides sampling variability. | “We asked whether perturbing validation patients moves the cutoff.” |
| ECE sensitivity | ECE depends on binning. | Equal-width/equal-frequency and 10/15/20-bin grid. | One reference number could hide estimator sensitivity. | “We report the declared reference and its binning sensitivity.” |
| Logistic extension | Compare a richer affine-logit map alongside temperature. | Separate extension; validation metrics may be optimistic. | No alternative calibration comparator. | “This is extension analysis, not primary A/B/C/D.” |
| Prevalence sensitivity | Explore effect of prevalence at frozen score/threshold using negatives-only sampling. | Simulated prevalence-only scenarios; recall pinned. | No controlled prevalence response view. | “This changes prevalence only; it is not a new dataset.” |
| Grad-CAM | Qualitative review of selected frozen-model cases; project includes NIH boxes for a subset. | Exploratory maps plus 43-case box sanity check. | No visual error-review aid. | “It suggests regions but proves neither causality nor clinical localization.” |
| External validation | Needed to assess cross-cohort generalization; CheXpert files unavailable. | Status pending; no external metrics. | Cross-cohort claims remain untested. | “We mark it pending rather than fabricate a result.” |

## 26. Complete Results Table

Canonical results are in Tables 2–6, 10–14 under `outputs/paper/tables/`, with full source outputs under `outputs/final_results/` and `outputs/metrics/`. Core table (only actually available fields):

| Experiment | Label | Arm | AUROC | AUPRC | Precision | Recall | F1 | Specificity | Threshold | Calibration | CI / p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Baseline | Cardiomegaly | A raw/.50 | .8972 | .3043 | .1787 | .6867 | .2836 | .9153 | .5000 | raw | DeLong CI AUROC [.8821,.9124] |
| Main policy | Cardiomegaly | D temp/F1 | .8972 | .3043 | .3377 | .3711 | .3536 | .9805 | .8666 calibrated | temperature | D−A F1 +.0700 [.0280,.1065], Holm .0004 |
| Baseline | Effusion | A raw/.50 | .8587 | .4700 | .3320 | .7902 | .4676 | .7714 | .5000 | raw | DeLong CI AUROC [.8507,.8667] |
| Main policy | Effusion | D temp/F1 | .8587 | .4700 | .4440 | .5383 | .4866 | .9031 | .7035 calibrated | temperature | D−A F1 +.0191 [.0050,.0326], Holm .0262 |

AUPRC/threshold results and policy metrics are not repeated for all arms here; see `outputs/paper/tables/table_03_main_results.csv`, `table_04_ablation.csv`, and `table_05_thresholds.csv`. C/D have the same hard classifications under their corresponding fitted thresholds; do not present them as independent prediction ensembles.

## 27. Figure-by-Figure Explanation

All 14 figure descriptions, exact file paths, axes, findings, verbal explanations and caution language are in [Figure and Table Explanation Guide](FIGURE_TABLE_EXPLANATION_GUIDE.md#figures). Current canonical set: `outputs/paper/figures/figure_01_architecture.png` through `figure_14_prevalence_sensitivity.png`. Older final-results names number figures differently; paper set is authoritative for the presentation package.

## 28. Table-by-Table Explanation

All 14 table explanations are in [Figure and Table Explanation Guide](FIGURE_TABLE_EXPLANATION_GUIDE.md#tables), including caution around misleading old output filenames. Critical discrepancy: `table_02_baseline.csv` is baseline metrics, not model config; use source and reproducibility manifest for config. `outputs/final_results/tables/` has an older eight-table set; paper `outputs/paper/tables/table_01`–`table_14` is the current package.

## 29. UI Architecture

`research_demo_app.py` builds a Gradio app; `src/ui_data.py` is the read-only evidence loader. Static tabs read JSON/CSV/PNG artifacts, not recompute experiments. Live image path: upload → RGB conversion → exact evaluation preprocessing (224×224, ImageNet normalization) → frozen baseline checkpoint → logits → raw sigmoid → frozen temperature map → validation-frozen calibrated F1 threshold → per-label positive/negative policy output. Live tab does not generate Grad-CAM. UI source paths are displayed in Technical mode. Start per `DEMO_GUIDE.md`: `.venv/bin/python research_demo_app.py`; local address defaults to `http://127.0.0.1:7860`.

## 30. UI Tab-by-Tab Explanation

Each tab, source, action, speaking line, caveat, and suggested time:

| Tab | Purpose/data | Point to / say | Avoid | Time |
|---|---|---|---|---:|
| Research Overview | Question, pipeline and frozen design from manifest | “Here is the controlled question and the boundary: retrospective NIH study.” | Clinical claims | 25s |
| Live X-ray Demonstration | Uploaded image and frozen checkpoint | Point to raw probability, calibrated score, frozen τ, predicted label; say score→policy output | “Diagnosis”; confidence as certainty | 60–90s |
| Calibration | Validation reports, temperatures, logistic extension, ECE figure | Explain NLL/Brier vs ECE and map monotonicity | “All metrics improved” | 45s |
| A/B/C/D Decision Policy | Test arm results CSV | State exact four arm definitions, then D−A | Redefining arms | 45s |
| Statistical Evidence | Patient bootstrap, paired tests, DeLong | Highlight patient unit, CI, Holm p | Clinical significance | 45s |
| Threshold Stability | Validation patient-bootstrap table/figure | Show calibrated F1 intervals/±10% | Perfect robustness | 30s |
| Error Analysis | Saved error strata, figure, Grad-CAM | Show high-conf errors and exploratory heatmap | Causal explanation | 35s |
| Decision Policy Comparison | Five policy comparison table/figure | Connect policy to different operating preferences | One policy universally best | 30s |
| Prevalence Sensitivity | Frozen negative-subsampling simulation | Show simulated prevalence changes precision/recall limits | External validation | 25s |
| Reproducibility & Limitations | Manifest, source paths, external status | Point to frozen hashes and pending external status | “Validated externally” | 35s |

**Display modes:** Presentation hides source/path detail for a clean narrative; Technical exposes more provenance. The dashboard structure and frozen artifact values remain the same. The UI is a demonstrator, not a clinical system.

## 31. Live Demo Script

1. Select Presentation mode and open Research Overview: “This view summarizes a fixed classifier and evaluation; it reads the completed evidence package.”
2. If practical, open Live X-ray Demonstration and choose a curated sample. “The app converts the upload to RGB, resizes/normalizes it with the evaluation transform, and runs the frozen checkpoint.”
3. Define terms: **Raw probability** = sigmoid of the model logit before post-hoc calibration. **Calibrated probability** = temperature-transformed score; this label means mapped score, not guaranteed perfect calibration. **Threshold** = validation-selected cutoff used to turn a score into a positive/negative policy output. **Predicted label** = whether mapped score crosses that cutoff. **Confidence** = informal score magnitude, not correctness probability guarantee.
4. State: “This is a research demonstration, not a clinical diagnostic system. No live Grad-CAM is computed.”
5. Switch to Calibration and A/B/C/D; point to frozen values and source. Technical mode can show provenance paths. Finish at Reproducibility & Limitations and state external validation is pending.

Do not upload PHI; use provided research samples only.

## 32. 15-Minute Presentation

Follow `docs/PRESENTATION_SLIDE_PLAN.md`. Core sequence: title/scope; problem; question; dataset/split; fixed model; literature gap; A/B/C/D; calibration; thresholds; results; statistics; stability/error; UI; limitations/contribution. Full target slide timing plan is detailed there.

## 33. 10-Minute Presentation

Compress dataset+split and model into 90 seconds; literature to 30 seconds; A/B/C/D and results get 3 minutes; show one calibration caveat, primary CIs, 60-second UI, limitations, conclusion. Drop extension details unless asked. Never omit the single-dataset and pending external caveats.

## 34. 20-Minute Presentation

Use full 18-slide plan, expanding literature comparison, five threshold policies, patient-level statistics, ECE sensitivity, threshold stability and UI to ~2 minutes. Add backup details on prevalence simulation and table-level metrics. Avoid implying the additional analyses increase external validity.

## 35. Exact Speaking Script

**Opening — 30 s:** “This project studies how probability calibration and decision thresholds affect a fixed chest X-ray classifier. I use two NIH ChestX-ray14 labels, Cardiomegaly and Effusion, and keep the trained DenseNet121 fixed. The main point is to separate probability quality from ranking and from the final positive/negative operating decision.”

**Problem — 60 s:** “A classifier can rank cases usefully and still have probabilities that do not match observed frequencies. A threshold then turns each score into a decision. Under class imbalance, the default 0.5 cutoff may give a poor precision/recall balance. So I evaluate these three things separately: ranking, probability metrics, and thresholded operating metrics. The goal is not to claim one cutoff is clinically correct; it is to make the consequences measurable.”

**Dataset — 60 s:** “The frozen analysis uses 109,312 available images from 29,720 patients, with Cardiomegaly and Effusion targets. The patient-level split contains 76,977 training images, 16,451 validation images, and 15,884 test images. The NIH labels are report-derived. I should flag one documented unreadable-image replacement issue: the split table lists 1,996 test Effusion positives while cached prediction support is 1,997. The study preserves the frozen processing behavior and reports that caveat transparently.”

**Model — 60 s:** “The model is torchvision DenseNet121 initialized with ImageNet-1K weights, followed by global pooling and a linear two-logit head. It was trained with weighted binary cross entropy in two phases: five partially frozen epochs, then five fully unfrozen epochs. The best checkpoint was selected using mean validation AUROC and then frozen for all calibration and threshold experiments. This is an established architecture choice, not a novel network.”

**Literature/gap — 90 s:** “Temperature scaling was established by prior calibration work, and Rajaraman and colleagues already studied calibration and threshold choice for imbalanced medical image classification, including chest radiographs. Their study considered multiple model and data settings and compared default and PR-guided thresholds. I therefore do not claim that I discovered threshold selection or that calibration universally fails. My narrower contribution is an explicit A/B/C/D decomposition for one fixed two-label NIH classifier, with paired patient-level uncertainty, threshold stability and calibration-bin sensitivity reported alongside the operating tradeoffs.”

**Methodology — 120 s:** “The model emits one logit per label. Temperature scaling divides each logit by a positive scalar fitted by validation negative log likelihood, then applies sigmoid. Separately, thresholds are selected on validation: fixed 0.5, F1-optimal, Youden, sensitivity-constrained or precision-constrained. The test set is reserved for evaluation. The model is never retrained after calibration; calibration changes score values, and threshold selection determines the final hard decisions. Patient grouping is used both in splitting and in uncertainty resampling to recognize repeated images within the same patient.”

**A/B/C/D — 120 s:** “A is raw score at 0.5. B is temperature-scaled score at 0.5. C is raw score at the validation F1 threshold. D is temperature-scaled score at its validation F1 threshold. A versus B isolates the calibration map at the fixed cutoff; A versus C isolates the threshold policy on raw scores; B versus D does so on calibrated scores. D versus A is the combined comparison. Temperature scaling is monotonic, so it preserves ranking. With thresholds fitted on each corresponding scale, C and D make the same decisions here. This design is controlled for this fixed model and does not establish a general causal law.”

**Results — 120 s:** “At 0.5, test F1 is 0.2836 for Cardiomegaly and 0.4676 for Effusion. Under D, it is 0.3536 and 0.4866. Precision rises from 0.1787 to 0.3377 and 0.3320 to 0.4440, while recall falls from 0.6867 to 0.3711 and 0.7902 to 0.5383. The patient-bootstrap D-minus-A F1 intervals are positive for both labels. Temperature scaling improves NLL and Brier but worsens the equal-width 15-bin ECE on the test artifacts. So I describe calibration as metric-dependent and threshold policy as the source of the operating point change.”

**UI demo — 120 s:** “The dashboard reads frozen outputs. The overview summarizes the question. Calibration shows fitted values and metric comparisons. A/B/C/D shows the four decisions. Statistical Evidence shows patient-level paired results. Threshold Stability shows how fitted cutoffs move under validation resampling. Error Analysis contains test errors and exploratory Grad-CAM. The live upload path uses the frozen checkpoint and the same evaluation preprocessing, applies saved temperature and threshold values, and returns model score/policy labels—not diagnosis. I can switch to Technical mode to show artifact paths.”

**Limitations — 45 s:** “This is one retrospective dataset, one split and two labels, with report-derived labels and a documented image-read defect. External validation is pending because the CheXpert data were unavailable, and no external metrics were produced. Grad-CAM is exploratory. There is no prospective, regulatory, or clinical utility evaluation.”

**Conclusion — 30 s:** “The study shows that for this fixed NIH setup, validation-frozen thresholds change measured operating characteristics, while temperature scaling has mixed probability-metric effects and preserves ranking. The evidence is traceable, but generalization remains untested. External validation under frozen parameters is the next required step.”

## 36. Likely Faculty Questions

See [Faculty QA](FACULTY_QA.md) for 44 questions with ideal answers, technical backup, and mistakes to avoid. Use answers that say “not verified” where appropriate; do not improvise unsupported rationales.

## 37. Q&A Answers

The detailed Q&A file covers dataset rationale, split, model/training, calibration, thresholds, statistical methods, literature, novelty, explainability, external validation, and UI. It includes required questions such as why τ=0.9021, why not fit on test, why ECE worsened, and how the work differs from Rajaraman et al.

## 38. Novelty Defense

| Claim category | Status |
|---|---|
| Novel architecture | **No** |
| Novel calibration algorithm | **No** |
| Novel medical dataset | **No** |
| Clinical validation | **No** |
| General theoretical discovery | **No** |
| Defensible work | Controlled empirical decomposition, pathology-specific analysis, calibration/threshold separation, patient-level inference, threshold stability, metric-sensitive calibration, decision-policy and prevalence sensitivity, evidence traceability, candid limitation reporting |

These are defensible characteristics of this repository's analysis, not claims of literature priority. See `docs/LITERATURE_POSITIONING.md`.

## 39. What Not To Say

| Do not say | Safer wording |
|---|---|
| “We invented temperature scaling.” | “We applied a known post-hoc temperature-scaling method.” |
| “We created a new DenseNet.” | “We used a fixed torchvision DenseNet121 baseline.” |
| “The model is clinically validated.” | “This is a retrospective research demonstration; external validation is pending.” |
| “Calibration always improves performance.” | “NLL/Brier improved while the reference ECE worsened in this run.” |
| “External validation was completed.” | “The evaluator exists, but no external cohort/metrics were available.” |
| “Grad-CAM proves the model uses the heart.” | “The heatmaps are exploratory; box checks are descriptive and limited.” |
| “We are state of the art.” | “No state-of-the-art comparison is claimed.” |
| “We discovered threshold optimization.” | “Threshold selection is established; we quantify its effects in this fixed setup.” |
| “D is clinically best.” | “D optimizes validation F1 and has a specific measured precision/recall tradeoff.” |
| “The prevalence analysis validates another population.” | “It is negative-subsampling simulation on frozen predictions, not external validation.” |

## 40. Limitations

Single public dataset and split; two labels; report-mined labels; class imbalance; documented image-read substitution mismatch; no multi-seed or cross-validation estimate; historical mixed-precision edge behavior; calibration metrics depend on estimator/binning; F1 is not clinical utility; thresholds depend on validation sample; logistic extension evaluated in-sample on validation; Grad-CAM not clinical localization validation; prevalence simulation changes only prevalence; no external cohort, prospective study, regulatory review, or clinical utility assessment.

## 41. Reproducibility

Authoritative frozen objects include checkpoint manifest/hash, split CSV, prediction CSVs with metadata, temperature JSON, threshold JSON, final result manifests, evidence map and claim audit. Core source map:

| Stage | Source | Input → output; split/fitting discipline |
|---|---|---|
| Config/targets | `src/config.py` | frozen labels, transforms, fractions and seeds |
| Dataset/labels/split | `src/dataset.py` | NIH metadata/images → patient-grouped data; train/val/test |
| Model | `src/model.py` | RGB tensor → two logits; pretrained fixed architecture |
| Train/checkpoint | `src/train.py` | train/val loaders → frozen baseline; validation selects best |
| Prediction/inference | `src/inference.py` | frozen checkpoint + images → cached raw logits/scores |
| Baseline/evaluation | `src/baseline_eval.py`, `src/evaluate.py`, `src/metrics.py` | probabilities/labels → metrics |
| Calibration | `src/calibration.py`, `src/fit_parameters.py` | validation logits/labels → T; test only transform/evaluate |
| Thresholds | `src/thresholds.py`, `src/fit_parameters.py` | validation scores → τ; frozen application |
| A/B/C/D | `src/experiments.py`, `src/statistics/data.py` | cached predictions + frozen params → arm tables |
| CI/tests/DeLong | `src/uncertainty.py`, `src/statistics/bootstrap.py`, `src/statistics/tests.py`, `src/statistics/delong.py` | patient-cluster paired inference; test outcomes |
| Error / CAM | `src/error_analysis.py`, `src/gradcam.py` | test predictions/checkpoint → strata, fixed cases, box check |
| Paper evidence | `src/paper_artifacts.py`, `src/paper_build.py`, `outputs/paper/paper_evidence_map.json` | existing metrics → tables/figures/prose traceability |
| Dashboard | `src/ui_data.py`, `research_demo_app.py` | read-only evidence plus live frozen inference |

No artifacts were regenerated for this documentation task. Existing tests should be run as a check, but if tests rewrite timestamps/metadata in tracked outputs, restore only those incidental changes before finishing and verify research artifact hashes/status.

## 42. Presentation Checklist

- Lead with scope and one research question.
- Keep the A/B/C/D definitions verbatim.
- State validation-only fitting and test-only evaluation.
- Separate ranking, probability metrics, and thresholded decisions.
- Report both gains and recall tradeoff.
- State patient-level inference method and CI.
- Mention literature overlap before contribution.
- Name all uncertainty around dataset-table Effusion support.
- Call Grad-CAM exploratory.
- Call prevalence results simulations.
- Say external validation is pending/no metrics.
- In UI, call outputs scores/operating labels, never diagnoses.

## 43. Source Traceability

| Claim | Artifact/source | Paper item | UI/slide |
|---|---|---|---|
| Split sizes / patient groups | `data/processed/split_index.csv`, `table_01_dataset.csv` | Figure 2/Table 1 | Overview / dataset slide |
| Baseline metrics | `outputs/metrics/baseline/baseline_metrics_test.json` | Table 2; Figures 3–4 | Overview / results |
| Calibration parameters/metrics | `outputs/metrics/calibration/*.json` | Figures 5,10–11; Tables 6,13 | Calibration |
| Frozen thresholds/policies | `outputs/metrics/thresholds/*.json` | Figures 6,13; Tables 4–5 | A/B/C/D / policies |
| Primary D−A result | `outputs/final_results/arm_differences.csv`, `statistical_report.json` | Table 11 | Statistical Evidence |
| Test AUROC uncertainty | `outputs/metrics/patient_stats/delong_auroc.json` | Table 3/10 | Statistical Evidence |
| Stability | `outputs/metrics/threshold_stability/*` | Figure 12/Table 12 | Threshold Stability |
| Errors and patient concentration | `outputs/metrics/error_analysis/error_analysis_test.json`, final CSV | Figure 8/Table 7 | Error Analysis |
| Grad-CAM | `outputs/gradcam/gradcam_summary.json`, `bbox_localization.json` | Figure 9/Table 8 | Error Analysis |
| External absence | `outputs/metrics/external/status.json` | Figure/table status item 9 | Reproducibility & Limitations |
| Frozen model | `outputs/checkpoints/baseline/baseline_manifest.json` | reproducibility manifest | Overview / Live demo |
| All paper claims | `outputs/paper/paper_evidence_map.json`, `claim_audit.csv` | entire paper package | Technical mode |

Research question → paper sections §1–3 → A/B/C/D → cached test predictions and `arm_differences.csv` → Tables 3/4/11 → UI A/B/C/D and Statistical Evidence → slides 8–12.

## 44. Final Readiness Assessment

| Area | Status | Evidence / limit |
|---|---|---|
| Model | READY / frozen | Hash-identified DenseNet121 checkpoint. |
| Data | NEEDS DISCLOSURE | Single NIH set, weak labels, partial image set, Effusion support mismatch documented. |
| Split | READY WITH CAVEAT | Patient-level 70/15/15, seed 42, zero overlap; not NIH official split. |
| Statistical | READY WITH CAVEATS | Patient-level paired CI/tests; single seed/split; distinguish statistical from clinical. |
| Literature | READY FOR POSITIONING | Closest work explicitly compared; not a systematic review. |
| UI | READY FOR DEMONSTRATION | Ten tabs; artifact viewer and frozen live inference; not a clinical product. |
| Reproducibility | READY WITH CAVEATS | Frozen artifacts/mapping; AMP boundary and augmentation-doc discrepancy; artifact regeneration not needed. |
| External validation | PENDING / BLOCKED | CheXpert unavailable; no metrics. |
| IEEE presentation | READY FOR AUTHOR REVIEW | Story and source links prepared; author should resolve/document augmentation wording and disclose data-count mismatch. |

**Unresolved inconsistencies and wording risks:** (1) 1,996 raw/split Effusion positives vs 1,997 prediction support; (2) augmentation 0.15 in current config/manifest vs ±20% in stale `docs/BASELINE.md`; (3) old phase documents refer to 193 tests/2,000 image resamples while current final statistical report uses 5,000 patient resamples; (4) old project title/modality scope; (5) `outputs/final_results/tables/` has eight tables while the current paper package has fourteen; (6) `outputs/paper/tables/table_02_baseline.csv` is metrics, not configuration. Use current source/config and paper/final artifacts, and state historical mismatch instead of silently editing results. Do not present the confusion as resolved beyond recorded evidence.

**Overall:** READY for a faculty presentation of the bounded retrospective findings after explicitly disclosing limitations and number/doc inconsistencies. **Not ready for claims of clinical use or external generalization.**
