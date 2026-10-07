# Faculty Viva Questions and Answers

Answers state what the repository supports. “Not verified from the repository” is used where an implementation choice lacks an explicit rationale.

## Dataset and split

### 1. Why NIH ChestX-ray14?
**Answer:** It is the dataset present in the repository: NIH image metadata, available PNGs, and labels are assembled into the frozen split. The study uses 109,312 available images from 29,720 patients. This enables a reproducible within-dataset experiment; it does not establish external generalization. **Backup:** `src/config.py`, `src/dataset.py`, `data/processed/split_index.csv`, `outputs/paper/tables/table_01_dataset.csv`. NIH's paper describes report-mined multi-label data [Wang et al.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf). **Avoid:** “NIH represents every clinical population.”

### 2. Why only two labels?
**Answer:** The implemented target list is Cardiomegaly and Effusion. The repository does not provide a verified scientific rationale for excluding the other labels; broader coverage is out of scope for this frozen experiment. **Backup:** `src/config.py::TARGET_LABELS`, `docs/PROJECT_SCOPE.md`. **Avoid:** inventing a selection rationale.

### 3. Why patient-level split?
**Answer:** Images from the same person can share anatomy, acquisition, and disease patterns. If a patient's images were placed in both training and test, the test could partly measure familiarity with that patient. Grouping by Patient ID keeps each patient in one split. **Backup:** `src/dataset.py::assign_splits`; overlap is 0 across split pairs. **Avoid:** claiming all forms of leakage are impossible.

### 4. Why not split images independently?
**Answer:** Repeated views from one patient could leak patient-specific patterns across splits and make independent-image uncertainty too optimistic. Patient grouping addresses this dependence at splitting and inference. **Avoid:** saying it removes all dataset bias.

### 5. Why 70/15/15?
**Answer:** It is the frozen configuration in `src/config.py` and yields 76,977/16,451/15,884 images. The particular rationale beyond the project configuration is **not verified from the repository**. **Avoid:** calling the ratio universally optimal.

### 6. Why seed 42?
**Answer:** Seed 42 is the configured split/training and bootstrap seed. Why 42 specifically was chosen is not verified; it is a reproducibility constant, not a scientific parameter. **Avoid:** implying it is uniquely principled.

### 7. Does the split match NIH's official split?
**Answer:** The repo's reproducibility notes state the official split lists are absent; the split was made patient-level with the local seed. Absolute performance should not be directly compared with work using a different split. **Avoid:** presenting it as the official NIH split.

### 8. Why does the dataset table show 1,996 test Effusion positives but metric support is 1,997?
**Answer:** This is a documented input defect: the unreadable/truncated PNG `00029705_000.png` causes the loader's deterministic replacement behavior to use the next readable sample's label. The split table's actual target sum is 1,996, while cached predictions/metrics support count reports 1,997. Preserve the discrepancy, cite `docs/BASELINE.md` and `docs/REPOSITORY_AUDIT.md`; do not silently reconcile. **Avoid:** calling it an independently verified 1,997 unique cases in raw labels.

## Model and training

### 9. Why DenseNet121?
**Answer:** It is the implemented, frozen baseline and has precedent in chest radiograph studies (including CheXNet). The repository does not document a comparative backbone selection. **Backup:** `src/model.py`, `src/config.py`; [DenseNet paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html). **Avoid:** “DenseNet is best for this task.”

### 10. Why ImageNet initialization?
**Answer:** `src/model.py` explicitly uses torchvision `DenseNet121_Weights.IMAGENET1K_V1`. It is a training design choice; no random-initialization ablation exists. **Avoid:** claiming the project proved pretraining improves performance.

### 11. Why 224×224?
**Answer:** This is the configured input resolution and evaluation transform. The repository does not show an input-size ablation. **Avoid:** presenting it as optimal.

### 12. What is the head and output?
**Answer:** Global pooled 1024-dimensional DenseNet features feed a `Linear(1024,2)` head. It returns two logits; sigmoid is applied for probabilities/metrics. The targets are independently represented binary findings, so the implemented loss is BCE-with-logits with train-derived positive weights.

### 13. What training schedule?
**Answer:** Five epochs with backbone frozen except denseblock4/norm5 plus classifier at learning rate 1e-4, then five epochs fully unfrozen at 1e-5; Adam, weight decay 0, batch size 32, best checkpoint by mean validation AUROC (epoch 8). See `src/config.py`, `src/train.py`, `docs/BASELINE.md`.

### 14. Why no model comparison or transformer?
**Answer:** The research question deliberately holds the model fixed to isolate calibration and decision policy. No evidence in this repo supports a cross-architecture ranking. **Avoid:** claiming architecture superiority.

## Calibration and thresholds

### 15. What is calibration?
**Answer:** Whether predictions assigned probability near p correspond to events occurring at approximately p frequency, over a population. Discrimination/ranking and calibration are different properties.

### 16. Why temperature scaling?
**Answer:** It is a recognized one-parameter post-hoc method from Guo et al. The repo fits one positive scalar per label to validation NLL and maps sigmoid probabilities via `sigmoid(z/T)`. It changes probability scale while preserving order. **Avoid:** calling it novel.

### 17. Why fit only validation?
**Answer:** Fitting on test would use evaluation outcomes to select parameters and bias reported test results. Runtime guards require split `val`; test receives frozen parameters. **Backup:** `src/calibration.py`, `src/thresholds.py`, tests.

### 18. Why does AUROC remain unchanged?
**Answer:** T is positive and sigmoid(z/T) is strictly increasing in z, so pairwise score ordering is unchanged. AUROC and AUPRC are rank-based and therefore unchanged, subject to numerical ties/implementation. Thresholded metrics can change if the numerical threshold is not transformed consistently.

### 19. Why did ECE worsen after temperature scaling?
**Answer:** The validation report shows NLL and Brier improve but equal-width 15-bin ECE rises for both labels. These metrics summarize different aspects, and ECE depends on bins. The honest result is mixed; no universal calibration improvement is claimed.

### 20. Why is threshold 0.5 used?
**Answer:** It is the baseline operating point (arm A) and the common reference defined in the experiment. It is not asserted optimal under the observed class imbalance. **Avoid:** saying 0.5 is inherently correct.

### 21. Why is the raw Cardiomegaly F1 threshold 0.9021?
**Answer:** It is the threshold selected by maximizing validation F1 on raw Cardiomegaly probabilities; it is frozen and applied to test. It is not a clinical cutoff, nor selected from test. The calibrated threshold for D is 0.8666406 because the calibrated score scale differs.

### 22. Why use F1?
**Answer:** F1 is the study's specified summary of precision/recall trade-off and primary endpoint; it weights neither false positives nor false negatives according to a clinical utility model. It is not a universal clinical objective.

### 23. What do policies mean?
**Answer:** Fixed = 0.5; F1-optimal maximizes validation F1; Youden maximizes sensitivity+specificity−1; sensitivity-constrained targets sensitivity ≥0.90; precision-constrained targets precision ≥0.50 (as configured). These encode different operating preferences. Verify feasibility/selection rules in `src/thresholds.py` before explaining edge cases.

## A/B/C/D and inference

### 24. What are A/B/C/D?
**Answer:** A raw+0.50; B temperature+0.50; C raw+validation F1 threshold; D temperature+validation F1 threshold. No retraining. Definitions are frozen in `outputs/paper/draft_results.md` and `src/statistics/data.py`.

### 25. What comparison isolates calibration?
**Answer:** A vs B at 0.50 and C vs D at F1-optimal threshold compare raw versus temperature-scaled probabilities at corresponding frozen policies. Because transformed thresholds are refit on the validation scale and temperature scaling is monotone, C and D make identical test classifications here. Probability metrics still differ.

### 26. What comparison isolates thresholding?
**Answer:** A vs C (raw), and B vs D (calibrated), compare fixed 0.50 with validation F1-optimal threshold. On this artifact, the observed F1 changes are identical by label across raw/calibrated policy pairs.

### 27. Can you claim a causal general law?
**Answer:** No. It is a controlled retrospective comparison on one frozen model, dataset, split, and label pair. It supports conditional empirical statements only.

### 28. Why patient bootstrap?
**Answer:** Images within one patient are clustered, so resampling images as if independent can understate uncertainty. The patient-level bootstrap resamples patients and retains their images; arms are paired on the same resample. It does not substitute for external replication.

### 29. Why permutation tests?
**Answer:** The paired permutation flips arm assignments within patient-level contributions under the null for paired differences; it provides a test aligned with the paired design and nonlinear aggregate metric. Thresholds remain validation-frozen.

### 30. Why Holm correction?
**Answer:** Two primary label-specific D−A F1 hypotheses are tested. Holm controls family-wise error across that declared pair. Other tests are exploratory/unadjusted. [Holm (1979)](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1572543024862166272).

### 31. Why DeLong?
**Answer:** It provides per-label AUROC uncertainty via placement-value covariance. Since calibration maps are strictly monotone, comparing raw and calibrated ROC curves is degenerate; no AUROC gain or pairwise p-value is claimed. It is not the test for F1.

### 32. Are the findings clinically significant?
**Answer:** Unknown. Statistical significance describes compatibility under a statistical test, not clinical utility; this study has no clinical utility analysis or prospective deployment evidence.

## Results and extensions

### 33. Main finding?
**Answer:** On the test split, D−A F1 is +0.070034 (95% patient-bootstrap CI 0.028025–0.106492; permutation p≈0.0002, Holm p=0.0004) for Cardiomegaly and +0.019090 (0.005024–0.032553; p≈0.0262, Holm p≈0.0262) for Effusion. Recall declines as precision/specificity rise. Calibration alone leaves 0.5 classifications unchanged.

### 34. Why does prevalence change F1?
**Answer:** With frozen scores/threshold and all positives retained, negative subsampling changes the false-positive pool and positive fraction; precision and F1 respond to prevalence while recall is fixed by positive examples and their predictions. This is a simulated negative-subsampling analysis, not external validation.

### 35. Why add logistic calibration?
**Answer:** As a separate extension comparing an affine-logit sigmoid map against temperature scaling. It is validation-fitted, strictly increasing with positive slope, not part of A/B/C/D. Near-zero in-sample validation ECE is not evidence of out-of-sample calibration; generalization must be tested on untouched data.

### 36. What does threshold stability show?
**Answer:** The project bootstrapped validation patients 2,000 times. Calibrated F1 threshold 95% intervals: Cardiomegaly 0.7860–0.8799 around first fit 0.8666; Effusion 0.6434–0.7444 around 0.7035. These lie within ±10% but not ±5% bands as the artifact marks. Thresholds have sampling variability; this is not perfect robustness.

### 37. What do ECE bin grids show?
**Answer:** ECE is computed under equal-width and equal-frequency schemes and multiple bin counts. Values depend on binning and finite-bin occupancy. See `outputs/metrics/ece_sensitivity/ece_sensitivity.csv`; the primary reference is equal-width, 15 bins. Do not select a favorable binning after inspection.

### 38. What is error analysis?
**Answer:** At the F1-optimal test operating point, 563 Cardiomegaly errors (274, 48.7%, confidence ≥0.9) and 2,268 Effusion errors (193, 8.5%) are reported. Confidence is not correctness; high-confidence failures motivate caution, not clinical conclusions.

### 39. Does Grad-CAM prove correct anatomy?
**Answer:** No. It is a gradient-based visualization at `encoder.features.denseblock4`, plus descriptive box comparisons on 43 eligible test images. Pointing hits 21/43. These are exploratory checks, not proof of causal focus or clinically valid localization.

### 40. Why no external validation?
**Answer:** The expected CheXpert cohort was unavailable in configured/probed locations. The evaluator exists, but `no_metrics_were_computed` is true. We report pending status rather than fabricate results. Future work must acquire an authorized cohort and evaluate with frozen model/calibration/thresholds under prespecified label mapping.

### 41. What is novel?
**Answer:** No novel architecture or algorithm is claimed. The defensible contribution is the controlled, artifact-traceable decomposition plus patient-level uncertainty and sensitivity analyses for this fixed study. Rajaraman et al. already studied calibration and operating thresholds in imbalanced medical imaging, including chest X-rays.

### 42. How is it different from Rajaraman et al.?
**Answer:** Their broader systematic sweep covers two modalities, multiple backbones/imbalance levels/calibration methods, and default versus PR-optimized threshold. This repository instead holds one NIH multi-label DenseNet121 fixed and adds explicit A/B/C/D separation, paired patient-level inference, stability and metric sensitivity. This is incremental scope/methodology, not a claim of priority or superior performance.

### 43. Can it be used clinically?
**Answer:** No such use is supported. It is a research demonstration, not a clinical diagnostic system; there is no external or prospective validation, regulatory evaluation, or clinical utility assessment.

### 44. What is the UI demonstrating?
**Answer:** Ten tabs read frozen artifacts; the live image tab preprocesses an uploaded image, executes the frozen checkpoint, shows raw/calibrated outputs and compares them to validation-frozen thresholds. The UI does not fit model parameters or regenerate research outputs. “Predicted label” is an operating-point output, not diagnosis.
