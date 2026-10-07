# Literature Positioning and Safe Claims

**Evidence labels:** Repository fact, Artifact fact, Literature fact, Interpretation. This note positions the frozen NIH ChestX-ray14 study; it does not change the experiment or claim priority.

## Core question

Can post-hoc calibration and validation-frozen, class-specific decision thresholds improve measured operating characteristics of a fixed two-label DenseNet121 on a held-out NIH ChestX-ray14 split? The experiment decomposes the calibration and threshold components through A/B/C/D. It is a retrospective, single-dataset study; external validation is pending.

## Literature table

| Work | Method / dataset / task | Calibration or threshold relevance | Relationship and permissible claim |
|---|---|---|---|
| Wang et al. (2017), *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, CVPR. [CVF paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) | Introduces the NIH chest-radiograph collection with report-derived multi-labels and localization benchmarks; the later commonly named ChestX-ray14 dataset has 14 labels. | No post-hoc calibration question central to our experiment. | Dataset and weak-label context, not a performance comparator unless split, label handling, and protocol match. We use the local NIH metadata and available images; do not imply exact comparability to its published split. |
| Rajpurkar et al. (2017), *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. [arXiv](https://arxiv.org/abs/1711.05225) | DenseNet121 on ChestX-ray14 for pneumonia, then multi-label extension. | Not the calibration/threshold decomposition studied here. | Establishes precedent for DenseNet121 and NIH multi-label chest X-ray classification. This repository's head, two labels, split, training, and endpoint differ; no state-of-the-art comparison is claimed. |
| Huang et al. (2017), *Densely Connected Convolutional Networks*, CVPR. [CVF paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) | Dense connectivity architecture. | Background for selected backbone. | Architecture precedent only; this study does not invent or materially alter DenseNet. |
| Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML. [PMLR](https://proceedings.mlr.press/v70/guo17a.html) | Post-hoc neural network calibration; introduces/empirically evaluates scalar temperature scaling among methods. | Directly motivates temperature scaling as a simple validation-fitted probability transform. | Temperature scaling itself is established prior work. Our study applies per-label sigmoid temperature scaling to a fixed multi-label chest X-ray baseline and evaluates several probability/decision metrics; it does not claim a new calibration algorithm. |
| Rajaraman, Ganesan & Antani (2022), *Deep learning model calibration for improving performance in class-imbalanced medical image classification tasks*, PLOS ONE 17(1):e0262838. [PLOS full text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262838) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8794113/) | Shenzhen TB chest X-rays and APTOS’19 fundus images; simulated imbalance levels; VGG-16, DenseNet-121, Inception-V3 and EfficientNet-B0 examined, with selected models taken forward. Calibration methods include Platt, beta, and spline. | Compares calibrated/uncalibrated predictions at fixed 0.5 and PR-derived F-score-optimal cutoffs. Reports calibration gains at default 0.5; calibrated performance is not significantly superior at the PR-guided cutoff. | This is the closest prior work identified. The present work must not claim discovering that threshold choice matters more than calibration, or that calibration improves classification universally. Our defensible incremental framing is a frozen-model, pathology-specific A/B/C/D decomposition on a patient-level NIH split, with paired patient-level inference, threshold stability, ECE binning sensitivity, explicit policy/prevalence analyses, and artifact traceability. This is a difference in controlled scope and reporting, not proof of novelty or superiority. |
| Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, ICCV. [IEEE Xplore](https://doi.org/10.1109/ICCV.2017.74) · [CVF paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) | Gradient-weighted class activation visualization for CNN decisions. | Basis for exploratory visualization in this repository. | Supports the method description, not a clinical explanation claim. Our 16 examples and 43 boxed test cases are exploratory/descriptive and cannot prove causal feature use or localization validity. |
| DeLong, DeLong & Clarke-Pearson (1988), *Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach*, Biometrics. [PubMed](https://pubmed.ncbi.nlm.nih.gov/3203132/) | Nonparametric covariance/variance for ROC AUCs, especially correlated curves on same cases. | Used here for per-label test AUROC SE/CI; monotonic calibration curves coincide, so pairwise test degenerates. | Statistical method reference. Do not describe DeLong as a test of thresholded F1 or a cross-label comparison here. |
| Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of Statistics 6(2):65–70. [Bibliographic record](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1572543024862166272) | Stepwise family-wise error adjustment. | Applied to the two primary per-label D−A F1 permutation tests. | A standard correction, not a research contribution. |

## Closest prior work: Rajaraman et al.

The authoritative PLOS article is available in full. Its abstract and methods explicitly study chest X-ray and fundus modalities, several backbones, different training imbalance degrees, multiple calibration methods, and two decision thresholds: default 0.5 and a PR-curve-derived optimum. Its abstract reports statistically significant calibration-associated performance differences at 0.5, but no significant difference at the PR-guided cutoff. Therefore, both the question of whether calibration changes thresholded classification behavior and the importance of operating-point choice are already in prior literature.

### Exact overlap

- Imbalanced medical image classification, including chest X-ray.
- Post-hoc calibration and thresholded performance.
- Default 0.5 compared with validation/PR-guided operating cutoffs.
- Recognition that calibration effects depend on operating threshold.

### Exact differences evidenced by our repository

- One frozen DenseNet121 baseline, two NIH labels (Cardiomegaly, Effusion), one patient-level 70/15/15 split, and fixed seed 42.
- Explicit four-arm A/B/C/D factorial-style decomposition: raw/calibrated × fixed/F1-optimal policy, with thresholds and temperatures fitted only on validation.
- Paired test-set inference clustered by patient, 5,000 patient bootstrap resamples and paired permutation tests with Holm adjustment for the two primary label-specific D−A F1 endpoints.
- Additional threshold stability, ECE bin-grid, logistic calibration extension, policy comparison, prevalence-subsampling simulation, and integrated evidence mapping.

These differences support “controlled, pathology-specific empirical decomposition and expanded uncertainty/sensitivity reporting under this fixed protocol.” They do **not**, alone, prove that the framework is the first, best, clinically useful, or generalizable. A comprehensive systematic search was not performed; priority claims remain unknown.

### Safe contribution language

> We report a controlled, retrospective analysis of how validation-fitted per-label temperature scaling and validation-frozen thresholds affect operating-point and probability-quality metrics for a fixed two-label DenseNet121 on a patient-level NIH ChestX-ray14 split. The study makes the component effects and uncertainty visible; external generalization remains untested.

## Literature facts versus project facts

External literature is used only for methodological and historical context. Every reported study result in the guide is sourced to repository artifacts. Rajaraman et al.'s findings are not imported as if reproduced here. No paper located here establishes that this particular repository's frozen model is clinically validated.
