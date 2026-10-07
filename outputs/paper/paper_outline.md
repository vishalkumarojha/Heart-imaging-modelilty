# Paper Outline

Evidence convention: each section lists the table/figure/artifact that supplies
its numbers. `T#` = `outputs/paper/tables/table_0X_*.csv`,
`F#` = `outputs/paper/figures/figure_0X_*.png`.

## 1. Title
Selected title: **Calibration-Aware Decision Policies for Multi-Label Chest
X-Ray Classification Under Class Imbalance**. Evidence: `titles.md`,
`abstract.md`.

## 2. Abstract
Numbers: AUROC 0.8972 / 0.8587 (Cardiomegaly / Effusion); F1 gains
0.2836→0.3536 and 0.4676→0.4866; precision 0.1787→0.3377 and 0.3320→0.4440.
Sources: `draft_results.md`, `table_02_baseline.csv`, `table_03_main_results.csv`,
`abstract.md`.

## 3. Keywords
calibration · threshold policy · class imbalance · chest X-ray · DenseNet121 ·
multi-label classification · external validation (pending). No numeric source
required.

## 4. Introduction
Research question (R) and why it matters. Numbers: none new; frames
high-confidence error problem with `table_07_error_analysis.csv`,
`F8` (error analysis).

## 5. Related Work
Calibration (temperature scaling, ECE/NLL/Brier definitions), threshold
selection, chest X-ray multi-label classification, Grad-CAM. No claims of
novelty; reference framing only. Artifacts: `table_06_calibration.csv`
(metric definitions instantiated).

## 6. Dataset
NIH ChestX-ray14: 109,312 images, 29,720 patients; split 70/15/15
patient-level (train 76,977 / val 16,451 / test 15,884); labels
Cardiomegaly + Effusion. Source: `table_01_dataset.csv`, `F2`
(dataset distribution), `data/processed/split_index.csv`.

## 7. Preprocessing
224×224, ImageNet mean/std normalisation; augmentation on train only
(`augmentation` in `reproducibility_manifest.json`). Sources:
`table_02_baseline.csv` header row doc, `reproducibility_manifest.json`.

## 8. Baseline Model
DenseNet121 (torchvision, ImageNet-1K V1 pretrained), head `Linear(1024, 2)`,
BCEWithLogitsLoss with positive-class weighting, Adam, 10 epochs,
checkpoint selected by max mean validation AUROC (epoch 8). Sources:
`table_02_baseline.csv`, `table_03_main_results.csv` (arm A),
`reproducibility_manifest.json`.

## 9. Proposed Decision Framework
Frozen-model pipeline: DenseNet121 → per-label logits → temperature scaling
(σ(z_c / T_c)) → validation-frozen class-specific threshold policy. Fit
discipline: NIH train → NIH val (fit T and τ) → freeze → NIH test. Uses both
raw and calibrated probabilities; monotone sigmoid ⇒ identical confusion
matrices at fixed τ=0.50 across arms. Sources: `F1` (architecture schematic),
`table_05_thresholds.csv`, `table_06_calibration.csv`.

## 10. Experimental Setup
Arms A (baseline), B (calibration only), C (threshold only), D (calibration +
threshold); five policies; evaluation on the held-out NIH test split only.
Sources: `table_04_ablation.csv`, `table_05_thresholds.csv`,
`draft_results.md` (Ablation section).

## 11. Evaluation Metrics
AUROC, AUPRC, F1, precision, recall/sensitivity, specificity (fixed-τ
confusion-based); calibration: NLL, Brier, ECE (equal-width, 15 bins, empty
bins contribute 0), MCE; statistical: bootstrap CIs (2,000 resamples, 95%,
seed 42, image-level). Sources: `table_02_baseline.csv`,
`table_06_calibration.csv`, `confidence_intervals.csv` (final_results),
`reproducibility_manifest.json`.

## 12. Results
Main results — arm A vs arm D per class. Sources: `table_03_main_results.csv`,
`draft_results.md`.

## 13. Ablation Study
A/B/C/D arms; F1 and precision columns; ECE and Brier across arms.
Sources: `table_04_ablation.csv`, `draft_results.md` (Ablation).

## 14. Error Analysis
Cardiomegaly 563 errors (274 high-confidence, 48.7% of errors); Effusion
2,268 errors (8.5% high-confidence); Jaccard overlap 0.131; 4,449 error
images across 1,317 patients; worst patient 60 error images.
Sources: `table_07_error_analysis.csv`, `F8`, `draft_results.md`
(Error Analysis).

## 15. Explainability Analysis
16 deterministic Grad-CAM cases (denseblock4); 43-case bounding-box sanity
cohort, pointing-game 21/43, mean concentration ratio 2.60, 38/43 above 1.
Exploratory only — not clinical localization validation.
Sources: `table_08_explainability.csv`, `F9`, `draft_results.md`
(Explainability).

## 16. Discussion
Main finding (threshold policies), calibration finding (mixed: NLL/Brier
improve, ECE rises), error finding, explainability boundary, external
validation gap. Sources: `draft_discussion.md`, `table_04_ablation.csv`,
`table_06_calibration.csv`.

## 17. Limitations
Single primary dataset; external validation pending; two diagnostic labels;
class imbalance; retrospective public dataset; noise in automatically
derived labels; explainability not clinical validation; no prospective
clinical evaluation; no regulatory validation. Source:
`draft_limitations.md`.

## 18. Conclusion
Direct answer to the research question: threshold policies improve operating
characteristics; calibration has mixed effects; the combined method is not
claimed universally superior. Source: `conclusion.md`.

## 19. Future Work
External validation on the official CheXpert cohort; class-specific
temperature/capacity analysis; sequential/cost-sensitive joint decisions;
prospective or adjudicated-label evaluation.

## 20. References
Cross-references to the evidence artifacts: `paper_evidence_map.json`,
`claim_audit.csv`, `reproducibility_manifest.json`, `master_results.csv`,
`verification_report.json`, `research_snapshot.json`. No numeric claims;
every number used elsewhere is audited.