# Faculty Presentation Slide Plan

**Target:** 15 minutes. Use one message per slide, restrained academic formatting, and large figure labels. Core figures/tables are under `outputs/paper/`. Keep full methodology tables as backup material. Each slide includes exact on-screen bullets, verbal emphasis, a caution, timing, and a likely question.

## 1. Title and scope — 15 s
**Show:** title; NIH ChestX-ray14; Cardiomegaly + Effusion; “retrospective research study.”
**Say:** “I study how calibration and threshold policies affect one fixed chest X-ray classifier.”
**Use:** no figure. **Avoid:** clinical system wording. **Likely question:** What did you build?

## 2. Why the problem matters — 25 s
**Bullets:** “Class imbalance”; “Scores need an operating cutoff”; “Ranking, probability quality, and decisions differ.”
**Say:** “A ranking can be useful even when a 0.5 cutoff gives an undesirable precision/recall tradeoff.”
**Use:** simple three-part schematic. **Avoid:** claiming clinical workflow needs from this study. **Question:** Why not just use AUROC?

## 3. Research question — 20 s
**Bullets:** “Fixed DenseNet121”; “Two NIH labels”; “Calibration × validation-frozen thresholds.”
**Say:** state the question as written in `outputs/paper/research_question.md`.
**Use:** research question artifact. **Avoid:** general medical-AI law. **Question:** What is the hypothesis?

## 4. Dataset overview — 30 s
**Bullets:** “109,312 available images”; “29,720 patients”; “Cardiomegaly / Effusion.”
**Use:** Figure 2, Table 1. **Say:** counts come from available image inventory and local split. **Avoid:** implying this represents every hospital. **Question:** Why these two labels?

## 5. Patient-level split — 30 s
**Bullets:** “Train 76,977 / 20,797 patients”; “Val 16,451 / 4,468”; “Test 15,884 / 4,455”; “Patient overlap = 0.”
**Use:** Table 1. **Say:** all images from a patient stay in one partition. **Avoid:** claiming no other bias. **Question:** Why not image-level random split?

## 6. Label and data caveats — 20 s
**Bullets:** “Report-derived labels”; “2,808 metadata images unavailable”; “Test Effusion support differs by one because of documented read substitution.”
**Use:** Table 1 + source note. **Say:** transparently identify 1,996 split target positives vs 1,997 prediction support. **Avoid:** silently reconciling counts. **Question:** How reliable are labels?

## 7. Fixed baseline architecture — 30 s
**Bullets:** “DenseNet121”; “ImageNet-1K V1”; “Global pool → Linear 1024→2 logits”; “Frozen for A/B/C/D.”
**Use:** Figure 1. **Say:** known architecture used to isolate post-hoc decision analysis. **Avoid:** novel/best architecture. **Question:** Why DenseNet?

## 8. Training and reproducibility — 30 s
**Bullets:** “224×224”; “5 epochs partially frozen + 5 fully fine-tuned”; “Adam 1e-4 → 1e-5”; “Best mean validation AUROC; seed 42.”
**Use:** Table 2 baseline/config source. **Say:** all results refer to hash-identified checkpoint. **Avoid:** inventing reason for 42. **Question:** How was checkpoint selected?

## 9. Literature positioning — 45 s
**Bullets:** “DenseNet / NIH are established”; “Temperature scaling is established”; “Rajaraman et al. studied calibration and operating thresholds in imbalanced medical images.”
**Use:** short literature comparison. **Say:** the current work is narrower, fixed-model decomposition with patient-level uncertainty and added sensitivity analyses. **Avoid:** “first” or “we discovered thresholding.” **Question:** How does this differ from Rajaraman et al.?

## 10. Method pipeline — 30 s
**Bullets:** “Train model”; “Fit T and τ on validation”; “Freeze”; “Evaluate on test.”
**Use:** Figure 1. **Say:** test labels never select temperatures or thresholds. **Avoid:** test-driven tuning. **Question:** Where could leakage occur?

## 11. Probability calibration — 35 s
**Bullets:** “T_Cardio=1.1865”; “T_Effusion=1.3979”; “p=σ(z/T)”; “T fitted by validation NLL.”
**Use:** Figure 5, Table 6. **Say:** monotone map changes scale but preserves rank. **Avoid:** claiming every calibration metric improved. **Question:** Why does AUROC stay fixed?

## 12. Calibration results — 35 s
**Bullets:** “Test NLL/Brier improve”; “Reference 15-bin ECE worsens”; “MCE improves”; “ECE depends on bins.”
**Use:** Table 6, Figure 11. **Say:** calibration conclusion is metric-dependent. **Avoid:** hiding unfavorable ECE. **Question:** Why do metrics disagree?

## 13. Threshold policies — 35 s
**Bullets:** “Fixed 0.50”; “F1-optimal”; “Youden”; “Sensitivity ≥0.90”; “Precision ≥0.50.”
**Use:** Table 5/Figure 13. **Say:** each threshold represents a different validation objective. **Avoid:** calling one clinically optimal. **Question:** Why use F1?

## 14. A/B/C/D design — 45 s
**Bullets:** “A raw + 0.50”; “B temperature + 0.50”; “C raw + val F1 threshold”; “D temperature + val F1 threshold.”
**Use:** Table 4. **Say:** no retraining; the design separates probability mapping and operating cutoff. **Avoid:** redefining arms. **Question:** Which contrasts isolate each component?

## 15. Main test results — 55 s
**Bullets:** “Cardio F1 .2836→.3536”; “Effusion .4676→.4866”; “Precision rises”; “Recall falls.”
**Use:** Table 3/Figure 7. **Say:** quote both benefit and tradeoff. **Avoid:** “D is best for clinical use.” **Question:** Why does recall drop?

## 16. Statistical evidence — 45 s
**Bullets:** “5,000 patient-cluster bootstrap”; “Paired arms”; “D−A F1 CI excludes 0 for both”; “Holm-adjusted p .0004 / .0262.”
**Use:** Table 11. **Say:** evidence is conditional on test split; statistical ≠ clinical significance. **Avoid:** causal generalization. **Question:** Why patient-level resampling?

## 17. Threshold stability — 30 s
**Bullets:** “2,000 validation patient bootstraps”; “Cardio calibrated F1 τ CI .7860–.8799”; “Effusion .6434–.7444”; “within ±10%, not ±5%.”
**Use:** Figure 12/Table 12. **Say:** useful but imperfect stability. **Avoid:** perfect robustness. **Question:** Would another validation sample choose same τ?

## 18. Error analysis — 30 s
**Bullets:** “Cardio: 274/563 errors ≥.9 confidence”; “Effusion: 193/2,268”; “Errors cluster in patients.”
**Use:** Figure 8/Table 7. **Say:** confidence does not guarantee correctness. **Avoid:** attributing all errors to calibration. **Question:** How do you interpret high-confidence errors?

## 19. Grad-CAM — 25 s
**Bullets:** “Denseblock4”; “16 examples”; “43 boxed cases; pointing 21/43”; “Exploratory.”
**Use:** Figure 9/Table 8. **Say:** visual sanity check only. **Avoid:** localization proof. **Question:** Does it prove attention to the heart?

## 20. Prevalence sensitivity — 25 s
**Bullets:** “Frozen predictions”; “Negative-only subsampling”; “8 feasible target cells”; “Not external validation.”
**Use:** Figure 14/Table 14. **Say:** prevalence affects precision/F1 while recall remains fixed by kept positives. **Avoid:** calling simulated cohort external. **Question:** Why are lower targets infeasible?

## 21. UI demonstration — 60 s
**Bullets:** “Read-only evidence tabs”; “Frozen live inference”; “Presentation / Technical modes”; “Research demo only.”
**Use:** dashboard, tabs Overview → Calibration → A/B/C/D → Statistics → Limitations. **Say:** app applies saved checkpoint/T/τ, outputs scores and policy labels, not diagnosis. **Avoid:** suggesting UI retrains or produces live Grad-CAM. **Question:** What happens to an uploaded image?

## 22. Limitations and conclusion — 30 s
**Bullets:** “Single NIH split”; “Two labels”; “External validation pending”; “No clinical/prospective evaluation.”
**Say:** “Threshold policy changed operating metrics in this fixed setup; calibration had mixed metric effects; external validation is next.” **Use:** Table 9. **Avoid:** state-of-the-art or clinical claims. **Question:** Why was external validation not completed?

**15-minute timing:** these slide targets sum to about 11 minutes; use the remaining time for transitions, figure reading, and a 90-second UI path. Do not speed-read. The talk may run 14–16 minutes depending on demo reliability.

## 10-minute version
Combine slides 4–6, 7–8, 11–13 and 18–20. Keep A/B/C/D, primary paired results, external limitation, and a 60-second UI overview. Target: opening/problem 1 min; data/model/literature 2 min; method 2 min; results/statistics 2.5 min; UI 1 min; limitations/conclusion 1.5 min.

## 20-minute version
Use all 22 slides; expand Rajaraman comparison, five policy results, patient-vs-image CI widening, ECE bin grid, and UI path. Add backup slides for complete results table, all thresholds, logistic extension, prevalence rows, and data support discrepancy. Additional time is explanation depth, not stronger claims.

## Backup slides
1. Full model training configuration and provenance.
2. All five policy thresholds and test metrics (`table_05_thresholds.csv`).
3. All bootstrap metric intervals (`table_10_patient_bootstrap_ci.csv`).
4. All paired arm deltas (`table_11_arm_differences.csv`).
5. Threshold stability by policy (`table_12_threshold_stability.csv`).
6. Logistic calibration extension and ECE bin grid (`table_13_calibration_comparison.csv`).
7. Prevalence sensitivity grid (`table_14_prevalence_shift.csv`).
8. Error counts, patient concentration, and cross-label overlap.
9. External-validation status.
10. Source traceability / claim evidence map.
