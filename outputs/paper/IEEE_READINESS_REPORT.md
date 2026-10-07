# IEEE Readiness Report

## Summary
All primary statistical artifacts are present and verified. Patient-level inference, permutation+Holm, threshold stability, ECE sensitivity/grid, logistic calibration, extension arms, prevalence shift (correct feasibility), and updated evidence/tracing are in place.

## Key Verified Facts
- Primary endpoints (patient-level bootstrap + permutation/Holm): Cardiomegaly ΔF1 D-A +0.0700 [0.0280,0.1065] p=0.0002 Holm=0.0004 (significant); Effusion +0.0191 [0.0050,0.0326] p=0.0262 Holm=0.0262 (significant). Bootstrap CIs exclude zero.
- DeLong AUROC (test): Cardiomegaly 0.8972 [0.8821,0.9124] (SE 0.0077, n_pos 415/n_neg 15469); Effusion 0.8587 [0.8507,0.8667] (SE 0.0041, n_pos 1997/n_neg 13887).
- Patient-level CI widening (arm-D F1, calibrated f1-optimal): Cardiomegaly 0.1361 vs 0.0707 (image-level); Effusion 0.0590 vs 0.0306. Reported in confidence_intervals_patient.csv (172 rows, 5000 resamples, seed 42, percentile CI, patient cluster).
- Threshold stability: 2000 val bootstraps, vectorized==frozen fits (33 checks, atol 1e-9). First fits: raw f1-optimal Cardio 0.9021 (CI [0.8275,0.9140], width 0.0864, ±5% false, ±10% true); calibrated 0.8666 (±5% false, ±10% true); logistic 0.1680 (±5% false, ±10% false). Honest reporting.
- ECE (reference 15 bins/equal-width): Cardio raw 0.1061, calibrated 0.1169, logistic 0.0029; Eff 0.2077, 0.2292, 0.0136. Grid 10/15/20 × equal-width/equal-freq (36 rows) consistent; logistic validation NLL drops (Cardio 0.2337→0.0845, Eff 0.4889→0.2650), validation ECE 0.0024/0.0083, AUROC/AUPRC invariant.
- Extension: A==B (fixed 0.5 monotone), C==D==F (f1-optimal robust to monotone calibration); E (logistic+0.5) Cardio F1 0.1376 (prec 0.6400, rec 0.0771), Eff 0.3222 (prec 0.6030, rec 0.2198). E/F excluded from A–D ablation.
- Prevalence: observed Cardio 0.0261 (415/15469), Eff 0.1257 (1997/13887). 14 cells total: 8 simulated (above-natural: Cardio 5/10/20/30/50%, Eff 20/30/50%) and 6 below-natural dropped as infeasible (positives kept in full, negatives subsampled only raises prevalence). F1/precision rise with prevalence at frozen operating point; recall pinned (0.3711/0.5383).
- External: BLOCKED/PENDING; no metrics computed; explicitly documented.
- Evidence: 111 claims verified (evidence map); audit columns claim_id/claim_type/source_table_or_figure; 14 tables, 14 figures, 11 prose docs. Number hygiene passes.
- Reproducibility: frozen artifacts only, git status clean (excluded re-generated manifests/snapshots), deterministic runs (seed 42), matplotlib Agg.

## Artifacts
- evidence_map: outputs/paper/paper_evidence_map.json (111 claims, all verified)
- audit: outputs/paper/claim_audit.csv (claim_id, claim_type, source_table_or_figure)
- tables 10–14: patient bootstrap (172), arm differences (40), threshold stability (24), calibration comparison (6), prevalence shift (14)
- figures 10–14: calibration comparison, ece sensitivity, threshold stability, decision policy, prevalence sensitivity
- statistical_report.json, delong_auroc.json, paired_tests.json, logistic_calibration_report.json, ext_extension_arms.json, prevalence_shift.json, decision_policy_analysis.csv

## Verification
All tests pass: paper package tests (20/20), final results (47/47), statistical artifacts validated against ground-truth values from artifacts.

This report is prepared for IEEE submission readiness. No results were fabricated or retrained.
