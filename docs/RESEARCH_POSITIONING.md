# Research Positioning

This work addresses automated chest radiograph classification with frozen backbone and calibration-aware threshold policies, evaluated under strict reproducibility (patient-level splits, frozen validation-fitted hyperparameters).

Key positioning notes:
- All reported numbers are extracted from frozen artifacts (master_results.csv, patient-level bootstrap/DeLong, calibration reports). No retraining or fabrication occurs.
- Patient-level cluster bootstrapping (resampling unit = patient) is used for CIs to account for intra-patient correlation; this widens CIs compared to image-level bootstraps. We report both transparently in artifacts (confidence_intervals.csv image-level, confidence_intervals_patient.csv patient-level).
- External validation remains PENDING (no CheXpert cohort available). Pipeline is implemented (src/external_eval.py) and results are explicitly absent.
- Prevalence-shift analysis follows the stated constraint: positives kept in full, negatives subsampled only to raise prevalence; below-natural targets are infeasible by design and are reported as not feasible (8 simulated above-natural, 6 infeasible below-natural).
- Logistic calibration is strictly monotone (a = exp(c)>0) on validation-only fits, so AUROC/AUPRC are invariant by construction. Decision policies are frozen from validation (operating-point delta, not threshold refitting).
- Arms E/F (logistic+fixed and logistic+f1-optimal) are reported as extension experiments and are never part of the published A–D ablation; the master results table for the ablation uses A–D only.
- We do not claim first/novel/state-of-the-art performance. Results are reproducible from the released artifacts. See also Rajaraman/Ganesan/Antani 2022 and established literature for context.

This statement is traceable to evidence in outputs/final_results/, outputs/metrics/, and the paper package (table_14, figure_14, IEEE_READINESS_REPORT.md).
