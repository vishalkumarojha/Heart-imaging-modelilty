# IEEE Readiness

The paper package has been strengthened for IEEE submission with the following additions:

- Patient-level statistical inference (cluster bootstrap by patient; DeLong AUROC) with explicit CI widening relative to image-level bootstraps.
- Primary endpoint validation (A vs D ΔF1) with paired permutations and Holm correction; both endpoints significant with CIs excluding zero.
- Threshold stability characterization (2000 val bootstraps, vectorized refit equality to frozen fits, ±5%/±10% bands reported honestly).
- Calibration comparison (reference + grid ECE, logistic Platt fitted on validation only, monotonicity => AUROC/AUPRC invariant).
- Decision-policy and extension arms documented clearly (E/F excluded from core ablation).
- Prevalence sensitivity with feasibility constraint: below-natural prevalence infeasible under "positives kept in full + negative subsampling" (reported transparently).
- Updated paper_build to emit 14 tables, 14 figures, evidence map with claim_type and source_table_or_figure, expanded claim audit columns, at_least support for test counts.
- Full test suite passes (paper package + final results tests). All prose numbers trace to verified artifacts or documented constants.

External validation remains PENDING. No results fabricated. All values derive from frozen artifacts.
