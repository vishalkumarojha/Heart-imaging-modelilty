# Contribution Statement

This work does **not** claim a novel DenseNet architecture, a novel
calibration algorithm, a novel thresholding algorithm, clinical validation,
or state-of-the-art performance. Its contribution is a controlled,
reproducible experimental study. Concretely:

1. **A reproducible baseline study** of a DenseNet121 multi-label chest X-ray
   classifier (Cardiomegaly + Effusion, NIH ChestX-ray14, 70/15/15
   patient-level split, seed 42), with all artifacts frozen and machine-
   verifiable.

2. **A controlled evaluation of probability calibration and class-specific
   decision policies** applied to the frozen baseline: per-label post-hoc
   temperature scaling fit on validation only, and five validation-frozen
   threshold policies (fixed, F1-optimal, Youden, sensitivity-constrained,
   precision-constrained).

3. **An ablation** isolating the contribution of calibration versus threshold
   selection (arms A–D), showing the F1/precision gains are primarily
   attributable to threshold selection.

4. **Error and explainability analysis** of high-confidence classification
   failures on the held-out test split, including cross-label error overlap,
   patient-level error concentration, and a deterministic 16-case Grad-CAM
   analysis with a 43-case bounding-box sanity cohort.

5. **Explicit reporting of the missing external-validation evidence**: the
   paper states clearly that external validation is pending because the
   required external cohort was unavailable, and no external performance is
   reported.

**Honesty constraints binding this package**

- Every number is traceable to an artifact (`paper_evidence_map.json`,
  `claim_audit.csv`).
- No result is described as clinically validated.
- No external-validation figure or metric is generated.
- The unfavorable ECE result is reported, not hidden.