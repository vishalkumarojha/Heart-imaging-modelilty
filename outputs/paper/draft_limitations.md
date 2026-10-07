# Draft — Limitations

The following limitations are stated explicitly. No additional limitations are
claimed beyond those actually present in this study.

1. **Single primary dataset evaluation.** All reported numbers come from the
   held-out test split of NIH ChestX-ray14. Results are conditional on this
   dataset and this preprocessing pipeline.

2. **External validation unavailable.** No independent external cohort was
   available, so cross-dataset generalization is not established. External
   validation is PENDING (`table_09_external_validation_status.csv`). The
   project is not represented as externally validated.

3. **Two selected labels.** The system predicts Cardiomegaly and Effusion only;
   the 12 remaining NIH labels were not modeled.

4. **Class imbalance.** Both labels are imbalanced (Cardiomegaly is rare
   relative to Effusion). Threshold policies that maximize F1 do so by trading
   recall for precision, which changes the error profile materially.

5. **Retrospective public dataset.** NIH ChestX-ray14 is retrospective, and
   its labels are automatically mined from radiology reports; label noise is a
   known property of the dataset.

6. **Noise inherent to automatically derived labels.** Downstream conclusions
   inherit this noise; small error counts for rare classes may be affected.

7. **Explainability analysis is not clinical validation.** The Grad-CAM /
   bounding-box sanity cohort is descriptive and does not establish that the
   model localizes pathology the way a clinical device would.

8. **No clinical prospective evaluation.** No prospective reader study or
   clinical deployment setting was evaluated.

9. **No regulatory validation.** No regulatory clearance pathway has been
   followed and none is claimed.