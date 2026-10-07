# Draft — Discussion

## Main finding

The dominant, reproducible improvement in operating characteristics comes from
the **threshold policy**, not from recalibration. On the held-out test split,
switching from a fixed 0.50 cutoff to a validation-frozen F1-optimal class-
specific threshold raised F1 from 0.2836 to 0.3536 (Cardiomegaly) and from
0.4676 to 0.4866 (Effusion), and raised precision from 0.1787 to 0.3377 and
from 0.3320 to 0.4440. These gains are exact operating-point choices on an
already-trained model; they cost no additional training and are reproduced by
arms C and D independently.

## Calibration finding

Per-label temperature scaling improved probability-quality metrics — NLL fell
(Cardiomegaly 0.2297→0.2262; Effusion 0.4819→0.4690) and Brier fell
(Cardiomegaly 0.0681→0.0678; Effusion 0.1561→0.1533) — but **ECE increased**
in this implementation (Cardiomegaly 0.1061→0.1169; Effusion 0.2077→0.2292).
The ECE result is reported as measured, not hidden. Because the sigmoid is
monotone, temperature scaling alone cannot change any fixed-threshold confusion
matrix (arm A vs arm B), so calibration did not independently explain the F1
improvement. The F1 gain is attributed to threshold selection, and the
interpretation of the calibration benefit is restricted to the probability-
quality metrics that improved.

## Error finding

A substantial portion of Cardiomegaly errors remain high-confidence: at the
F1-optimal operating point, 563 errors were made, 274 of which carry confidence
≥ 0.9 (48.7% of the label's errors). Effusion shows a far smaller share (8.5%).
High-confidence Cardiomegaly errors concentrate in relatively few patients
(4,449 error images across 1,317 patients; a worst case of 60 error images for
one patient), which is consistent with a small positive class under imbalanced
training and suggests that decision support should not simply trust the highest
confidences for the rare class.

## Explainability finding

The Grad-CAM analysis (16 deterministic cases; 43-case bounding-box sanity
cohort, pointing-game 21/43, mean concentration ratio 2.60) is **exploratory**
and should not be interpreted as clinical localization validation. Gradient-
based attention on a frozen model is a descriptive sanity check only.

## External validation

Cross-dataset generalization cannot be established: the external cohort was
unavailable, so no external performance was measured. The absence of external
evidence is an explicit limitation, not an implicit strength.

## Overstatement guard

None of the findings are claimed to transfer clinically. The combined method is
not described as universally superior: it trades recall for precision at the
F1-optimal point (Cardiomegaly recall 0.6867→0.3711; Effusion 0.7902→0.5383),
and different operating requirements would select different policies.