# Conclusion

The research question — *can calibration-aware, class-specific decision
policies improve the operating characteristics of a fixed DenseNet121
multi-label chest X-ray classifier under class imbalance?* — is answered
directly and conditionally.

**Yes, with respect to the threshold policy.** A validation-frozen,
class-specific F1-optimal threshold raised test F1 from 0.2836 to 0.3536
(Cardiomegaly) and from 0.4676 to 0.4866 (Effusion), and precision from
0.1787 to 0.3377 and from 0.3320 to 0.4440. These gains come from operating-
point choice on an unchanged model, and are reproduced in the ablation arms C
and D.

**But calibration has mixed effects.** Per-label temperature scaling improved
NLL and Brier (Cardiomegaly NLL 0.2297→0.2262, Brier 0.0681→0.0678; Effusion
NLL 0.4819→0.4690, Brier 0.1561→0.1533) yet increased ECE in this
implementation (0.1061→0.1169 and 0.2077→0.2292). It did not change any
fixed-threshold operating point, so it did not independently explain the F1
gain.

The combined method is therefore **not described as universally superior**: it
selects a precision-favoring operating point (recall fell from 0.6867 to 0.3711
for Cardiomegaly and from 0.7902 to 0.5383 for Effusion), and a different
operating requirement would select a different policy.

External validation was not completed because the required external cohort was
unavailable; the project is explicitly not represented as externally validated.
The contribution of this work is a controlled, reproducible, artifact-verified
study of decision-policy effects on a fixed model — not a claim of clinical
deployment readiness.