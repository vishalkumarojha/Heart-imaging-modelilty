# Abstract

**Title.** Calibration-Aware Decision Policies for Multi-Label Chest X-Ray
Classification Under Class Imbalance.

## Background

Chest X-ray triage benefits from classifiers that operate at an explicitly
chosen operating point. Under class imbalance, the default 0.50 probability
threshold may be far from a usable precision/recall trade-off, and post-hoc
calibration is often proposed to improve the probabilities behind such
decisions.

## Problem

Whether calibration-aware, class-specific decision policies improve the
operating characteristics of a fixed DenseNet121 multi-label chest X-ray
classifier under class imbalance — and how much of any improvement is due to
calibration versus threshold selection.

## Method

A frozen DenseNet121 baseline (NIH ChestX-ray14, 109,312 images, 29,720
patients, 70/15/15 patient-level split, seed 42; Cardiomegaly + Effusion) is
augmented with per-label temperature scaling (T = 1.187 / 1.398) and five
validation-frozen class-specific threshold policies. A four-arm ablation (A
baseline, B calibration only, C threshold only, D both) isolates each
component. All thresholds and temperatures are fit on validation only; every
reported number is measured on the held-out test split.

## Experiments

Baseline Cardiomegaly AUROC = 0.8972, Effusion AUROC = 0.8587. The validation-
frozen F1-optimal policy raises test F1 from 0.2836 to 0.3536 (Cardiomegaly)
and from 0.4676 to 0.4866 (Effusion), and precision from 0.1787 to 0.3377 and
from 0.3320 to 0.4440. Temperature scaling improves NLL and Brier but increases
ECE (Cardiomegaly 0.1061 to 0.1169; Effusion 0.2077 to 0.2292). Because the
sigmoid is monotone, arm B (calibration only) leaves the fixed-0.50 confusion
matrices unchanged; the F1 gain is attributed to threshold selection. Error
analysis finds 274 of 563 Cardiomegaly errors are high-confidence (≥ 0.9, 48.7%
of errors).

## Limitations

Single primary dataset; external validation pending because the external
cohort was unavailable; two diagnostic labels; class imbalance; retrospective
automatically labeled data; explainability not clinical validation; no
prospective or regulatory evaluation.

## Conclusion

Validation-frozen, class-specific threshold policies improve the operating
characteristics of a fixed imbalanced multi-label classifier; temperature
scaling has mixed effects on probability quality and does not independently
account for the F1 gain. No external-validation performance is claimed.