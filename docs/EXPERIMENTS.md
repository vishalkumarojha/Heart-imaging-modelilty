# Experiments 1–4 (measured, frozen)

All four experiments run from **one cached test prediction table**
(`outputs/predictions/raw/test__densenet121_best__35965f610c8b.csv`). Raw and
calibrated variants differ only by the frozen per-label temperature
(`outputs/metrics/calibration/temperature_scalers.json`). Parameters are fit on
validation and never on test (enforced by `src.fit_parameters` and tests).

Artifacts: `outputs/metrics/experiments/exp{1,2,3,4}_*.json` +
`experiments_summary.csv`.

## Experiment 1 — calibration (temperature scaling)

`exp1_calibration.json`, `splits.{test,val}`, per label `{raw, calibrated}`.

Test-split (n = 15 884):

| label | NLL raw→cal | Brier raw→cal | ECE raw→cal | AUROC/PRC (invariant) |
|---|---|---|---|---|
| Cardiomegaly | 0.22967 → 0.22617 | 0.06808 → 0.06776 | 0.10614 → 0.11688 | 0.8972 / 0.3043 |
| Effusion | 0.48193 → 0.46902 | 0.15615 → 0.15332 | 0.20774 → 0.22916 | 0.8587 / 0.4700 |

Temperatures (fit on val NLL): **1.1865** (Cardiomegaly), **1.3979** (Effusion).

**Findings (all reproduced by `tests/test_experiment_outputs.py`):**
* Temperature scaling improves **NLL and Brier** but **raises binned ECE** —
  NLL and binned ECE are different objectives.
* ECE is essentially **bin-invariant** (5/10/15/20 → same value to 4 dp)
  because every confidence bin is over-confident, giving
  `ECE ≈ mean(p) − prevalence` (Cardio 0.132 − 0.026 ≈ 0.106).
* AUROC/AUPRC are **exactly invariant** to temperature (asserted by a test).

## Experiment 2 vs 3 — threshold policies, raw vs calibrated

`exp2_thresholds_raw.json` (raw probs, raw τ) and
`exp3_thresholds_calibrated.json` (calibrated probs, calibrated τ),
`evaluation.test.<label>.<policy>`.

| policy | Cardio F1 (test) | Effusion F1 (test) |
|---|---|---|
| fixed (0.5) | 0.2836 | 0.4676 |
| **f1_optimal** | **0.3536** | **0.4866** |
| youden | 0.1852 | 0.4545 |
| sensitivity_constrained (recall ≥ 0.90) | 0.1281 | 0.4123 |
| precision_constrained (precision ≥ 0.50) | 0.2559 | 0.4673 |

* The F1-optimal policy is the best operating point of the five on test; the
  constrained policies document the rule-out / rule-in trade-offs.
* **exp3 reproduces exp2's confusion matrices exactly** (monotone equivalence —
  asserted by test; not a bug). Only the numeric τ differ (raw vs scaled space).

## Experiment 4 — robustness

`exp4_robustness.json`:
* **ECE binning sweep** (5/10/15/20 bins): stable to 4 dp (see Experiment 1).
* **Constraint target sweeps**: how the constrained thresholds shift as the
  requested recall / precision target changes.
* **Bootstrap threshold stability** (10 resamples): the τ estimates are stable.

## Master roll-up (final phase)

`outputs/final_results/master_results.csv` — 24 rows encoding the four ablation
arms **A** baseline (raw, 0.5) · **B** calibration-only (calibrated, 0.5) ·
**C** threshold-only (raw, each policy) · **D** combined (calibrated, each
policy), per label. The headline comparison A→D:

* Cardiomegaly F1 0.2836 → **0.3536** (Δ +0.0700), precision 0.1787 → 0.3377.
* Effusion F1 0.4676 → **0.4866** (Δ +0.0191), precision 0.3320 → 0.4440.
* AUROC/AUPRC unchanged (temperature-invariant, by design).

Bootstrap 95% CIs for every master metric: `confidence_intervals.csv`
(2 000 resamples, seed 42, resampling unit = test image).

## Reproduction

```bash
.venv/bin/python -m src.experiments --exp all        # recompute exp 1-4
.venv/bin/python -m src.master_results                # roll up the master table
.venv/bin/python -m src.uncertainty                   # bootstrap CIs
```