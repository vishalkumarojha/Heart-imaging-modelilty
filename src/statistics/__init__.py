"""Statistical machinery for the IEEE-upgrade decision-policy analysis.

Patient-level resampling is the unifying theme. Images from the same patient
are *dependent* (correlated finding status inside one scan session), so every
bootstrap / permutation below resamples patients — never rows — and the 
confidence intervals are *paired* across calibration arms because each arm is
evaluated on the identical patient resample.

Modules
-------
data.py                   frozen prediction loaders + per-variant probabilities
vectorized_thresholds.py  policies replicated without the scalar candidate loop
bootstrap.py              patient-level cluster bootstrap (CIs + paired deltas)
delong.py                 DeLong AUROC variance / CI / two-curve test
tests.py                  paired patient-level permutation + Holm-Bonferroni

Run order (see src/statistical_upgrade.py): the logistic scaler and its
validation thresholds must be fitted BEFORE the patient bootstrap, which needs
the 'logistic' variant probabilities on the test split.

    python -m src.statistics.bootstrap
    python -m src.statistics.delong
    python -m src.statistics.tests
"""
from . import bootstrap, data, delong, tests, vectorized_thresholds

__all__ = ["bootstrap", "data", "delong", "tests", "vectorized_thresholds"]