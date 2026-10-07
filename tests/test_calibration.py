"""Tests for src/calibration.py — ECE/Brier/reliability and temperature scaling.

Run:  python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.calibration import (
    TemperatureScaler,
    brier_score,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    fit_temperatures,
    negative_log_likelihood,
    reliability_bins,
    sigmoid,
)


class TestECEAndBrier(unittest.TestCase):
    def test_perfectly_calibrated_single_bin_has_zero_ece(self):
        y = np.array([0, 1])
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0, places=12)

    def test_certain_and_wrong_has_unit_ece(self):
        y = np.zeros(10)
        p = np.ones(10)  # claims 100 % confident, always wrong
        # probabilities are clipped to 1−1e-12 internally, hence 9 places
        self.assertAlmostEqual(expected_calibration_error(y, p, n_bins=10), 1.0, places=9)

    def test_ece_bounds(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 5000)
        p = rng.random(5000)
        ece = expected_calibration_error(y, p)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_brier_matches_sklearn(self):
        from sklearn.metrics import brier_score_loss

        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        self.assertAlmostEqual(brier_score(y, p), brier_score_loss(y, p), places=12)

    def test_brier_decomposes_into_uncalibrated_terms(self):
        y = np.array([0, 1])
        p = np.array([0.2, 0.8])
        self.assertAlmostEqual(brier_score(y, p), np.mean([0.04, 0.04]), places=12)

    def test_reliability_bins_shapes_and_empty_bins(self):
        y = np.array([0, 1])
        p = np.array([0.1, 0.9])
        b = reliability_bins(y, p, n_bins=10)
        self.assertEqual(b["bin_edges"].shape, (11,))
        self.assertEqual(b["count"].shape, (10,))
        self.assertEqual(int(b["count"].sum()), 2)
        empty = b["count"] == 0
        self.assertTrue(np.all(np.isnan(b["empirical_accuracy"][empty])))
        self.assertTrue(np.all(np.isnan(b["mean_confidence"][empty])))

    def test_calibration_report_keys(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 300)
        p = rng.random(300)
        r = calibration_report(y, p)
        for k in ("ece", "mce", "brier", "nll", "n", "prevalence", "mean_confidence"):
            self.assertIn(k, r)
        self.assertEqual(r["n"], 300)


class TestTemperatureFitting(unittest.TestCase):
    def test_recovers_known_temperature(self):
        """Scores generated with T=3 → the fit should find T ≈ 3."""
        rng = np.random.default_rng(0)
        z = rng.normal(0, 2.0, 20000)
        y = (rng.random(20000) < sigmoid(z / 3.0)).astype(float)  # true T = 3
        t = fit_temperature(z, y)
        self.assertGreater(t, 2.4)
        self.assertLess(t, 3.7)

    def test_overconfident_model_gets_t_above_one(self):
        """Perfectly separable data with exaggerated logits should shrink (T > 1)."""
        rng = np.random.default_rng(1)
        z = rng.normal(0, 1.0, 5000)
        y = (rng.random(5000) < sigmoid(z * 0.2)).astype(float)  # signal much weaker than logits
        self.assertGreater(fit_temperature(z, y), 1.0)

    def test_temperature_reduces_nll(self):
        rng = np.random.default_rng(2)
        z = rng.normal(0, 5.0, 8000)
        y = (rng.random(8000) < sigmoid(z / 4.0)).astype(float)
        t = fit_temperature(z, y)
        nll_before = negative_log_likelihood(y, sigmoid(z))
        nll_after = negative_log_likelihood(y, sigmoid(z / t))
        self.assertLessEqual(nll_after, nll_before + 1e-9)

    def test_degenerate_inputs_return_one(self):
        self.assertEqual(fit_temperature(np.zeros(50), np.zeros(50)), 1.0)   # constant logits
        self.assertEqual(fit_temperature(np.random.default_rng(0).normal(size=50),
                                        np.zeros(50)), 1.0)                   # single class

    def test_fit_temperatures_shape(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(0, 3, (3000, 2))
        y = (rng.random((3000, 2)) < sigmoid(logits / 2.0)).astype(float)
        t = fit_temperatures(logits, y)
        self.assertEqual(t.shape, (2,))
        self.assertTrue(np.all(t > 0))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fit_temperature(np.zeros(10), np.zeros(9))


class TestTemperatureScaler(unittest.TestCase):
    def _toy(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 4.0, (6000, 2))
        y = (rng.random((6000, 2)) < sigmoid(logits / 3.0)).astype(float)
        return logits, y

    def test_monotone_invariance_of_ranking_metrics(self):
        from src.metrics import auroc, auprc

        logits, y = self._toy()
        scaler = TemperatureScaler(["a", "b"]).fit(logits, y, split="val")
        cal = scaler.probabilities(logits)
        for i, name in enumerate(["a", "b"]):
            raw = sigmoid(logits[:, i])
            self.assertAlmostEqual(auroc(y[:, i], raw), auroc(y[:, i], cal[:, i]), places=10)
            self.assertAlmostEqual(auprc(y[:, i], raw), auprc(y[:, i], cal[:, i]), places=8)

    def test_fit_refused_on_evaluation_splits(self):
        logits, y = self._toy()
        scaler = TemperatureScaler(["a", "b"])
        for bad in ("test", "val_vs_test", "external", "CheXpert"):
            with self.assertRaises(ValueError, msg=f"split {bad} must be refused"):
                scaler.fit(logits, y, split=bad)

    def test_fit_on_val_is_allowed(self):
        logits, y = self._toy()
        scaler = TemperatureScaler(["a", "b"]).fit(logits, y, split="val")
        self.assertEqual(scaler.metadata_["split"], "val")
        self.assertEqual(scaler.metadata_["n_rows"], 6000)
        self.assertEqual(len(scaler.temperatures_), 2)

    def test_save_load_roundtrip(self):
        logits, y = self._toy()
        scaler = TemperatureScaler(["a", "b"]).fit(logits, y, split="val")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.json"
            scaler.save(path)
            loaded = TemperatureScaler.load(path)
        np.testing.assert_array_equal(scaler.temperatures_, loaded.temperatures_)
        np.testing.assert_allclose(
            scaler.transform(logits), loaded.transform(logits), rtol=0, atol=0
        )

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TemperatureScaler(["a"]).transform(np.zeros((3, 1)))

    def test_transform_column_mismatch_raises(self):
        logits, y = self._toy()
        scaler = TemperatureScaler(["a", "b"]).fit(logits, y, split="val")
        with self.assertRaises(ValueError):
            scaler.transform(logits[:, :1])

    def test_one_dimensional_logits_supported(self):
        logits, y = self._toy()
        scaler = TemperatureScaler(["a"]).fit(logits[:, 0], y[:, 0], split="val")
        out = scaler.transform(logits[:, 0])
        self.assertEqual(out.shape, (6000,))

    def test_nll_reported_before_after(self):
        logits, y = self._toy()
        scaler = TemperatureScaler(["a", "b"]).fit(logits, y, split="val")
        for lbl in ("a", "b"):
            self.assertLessEqual(
                scaler.metadata_["nll_after"][lbl], scaler.metadata_["nll_before"][lbl] + 1e-9
            )


if __name__ == "__main__":
    unittest.main()
