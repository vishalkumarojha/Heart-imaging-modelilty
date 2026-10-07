"""Tests for src/thresholds.py — the five validation-derived policies.

Run:  python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from src.metrics import binary_metrics
from src.thresholds import (
    POLICIES,
    apply_thresholds,
    find_threshold,
    fit_thresholds,
    threshold_f1,
    threshold_precision,
    threshold_sensitivity,
    threshold_youden,
)


def synthetic(n=4000, seed=0, separation=3.0):
    """Scores with real signal so every policy has a meaningful optimum."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    logits = separation * (2 * y - 1) + rng.normal(0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-logits))
    return y, p


class TestDispatcher(unittest.TestCase):
    def test_unknown_policy_raises(self):
        y, p = synthetic()
        with self.assertRaises(ValueError):
            find_threshold(y, p, "brand_new_policy")

    def test_all_documented_policies_are_supported(self):
        y, p = synthetic()
        for policy in POLICIES:
            rec = find_threshold(y, p, policy)
            self.assertEqual(rec["policy"], policy)
            self.assertGreaterEqual(rec["threshold"], 0.0)
            self.assertLessEqual(rec["threshold"], 1.0)
            self.assertIn("fit_metrics", rec)

    def test_aliases_resolve(self):
        y, p = synthetic()
        self.assertEqual(find_threshold(y, p, "f1")["policy"], "f1_optimal")
        self.assertEqual(find_threshold(y, p, "0.5")["policy"], "fixed")
        self.assertEqual(find_threshold(y, p, "recall_constrained")["policy"],
                         "sensitivity_constrained")

    def test_deterministic(self):
        y, p = synthetic()
        for policy in POLICIES:
            a = find_threshold(y, p, policy)["threshold"]
            b = find_threshold(y, p, policy)["threshold"]
            self.assertEqual(a, b)


class TestIndividualPolicies(unittest.TestCase):
    def test_fixed_is_half(self):
        y, p = synthetic()
        self.assertEqual(find_threshold(y, p, "fixed")["threshold"], 0.5)

    def test_f1_optimal_beats_or_equals_f1_at_half(self):
        y, p = synthetic(seed=1)
        tau = threshold_f1(y, p)["threshold"]
        self.assertGreaterEqual(
            binary_metrics(y, p, tau)["f1"], binary_metrics(y, p, 0.5)["f1"] - 1e-12
        )

    def test_youden_beats_or_equals_j_at_half(self):
        y, p = synthetic(seed=2)
        tau = threshold_youden(y, p)["threshold"]
        j = lambda t: (binary_metrics(y, p, t)["sensitivity"]
                       + binary_metrics(y, p, t)["specificity"] - 1.0)
        self.assertGreaterEqual(j(tau), j(0.5) - 1e-12)

    def test_sensitivity_constraint_is_met(self):
        y, p = synthetic(seed=3)
        for target in (0.80, 0.90, 0.95):
            rec = threshold_sensitivity(y, p, target)
            self.assertTrue(rec["constraint_met"], f"target {target} not met")
            m = binary_metrics(y, p, rec["threshold"])
            self.assertGreaterEqual(m["sensitivity"], target - 1e-12)

    def test_sensitivity_constraint_maximises_specificity(self):
        y, p = synthetic(seed=4)
        target = 0.90
        rec = threshold_sensitivity(y, p, target)
        # no feasible threshold may be strictly better on specificity
        for tau in np.unique(p):
            m = binary_metrics(y, p, float(tau))
            if m["sensitivity"] >= target - 1e-12:
                self.assertGreaterEqual(
                    binary_metrics(y, p, rec["threshold"])["specificity"],
                    m["specificity"] - 1e-12,
                )

    def test_precision_constraint_met_when_achievable(self):
        y, p = synthetic(seed=5)
        rec = threshold_precision(y, p, 0.50)
        self.assertTrue(rec["constraint_met"])
        self.assertGreaterEqual(
            binary_metrics(y, p, rec["threshold"])["precision"], 0.50 - 1e-12
        )

    def test_precision_constraint_reports_failure_honestly(self):
        y, p = synthetic(seed=6, separation=0.2)  # weak model, target unreachable
        rec = threshold_precision(y, p, 0.99)
        self.assertFalse(rec["constraint_met"])
        self.assertIn("reason", rec)
        # fallback = best achievable precision, still a valid threshold
        m = binary_metrics(y, p, rec["threshold"])
        best = max(binary_metrics(y, p, float(t))["precision"] for t in np.unique(p)
                   if np.isfinite(binary_metrics(y, p, float(t))["precision"]))
        self.assertAlmostEqual(m["precision"], best, places=9)

    def test_no_positives_reports_reason(self):
        y = np.zeros(200)
        p = np.random.default_rng(0).random(200)
        for fn, kw in ((threshold_sensitivity, {}), (threshold_precision, {})):
            rec = fn(y, p, **kw)
            self.assertFalse(rec["constraint_met"])
            self.assertIn("reason", rec)


class TestFitThresholds(unittest.TestCase):
    def test_refuses_evaluation_splits(self):
        y, p = synthetic()
        Y = np.c_[y, y]
        P = np.c_[p, p]
        for bad in ("test", "external", "held_out_test"):
            with self.assertRaises(ValueError, msg=f"split {bad} must be refused"):
                fit_thresholds(Y, P, ["a", "b"], split=bad)

    def test_allows_val_and_train(self):
        y, p = synthetic()
        for good in ("val", "train"):
            out = fit_thresholds(np.c_[y, y], np.c_[p, p], ["a", "b"], split=good)
            self.assertEqual(set(out), set(POLICIES))
            self.assertEqual(set(out["fixed"]), {"a", "b"})

    def test_records_are_self_describing(self):
        y, p = synthetic()
        out = fit_thresholds(np.c_[y, y], np.c_[p, p], ["a", "b"], split="val")
        rec = out["f1_optimal"]["a"]
        self.assertEqual(rec["label"], "a")
        self.assertEqual(rec["split"], "val")
        self.assertEqual(rec["fit_n"], len(y))
        for k in ("sensitivity", "specificity", "precision", "f1", "tp", "fp", "fn", "tn"):
            self.assertIn(k, rec["fit_metrics"])

    def test_shape_mismatch_raises(self):
        y, p = synthetic()
        with self.assertRaises(ValueError):
            fit_thresholds(np.c_[y, y], np.c_[p, p], ["only_one"], split="val")


class TestApplyThresholds(unittest.TestCase):
    def test_matches_per_label_comparison(self):
        rng = np.random.default_rng(0)
        p = rng.random((50, 2))
        thr = {"a": 0.3, "b": 0.7}
        pred = apply_thresholds(p, thr, ["a", "b"])
        np.testing.assert_array_equal(pred[:, 0], (p[:, 0] >= 0.3).astype(int))
        np.testing.assert_array_equal(pred[:, 1], (p[:, 1] >= 0.7).astype(int))

    def test_one_dimensional_input(self):
        p = np.array([0.1, 0.9])
        pred = apply_thresholds(p, {"a": 0.5}, ["a"])
        np.testing.assert_array_equal(pred.ravel(), [0, 1])


if __name__ == "__main__":
    unittest.main()
