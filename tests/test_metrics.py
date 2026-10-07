"""Tests for src/metrics.py — the metric layer every result flows through.

Run:  python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from src.metrics import (
    auroc,
    auprc,
    binary_metrics,
    multilabel_report,
    report_to_rows,
)


class TestThresholdFreeMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 2, 2000)
        self.p = rng.random(2000)

    def test_auroc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        self.assertAlmostEqual(auroc(self.y, self.p), roc_auc_score(self.y, self.p), places=12)

    def test_auprc_matches_sklearn(self):
        from sklearn.metrics import average_precision_score

        self.assertAlmostEqual(
            auprc(self.y, self.p), average_precision_score(self.y, self.p), places=12
        )

    def test_undefined_metrics_return_nan_not_zero(self):
        p = np.random.default_rng(1).random(50)
        self.assertTrue(np.isnan(auroc(np.zeros(50), p)))      # single class
        self.assertTrue(np.isnan(auprc(np.zeros(50), p)))       # no positives

    def test_rank_invariance_under_monotone_map(self):
        """Temperature scaling is monotone: discrimination must not move."""
        odds = self.p / (1.0 - self.p)
        for t in (0.3, 1.0, 2.5, 10.0):
            p2 = 1.0 / (1.0 + np.exp(-np.log(odds) / t))
            self.assertAlmostEqual(auroc(self.y, self.p), auroc(self.y, p2), places=12)
            self.assertAlmostEqual(auprc(self.y, self.p), auprc(self.y, p2), places=9)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            auroc(self.y, self.p[:-1])

    def test_nan_is_not_silently_zero(self):
        """A NaN must survive to the report instead of becoming 0."""
        y = np.zeros(10)
        p = np.linspace(0, 1, 10)
        rep = multilabel_report(y.reshape(-1, 1), p.reshape(-1, 1), ["a"], 0.5)
        self.assertTrue(np.isnan(rep["per_label"]["a"]["auroc"]))
        self.assertTrue(np.isnan(rep["per_label"]["mean"]["auroc"]))


class TestOperatingPointMetrics(unittest.TestCase):
    def test_confusion_counts_are_consistent(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        m = binary_metrics(y, p, 0.5)
        self.assertEqual(m["tp"] + m["tn"] + m["fp"] + m["fn"], 500)
        self.assertEqual(m["tp"] + m["fn"], m["support_pos"])
        self.assertEqual(m["tn"] + m["fp"], m["support_neg"])

    def test_rates_match_sklearn(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        m = binary_metrics(y, p, 0.5)
        pred = p >= 0.5
        self.assertAlmostEqual(m["accuracy"], accuracy_score(y, pred), places=12)
        self.assertAlmostEqual(m["precision"], precision_score(y, pred), places=12)
        self.assertAlmostEqual(m["recall"], recall_score(y, pred), places=12)
        self.assertAlmostEqual(m["f1"], f1_score(y, pred), places=9)

    def test_threshold_extremes(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        all_neg = binary_metrics(y, p, 1.1)
        self.assertEqual(all_neg["tp"], 0)
        self.assertEqual(all_neg["fn"], 2)
        self.assertTrue(np.isnan(all_neg["precision"]))  # no predicted positives
        all_pos = binary_metrics(y, p, 0.0)
        self.assertEqual(all_pos["fp"], 2)
        self.assertEqual(all_pos["tp"], 2)


class TestMultiLabelReport(unittest.TestCase):
    def test_structure_and_macro_mean(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, (400, 2))
        p = rng.random((400, 2))
        rep = multilabel_report(y, p, ["a", "b"], 0.5)
        self.assertEqual(set(rep["per_label"]), {"a", "b", "mean"})
        per = rep["per_label"]
        self.assertAlmostEqual(
            per["mean"]["auroc"], np.mean([per["a"]["auroc"], per["b"]["auroc"]]), places=12
        )
        self.assertAlmostEqual(
            per["mean"]["auprc"], np.mean([per["a"]["auprc"], per["b"]["auprc"]]), places=12
        )

    def test_report_to_rows_includes_mean(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, (200, 2))
        p = rng.random((200, 2))
        rows = report_to_rows(multilabel_report(y, p, ["a", "b"], 0.5), ["a", "b"])
        self.assertEqual([r["label"] for r in rows], ["a", "b", "mean"])

    def test_column_count_mismatch_raises(self):
        y = np.zeros((10, 2))
        p = np.full((10, 2), 0.5)
        with self.assertRaises(ValueError):
            multilabel_report(y, p, ["only_one"], 0.5)


if __name__ == "__main__":
    unittest.main()
