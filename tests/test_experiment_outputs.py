"""Tests that the published experiment numbers can be re-derived from artifacts.

These are verification tests, not regression tests: they recompute reported
quantities from the stored prediction tables and from the frozen parameters, so
a results file that drifts from its own inputs fails loudly.

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.calibration import TemperatureScaler, brier_score, expected_calibration_error
from src.metrics import binary_metrics

EXP = C.EXPERIMENT_METRICS_DIR
LABELS = list(C.TARGET_LABELS)


def _load(name):
    return json.loads((EXP / name).read_text())


class TestExp1Calibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load("exp1_calibration.json")

    def test_completed_and_fitted_on_validation(self):
        self.assertEqual(self.exp["status"], "COMPLETED")
        self.assertEqual(self.exp["parameter_fit_split"], "val")
        self.assertEqual(self.exp["calibration_fitted_on"], "val")

    def test_ranking_is_invariant_to_temperature(self):
        self.assertTrue(self.exp["auroc_auprc_invariant_to_temperature"])

    def test_temperatures_match_the_frozen_file(self):
        frozen = json.loads(C.TEMPERATURE_FILE.read_text())
        self.assertEqual(
            {k: v for k, v in self.exp["temperatures"].items()},
            {l: t for l, t in zip(frozen["labels"], frozen["temperatures"])},
        )

    def test_reported_ece_recomputes_from_stored_predictions(self):
        scaler = TemperatureScaler.load(C.TEMPERATURE_FILE)
        for split in ("val", "test"):
            df = pd.read_csv(C.PREDICTIONS_DIR / "calibrated" /
                             f"calibrated_{split}_predictions.csv")
            for lbl in LABELS:
                y = df[f"true_{lbl}"].to_numpy()
                raw = df[f"prob_{lbl}"].to_numpy()
                cal = df[f"prob_cal_{lbl}"].to_numpy()
                rep = self.exp["splits"][split][lbl]
                self.assertAlmostEqual(
                    expected_calibration_error(y, raw, C.CALIBRATION_BINS,
                                               C.CALIBRATION_STRATEGY),
                    rep["raw"]["ece"], places=9)
                self.assertAlmostEqual(
                    expected_calibration_error(y, cal, C.CALIBRATION_BINS,
                                               C.CALIBRATION_STRATEGY),
                    rep["calibrated"]["ece"], places=9)
                self.assertAlmostEqual(brier_score(y, raw), rep["raw"]["brier"], places=9)
                # the stored calibrated probabilities are sigmoid(logit / T)
                t = float(scaler.temperatures_[LABELS.index(lbl)])
                logits = df[f"logit_{lbl}"].to_numpy()
                np.testing.assert_allclose(
                    cal, 1.0 / (1.0 + np.exp(-logits / t)), atol=1e-9)

    def test_temperature_scaling_improves_the_proper_scoring_rules(self):
        """The claim the project makes about calibration rests on this."""
        for split in ("val", "test"):
            for lbl in LABELS:
                rep = self.exp["splits"][split][lbl]
                self.assertLess(rep["calibrated"]["nll"], rep["raw"]["nll"])
                self.assertLess(rep["calibrated"]["brier"], rep["raw"]["brier"])


class TestExp2Exp3Thresholds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = _load("exp2_thresholds_raw.json")
        cls.cal = _load("exp3_thresholds_calibrated.json")
        cls.raw_thr = json.loads(C.THRESHOLDS_FILE.read_text())
        cls.cal_thr = json.loads(
            (C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json").read_text())

    def test_parameters_came_from_validation(self):
        for state in (self.raw_thr, self.cal_thr):
            for per_label in state["policies"].values():
                for rec in per_label.values():
                    self.assertEqual(rec["split"], "val")

    def test_test_metrics_recompute_from_thresholded_predictions(self):
        from src.thresholds import POLICIES

        df = pd.read_csv(C.PREDICTIONS_DIR / "thresholded" /
                         "raw__thresholded_test_predictions.csv")
        for lbl in LABELS:
            y = df[f"true_{lbl}"].to_numpy()
            entry = self.raw["evaluation"]["test"][lbl]
            for policy in POLICIES:
                thr = self.raw_thr["policies"][policy][lbl]["threshold"]
                recomputed = binary_metrics(y, df[f"prob_{lbl}"].to_numpy(), thr)
                reported = entry[policy]
                self.assertEqual(reported["tp"], recomputed["tp"])
                self.assertEqual(reported["fp"], recomputed["fp"])
                self.assertEqual(recomputed["f1"], reported["f1"])
                # stored hard predictions agree with the frozen threshold
                pred = df[f"pred_{policy}_{lbl}"].to_numpy()
                np.testing.assert_array_equal(
                    pred, (df[f"prob_{lbl}"].to_numpy() >= thr).astype(int))

    def test_fixed_policy_equals_the_baseline_at_half(self):
        baseline = json.loads((C.BASELINE_METRICS_DIR /
                               "baseline_metrics_test.json").read_text())
        for exp in (self.raw, self.cal):
            for lbl in LABELS:
                fixed = exp["evaluation"]["test"][lbl]["fixed"]
                base = baseline["metrics"]["per_label"][lbl]
                for metric in ("precision", "recall", "specificity", "f1", "accuracy"):
                    self.assertAlmostEqual(fixed[metric], base[metric], places=12,
                                           msg=f"{lbl}/{metric}")

    def test_half_threshold_decisions_are_identical_raw_vs_calibrated(self):
        """sigmoid(logit/T) ≥ 0.5 ⟺ sigmoid(logit) ≥ 0.5 for any T > 0."""
        for lbl in LABELS:
            r = self.raw["evaluation"]["test"][lbl]["fixed"]
            c = self.cal["evaluation"]["test"][lbl]["fixed"]
            for metric in ("tp", "fp", "fn", "tn", "precision", "recall", "f1"):
                self.assertEqual(r[metric], c[metric], msg=f"{lbl}/{metric}")

    def test_refit_policies_reproduce_the_same_operating_points(self):
        """Calibration is monotone: refitting a policy in either space picks the
        same confusion matrix (only the numeric threshold differs)."""
        from src.thresholds import POLICIES

        for lbl in LABELS:
            for policy in POLICIES:
                if policy == "fixed":
                    continue
                r = self.raw["evaluation"]["test"][lbl][policy]
                c = self.cal["evaluation"]["test"][lbl][policy]
                for metric in ("tp", "fp", "fn", "tn", "f1", "precision", "recall"):
                    self.assertEqual(r[metric], c[metric], msg=f"{lbl}/{policy}/{metric}")
                self.assertNotEqual(r["threshold"], c["threshold"],
                                    f"{lbl}/{policy}: thresholds should differ")

    def test_f1_optimal_improves_f1_over_the_baseline(self):
        for lbl in LABELS:
            m = self.raw["evaluation"]["test"][lbl]["f1_optimal"]
            self.assertGreater(m["delta_f1_vs_fixed"], 0.0)

    def test_constrained_policies_meet_their_val_fitted_constraints_on_val(self):
        for lbl in LABELS:
            sens = self.raw["evaluation"]["val"][lbl]["sensitivity_constrained"]
            prec = self.raw["evaluation"]["val"][lbl]["precision_constrained"]
            self.assertGreaterEqual(sens["recall"], C.SENSITIVITY_TARGET - 1e-9)
            self.assertGreaterEqual(prec["precision"], C.PRECISION_TARGET - 1e-9)


class TestExp4Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = _load("exp4_robustness.json")

    def test_all_three_studies_present(self):
        self.assertIn("ece_binning_sensitivity", self.exp)
        self.assertIn("constraint_target_sweeps", self.exp)
        self.assertIn("bootstrap_threshold_stability", self.exp)

    def test_every_sensitivity_constraint_was_met_on_the_fit_split(self):
        for rows in self.exp["constraint_target_sweeps"].values():
            for r in rows:
                self.assertTrue(r["constraint_met"], f"{r}")

    def test_higher_sensitivity_target_means_lower_threshold(self):
        rows = self.exp["constraint_target_sweeps"]["sensitivity_constrained"]
        for lbl in LABELS:
            taus = [r["threshold"] for r in rows if r["label"] == lbl]
            self.assertEqual(taus, sorted(taus, reverse=True),
                             f"{lbl}: thresholds must fall as the target rises")

    def test_bootstrap_summary_is_coherent(self):
        res = self.exp["bootstrap_threshold_stability"]["results"]
        self.assertEqual(self.exp["bootstrap_threshold_stability"]["n_bootstrap"], 10)
        for policy, per in res.items():
            for lbl, v in per.items():
                self.assertLessEqual(v["min"], v["mean"] + 1e-12)
                self.assertGreaterEqual(v["max"], v["mean"] - 1e-12)
                self.assertGreaterEqual(v["std"], 0.0)
                self.assertGreaterEqual(v["frozen_threshold"], v["min"] - 1e-12)
                self.assertLessEqual(v["frozen_threshold"], v["max"] + 1e-12)


class TestSummaryTable(unittest.TestCase):
    def test_summary_csv_exists_and_has_expected_experiments(self):
        df = pd.read_csv(EXP / "experiments_summary.csv")
        self.assertGreater(len(df), 100)
        self.assertLessEqual({"exp1_calibration", "exp4_robustness"},
                             set(df["experiment"].unique()))
        self.assertTrue(df["value"].notna().all())


if __name__ == "__main__":
    unittest.main()
