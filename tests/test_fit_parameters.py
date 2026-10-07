"""Tests for the frozen parameter artifacts produced by src/fit_parameters.py.

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

from src import config as C
from src.fit_parameters import fit_all


class TestFitGuards(unittest.TestCase):
    def test_refuses_evaluation_splits(self):
        for bad in ("test", "external", "chexpert"):
            with self.assertRaises(ValueError, msg=f"split {bad} must be refused"):
                fit_all(split=bad)


class TestFrozenParameterArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temps = json.loads(C.TEMPERATURE_FILE.read_text())
        cls.raw = json.loads(C.THRESHOLDS_FILE.read_text())
        cls.cal = json.loads(
            (C.THRESHOLD_METRICS_DIR / "thresholds_calibrated_val.json").read_text()
        )
        cls.report = json.loads(
            (C.CALIBRATION_METRICS_DIR / "calibration_report_val.json").read_text()
        )

    def test_temperatures_are_positive_and_per_label(self):
        self.assertEqual(self.temps["type"], "temperature_scaling")
        self.assertEqual(self.temps["labels"], list(C.TARGET_LABELS))
        self.assertEqual(len(self.temps["temperatures"]), len(C.TARGET_LABELS))
        for t in self.temps["temperatures"]:
            self.assertGreater(t, 0.0)

    def test_fitted_on_validation_only(self):
        self.assertEqual(self.temps["metadata"]["split"], "val")
        self.assertEqual(self.report["split"], "val")
        for state in (self.raw, self.cal):
            for policy, per_label in state["policies"].items():
                for rec in per_label.values():
                    self.assertEqual(rec["split"], "val", f"{policy} fitted elsewhere")

    def test_all_policies_present_for_both_label_sets(self):
        for state in (self.raw, self.cal):
            self.assertEqual(set(state["policies"]), set(C.THRESHOLD_POLICIES))
            for per_label in state["policies"].values():
                self.assertEqual(set(per_label), set(C.TARGET_LABELS))

    def test_thresholds_are_valid_probabilities(self):
        for state in (self.raw, self.cal):
            for per_label in state["policies"].values():
                for rec in per_label.values():
                    self.assertGreaterEqual(rec["threshold"], 0.0)
                    self.assertLessEqual(rec["threshold"], 1.0)

    def test_raw_and_calibrated_threshold_sets_differ(self):
        """Two sets exist because temperature scaling moves the scale."""
        for policy in ("f1_optimal", "youden", "sensitivity_constrained",
                       "precision_constrained"):
            for lbl in C.TARGET_LABELS:
                r = self.raw["policies"][policy][lbl]["threshold"]
                c = self.cal["policies"][policy][lbl]["threshold"]
                self.assertNotEqual(r, c, f"{policy}/{lbl}: sets unexpectedly identical")

    def test_validation_report_is_self_consistent(self):
        self.assertEqual(self.report["labels"], list(C.TARGET_LABELS))
        for lbl in C.TARGET_LABELS:
            raw = self.report["raw"][lbl]
            cal = self.report["calibrated"][lbl]
            for payload in (raw, cal):
                self.assertIn("ece", payload)
                self.assertIn("brier", payload)
                self.assertGreaterEqual(payload["ece"], 0.0)
                self.assertLessEqual(payload["ece"], 1.0)
            self.assertAlmostEqual(raw["n"], self.report["n_rows"])
            self.assertAlmostEqual(cal["n"], self.report["n_rows"])


if __name__ == "__main__":
    unittest.main()
