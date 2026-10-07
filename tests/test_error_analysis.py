"""Tests for Phase 7 — error analysis (Phase 7a).

These tests run against the *published* artifacts on disk: they recompute the
headline numbers from the stored prediction CSV and check internal consistency
(strata sum to n, shares in [0,1], case list deterministic and strata-complete).

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.error_analysis import (
    CASES_PER_STRATUM,
    STRATA,
    confidence_error_bins,
    frozen_thresholds,
    load_analysis_table,
    run,
    stratum_label,
)

SPLIT = "test"
REPORT = C.ERROR_ANALYSIS_DIR / f"error_analysis_{SPLIT}.json"


def load_report():
    assert REPORT.exists(), "run `python -m src.error_analysis --split test`"
    return json.loads(REPORT.read_text())


class TestStratumLabelling(unittest.TestCase):
    def test_confusion_naming(self):
        self.assertEqual(stratum_label(1, 1), "TP")
        self.assertEqual(stratum_label(1, 0), "FN")
        self.assertEqual(stratum_label(0, 1), "FP")
        self.assertEqual(stratum_label(0, 0), "TN")


class TestPublishedReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load_report()
        cls.df = load_analysis_table(SPLIT, "calibrated")

    def test_report_covers_the_whole_cohort(self):
        self.assertEqual(self.report["n_images"], len(self.df))
        self.assertEqual(self.report["n_images"], 15884)
        self.assertEqual(self.report["labels"], list(C.TARGET_LABELS))

    def test_strata_sum_to_n_for_every_policy_and_label(self):
        for policy, per_label in self.report["strata"].items():
            for label, entry in per_label.items():
                total = sum(entry["strata"][s]["count"] for s in STRATA)
                self.assertEqual(total, entry["n"], f"{policy}/{label}")
                positives = (entry["strata"]["TP"]["count"] +
                             entry["strata"]["FN"]["count"])
                self.assertEqual(positives, entry["positives"], f"{policy}/{label}")

    def test_strata_counts_recompute_from_predictions(self):
        thr = frozen_thresholds("calibrated")
        for policy in ("fixed", "f1_optimal"):
            for label in C.TARGET_LABELS:
                tau = thr[policy][label]
                y = self.df[f"true_{label}"].to_numpy()
                p = self.df[f"prob_{label}"].to_numpy()
                pred = (p >= tau).astype(int)
                counts = {
                    "TP": int(((pred == 1) & (y == 1)).sum()),
                    "FP": int(((pred == 1) & (y == 0)).sum()),
                    "FN": int(((pred == 0) & (y == 1)).sum()),
                    "TN": int(((pred == 0) & (y == 0)).sum()),
                }
                published = self.report["strata"][policy][label]["strata"]
                for s in STRATA:
                    self.assertEqual(published[s]["count"], counts[s],
                                     f"{policy}/{label}/{s}")

    def test_shares_and_error_rate_are_probabilities(self):
        for policy, per_label in self.report["strata"].items():
            for label, entry in per_label.items():
                for s in STRATA:
                    share = entry["strata"][s]["share"]
                    self.assertGreaterEqual(share, 0.0)
                    self.assertLessEqual(share, 1.0)
                self.assertGreaterEqual(entry["error_rate"], 0.0)
                self.assertLessEqual(entry["error_rate"], 1.0)
                hc = entry["high_confidence_errors"]
                self.assertGreaterEqual(hc["share_of_errors"], 0.0)
                self.assertLessEqual(hc["share_of_errors"], 1.0)

    def test_high_confidence_errors_are_a_subset_of_errors(self):
        for policy, per_label in self.report["strata"].items():
            for label, entry in per_label.items():
                n_err = round(entry["error_rate"] * entry["n"])
                self.assertLessEqual(entry["high_confidence_errors"]["count"], n_err)

    def test_confidence_error_bins_partition_the_cohort(self):
        for label, rows in self.report["confidence_error_bins"].items():
            self.assertEqual(sum(r["count"] for r in rows),
                             self.report["n_images"], label)
            for r in rows:
                if r["count"]:
                    self.assertGreaterEqual(r["error_rate"], 0.0)
                    self.assertLessEqual(r["error_rate"], 1.0)

    def test_low_confidence_bins_are_less_wrong_than_high_bins(self):
        # the model is over-confident (see Phase 5), but errors must still
        # concentrate in the high-probability bins
        for label, rows in self.report["confidence_error_bins"].items():
            have = [r for r in rows if r["count"]]
            low = np.mean([r["error_rate"] for r in have if r["hi"] <= 0.5])
            high = np.mean([r["error_rate"] for r in have if r["lo"] >= 0.5])
            self.assertLess(low, high, label)

    def test_cross_label_overlap_and_patient_concentration_are_consistent(self):
        ov = self.report["cross_label_overlap"]
        pc = self.report["patient_error_concentration"]
        self.assertGreaterEqual(ov["n_both_wrong"], 0)
        self.assertLessEqual(ov["n_both_wrong"], ov["n_either_wrong"])
        self.assertGreaterEqual(ov["jaccard_of_error_sets"], 0.0)
        self.assertLessEqual(ov["jaccard_of_error_sets"], 1.0)
        self.assertGreaterEqual(pc["max_errors_by_one_patient"], 1)
        self.assertLessEqual(pc["max_errors_by_one_patient"], pc["n_error_images"])


class TestGradCamCaseList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load_report()
        cls.cases = cls.report["gradcam_case_list"]

    def test_case_file_exists_and_matches_report(self):
        path = C.GRADCAM_DIR / "case_list.json"
        self.assertTrue(path.exists())
        payload = json.loads(path.read_text())
        self.assertEqual(payload["n_cases"], len(self.cases))
        self.assertEqual([c["case_id"] for c in payload["cases"]],
                         [c["case_id"] for c in self.cases])

    def test_every_stratum_of_every_label_is_represented(self):
        seen = {(c["label"], c["stratum"]) for c in self.cases}
        expected = {(l, s) for l in C.TARGET_LABELS for s in STRATA}
        self.assertEqual(seen, expected)

    def test_selection_is_deterministic_and_bounded(self):
        for label in C.TARGET_LABELS:
            for s in STRATA:
                got = [c for c in self.cases if c["label"] == label and c["stratum"] == s]
                self.assertLessEqual(len(got), CASES_PER_STRATUM)
        # re-running the selector must give byte-identical ids
        df = load_analysis_table(SPLIT, "calibrated")
        from src.error_analysis import select_gradcam_cases

        again = select_gradcam_cases(df, frozen_thresholds("calibrated"))
        self.assertEqual([c["case_id"] for c in again],
                         [c["case_id"] for c in self.cases])

    def test_case_probabilities_are_on_the_correct_side_of_the_threshold(self):
        for c in self.cases:
            if c["stratum"] in ("TP", "FP"):
                self.assertGreaterEqual(c["probability"], c["threshold"], c["case_id"])
            else:
                self.assertLess(c["probability"], c["threshold"], c["case_id"])
            if c["stratum"] in ("TP", "FN"):
                self.assertTrue(c["true_positive"], c["case_id"])
            else:
                self.assertFalse(c["true_positive"], c["case_id"])

    def test_all_case_images_resolve_to_existing_files(self):
        for c in self.cases:
            self.assertTrue(c["path"], c["case_id"])
            self.assertTrue(Path(c["path"]).exists(), c["path"])


class TestHelpers(unittest.TestCase):
    def test_variant_guard(self):
        with self.assertRaises(ValueError):
            load_analysis_table("test", "fancy")

    def test_thresholds_are_recorded_as_val_fitted(self):
        raw = json.loads(C.THRESHOLDS_FILE.read_text())
        self.assertEqual(set(raw["fit_split"].values()), {"val"})
        self.assertEqual(set(raw["policies"].keys()),
                         {"fixed", "f1_optimal", "youden",
                          "sensitivity_constrained", "precision_constrained"})

    def test_confidence_error_bins_handles_empty_bins(self):
        y = np.array([0, 1])
        p = np.array([0.01, 0.99])
        rows = confidence_error_bins(y, p, n_bins=10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(r["count"] for r in rows), 2)
        for r in rows:
            if not r["count"]:
                self.assertIsNone(r["error_rate"])

    def test_run_is_idempotent(self):
        payload = run(SPLIT, "calibrated")
        self.assertEqual(payload["status"], "COMPLETED")
        again = load_report()
        self.assertEqual([c["case_id"] for c in again["gradcam_case_list"]],
                         [c["case_id"] for c in payload["gradcam_case_list"]])


if __name__ == "__main__":
    unittest.main()
