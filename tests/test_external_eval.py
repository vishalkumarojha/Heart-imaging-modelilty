"""Tests for the BLOCKED external-validation phase (no external data present).

The property under test is an *honesty* property: with the cohort absent, the
phase must produce a BLOCKED status and **no metric file, no prediction file and
no plot** — a missing dataset may never be papered over with a number.

Run:  python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.external_eval import (
    BLOCKED_STATUS,
    ExternalDataNotAvailable,
    apply_uncertainty_policy,
    check_external_data,
    load_external_frame,
    run,
)


class TestAvailabilityDetection(unittest.TestCase):
    def test_cohort_is_reported_missing_in_this_repository(self):
        status = check_external_data()
        self.assertFalse(status["available"])
        self.assertTrue(any("valid.csv" in p for p in status["problems"]))
        self.assertTrue(any("images" in p for p in status["problems"]))

    def test_label_map_and_policy_are_declared(self):
        status = check_external_data()
        self.assertEqual(status["label_map"]["Effusion"], "Pleural Effusion")
        self.assertEqual(status["uncertainty_policy"], "u_zeroes")


class TestBlockedBehaviour(unittest.TestCase):
    def test_loading_raises_instead_of_returning_something(self):
        with self.assertRaises(ExternalDataNotAvailable):
            load_external_frame()

    def test_run_writes_only_a_blocked_status(self):
        payload = run()
        self.assertEqual(payload["status"], BLOCKED_STATUS)
        self.assertTrue(payload["no_metrics_were_computed"])
        self.assertTrue(Path(C.EXTERNAL_STATUS_JSON).exists())
        stored = json.loads(Path(C.EXTERNAL_STATUS_JSON).read_text())
        self.assertEqual(stored["status"], BLOCKED_STATUS)
        # no results artifacts may exist for a phase that never ran
        artifacts = list(C.EXTERNAL_METRICS_DIR.glob("external_eval_*"))
        self.assertEqual(artifacts, [], f"unexpected external results: {artifacts}")
        self.assertEqual(list(C.EXTERNAL_PREDICTIONS_DIR.glob("*.csv")), [])

    def test_status_lists_what_is_implemented_and_what_is_missing(self):
        payload = run()
        self.assertGreaterEqual(len(payload["what_is_implemented"]), 3)
        self.assertTrue(payload["what_is_missing"])
        self.assertIn("python -m src.external_eval", payload["how_to_unblock"])


class TestUncertaintyPolicy(unittest.TestCase):
    def test_u_zeroes(self):
        s = pd.Series([1, 0, -1, np.nan, -1])
        np.testing.assert_array_equal(apply_uncertainty_policy(s, "u_zeroes"),
                                      [1, 0, 0, 0, 0])

    def test_u_ones(self):
        s = pd.Series([1, 0, -1, np.nan])
        np.testing.assert_array_equal(apply_uncertainty_policy(s, "u_ones"),
                                      [1, 0, 1, 0])

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            apply_uncertainty_policy(pd.Series([1, -1]), "u_sometimes")

    def test_non_binary_input_raises(self):
        with self.assertRaises(ValueError):
            apply_uncertainty_policy(pd.Series([0.0, 0.5]), "u_zeroes")


class TestLoaderWithSyntheticCohort(unittest.TestCase):
    """Exercise the loader against a tiny fake CheXpert layout (temp dir)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.images = root / "images"
        (self.images / "study1").mkdir(parents=True)
        (self.images / "study2").mkdir(parents=True)
        from PIL import Image

        for name in ("study1/img1.png", "study2/img2.png"):
            Image.new("L", (32, 32), color=128).save(self.images / name)
        self.csv = root / "valid.csv"
        pd.DataFrame({
            "Path": ["study1/img1.png", "study2/img2.png"],
            "Cardiomegaly": [1.0, -1.0],
            "Pleural Effusion": [np.nan, 1.0],
        }).to_csv(self.csv, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_maps_and_binarises_labels(self):
        frame = load_external_frame(self.csv, self.images, split="valid")
        self.assertEqual(len(frame), 2)
        # Cardiomegaly: 1 -> 1, uncertain -1 -> 0 (u_zeroes)
        # Effusion:     blank -> 0, 1 -> 1
        np.testing.assert_array_equal(frame["true_Cardiomegaly"], [1, 0])
        np.testing.assert_array_equal(frame["true_Effusion"], [0, 1])
        self.assertTrue((frame["split"] == "valid").all())

    def test_missing_images_raise_before_any_evaluation(self):
        (self.images / "study2" / "img2.png").unlink()
        with self.assertRaises(FileNotFoundError):
            load_external_frame(self.csv, self.images)

    def test_missing_label_column_raises_with_context(self):
        bad = Path(self.tmp.name) / "bad.csv"
        pd.DataFrame({"Path": ["study1/img1.png"], "Support Devices": [1.0]}).to_csv(
            bad, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_external_frame(bad, self.images)
        self.assertIn("Cardiomegaly", str(ctx.exception))

    def test_missing_csv_raises_blocked_error(self):
        with self.assertRaises(ExternalDataNotAvailable):
            load_external_frame(Path(self.tmp.name) / "nope.csv", self.images)


if __name__ == "__main__":
    unittest.main()
