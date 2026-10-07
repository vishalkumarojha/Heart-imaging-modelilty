"""Tests for Phase 7b — Grad-CAM.

Two layers:
  * pure-math tests of `bbox_localization` on synthetic heat maps (no model);
  * integrity tests of the published `gradcam_summary.json` /
    `bbox_localization.json` (overlays exist, probabilities match the frozen
    table, pointing-game statistics recompute from the stored cases).

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.gradcam import (
    TARGET_LAYER,
    bbox_cohort,
    bbox_localization,
    load_bbox_reference,
)

SUMMARY = C.GRADCAM_DIR / "gradcam_summary.json"
BBOX = C.GRADCAM_DIR / "bbox_localization.json"


def load_summary():
    assert SUMMARY.exists(), "run `python -m src.gradcam`"
    return json.loads(SUMMARY.read_text())


class TestBboxLocalizationMath(unittest.TestCase):
    def test_heat_inside_box_scores_full_mass(self):
        cam = np.zeros((100, 100))
        cam[10:30, 10:30] = 1.0
        out = bbox_localization(cam, (10, 10, 20, 20), (100, 100))
        self.assertTrue(out["pointing_game_hit"])
        self.assertAlmostEqual(out["heat_mass_inside"], 1.0, places=6)
        self.assertAlmostEqual(out["box_area_fraction"], 0.04, places=6)
        self.assertAlmostEqual(out["concentration_ratio"], 1 / 0.04, places=4)

    def test_heat_outside_box_scores_no_mass_and_misses_argmax(self):
        cam = np.zeros((100, 100))
        cam[70:90, 70:90] = 1.0
        out = bbox_localization(cam, (10, 10, 20, 20), (100, 100))
        self.assertFalse(out["pointing_game_hit"])
        self.assertAlmostEqual(out["heat_mass_inside"], 0.0, places=6)
        self.assertAlmostEqual(out["concentration_ratio"], 0.0, places=6)

    def test_box_coordinates_scale_with_the_heatmap_resolution(self):
        # same physical box, heat map delivered at half resolution
        cam = np.zeros((50, 50))
        cam[5:15, 5:15] = 1.0
        out = bbox_localization(cam, (10, 10, 20, 20), (100, 100))
        self.assertAlmostEqual(out["box_area_fraction"], 0.04, places=2)
        self.assertTrue(out["pointing_game_hit"])

    def test_box_clipped_to_image_bounds_never_explodes(self):
        cam = np.ones((50, 50)) / (50 * 50)
        out = bbox_localization(cam, (-10.0, -10.0, 500.0, 500.0), (100, 100))
        self.assertAlmostEqual(out["heat_mass_inside"], 1.0, places=6)
        self.assertLessEqual(out["box_area_fraction"], 1.0)

    def test_uniform_heat_gives_ratio_near_one(self):
        cam = np.full((40, 40), 1.0 / 1600)
        out = bbox_localization(cam, (10, 10, 20, 20), (40, 40))
        self.assertAlmostEqual(out["concentration_ratio"], 1.0, places=4)
        self.assertAlmostEqual(out["heat_mass_inside"],
                               out["box_area_fraction"], places=4)


class TestBboxReference(unittest.TestCase):
    def test_reference_covers_both_target_labels(self):
        boxes = load_bbox_reference()
        self.assertGreater(len(boxes), 0)
        labels = {lab for (_, lab) in boxes}
        self.assertLessEqual(set(C.TARGET_LABELS), labels)
        for (x, y, w, h) in boxes.values():
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)


class TestPublishedSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()
        cls.bbox = json.loads(BBOX.read_text()) if BBOX.exists() else None

    def test_status_and_provenance(self):
        self.assertEqual(self.summary["status"], "COMPLETED")
        self.assertEqual(self.summary["target_layer"], TARGET_LAYER)
        self.assertEqual(self.summary["checkpoint_sha256"],
                         __import__("src.reproducibility", fromlist=["sha256_file"])
                         .sha256_file(C.BASELINE_CHECKPOINT_BEST))

    def test_every_case_has_an_overlay_file(self):
        self.assertGreater(self.summary["n_cases"], 0)
        for case in self.summary["cases"]:
            png = C.PROJECT_ROOT / case["overlay"]
            self.assertTrue(png.exists(), png)
            self.assertGreater(png.stat().st_size, 1000)

    def test_case_ids_match_the_frozen_case_list(self):
        case_list = json.loads((C.GRADCAM_DIR / "case_list.json").read_text())
        self.assertEqual([c["case_id"] for c in self.summary["cases"]],
                         [c["case_id"] for c in case_list["cases"]])

    def test_recorded_probability_matches_the_frozen_table(self):
        table = pd.read_csv(
            C.PREDICTIONS_DIR / "calibrated" / "calibrated_test_predictions.csv")
        table = table.set_index("Image Index")
        for case in self.summary["cases"]:
            row = table.loc[case["image_index"]]
            calibrated = float(row[f"prob_cal_{case['label']}"])
            # the case's `probability` is the calibrated score it was selected
            # with; it must match the table and the frozen-threshold side
            self.assertAlmostEqual(case["probability"], calibrated, places=9,
                                   msg=case["case_id"])
            self.assertEqual(case["probability"] >= case["threshold"],
                             calibrated >= case["threshold"], case["case_id"])
            # the raw score must be on the same side of 0.5 (T>1 cannot flip it)
            raw = float(row[f"prob_{case['label']}"])
            self.assertEqual(raw >= 0.5, calibrated >= 0.5, case["case_id"])

    def test_quantitative_note_declares_qualitative_default(self):
        note = self.summary["quantitative_note"].lower()
        self.assertIn("qualitative", note)
        self.assertIn("box", note)


class TestBboxCohortReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BBOX.exists():
            cls.bbox = json.loads(BBOX.read_text())

    def test_statistics_recompute_from_stored_cases(self):
        if not BBOX.exists():
            self.skipTest("no boxed test images — Grad-CAM stayed qualitative")
        cases = self.bbox["cases"]
        self.assertEqual(self.bbox["n_with_box"], len(cases))
        hits = [c["pointing_game_hit"] for c in cases]
        self.assertEqual(self.bbox["pointing_game_hits"], int(sum(hits)))
        self.assertAlmostEqual(self.bbox["pointing_game_accuracy"],
                               float(np.mean(hits)), places=6)
        ratios = [c["concentration_ratio"] for c in cases]
        self.assertAlmostEqual(self.bbox["mean_concentration_ratio"],
                               float(np.mean(ratios)), places=6)
        self.assertEqual(self.bbox["ratio_above_1"],
                         int(sum(r > 1.0 for r in ratios)))

    def test_cohort_is_the_fixed_boxed_subset_not_a_selection(self):
        if not BBOX.exists():
            self.skipTest("no boxed test images")
        boxes = load_bbox_reference()
        split_df = pd.read_csv(C.DATA_PROCESSED_DIR / "split_index.csv")
        test_ids = set(split_df.loc[split_df["split"] == "test", "Image Index"])
        expected = {(i, l) for (i, l) in boxes
                    if i in test_ids and l in C.TARGET_LABELS}
        got = {(c["image_index"], c["label"]) for c in self.bbox["cases"]}
        self.assertEqual(got, expected)

    def test_runner_without_boxed_images_returns_empty_not_fake(self):
        self.assertEqual(bbox_cohort(None, None, None, None, None, "nope"), [])


if __name__ == "__main__":
    unittest.main()
