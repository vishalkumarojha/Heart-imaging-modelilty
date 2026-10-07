import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src import config as C
from src.ui_data import LABELS, load_research_data, temperature_probabilities


class ResearchDemoTests(unittest.TestCase):
  def test_research_dashboard_imports_and_builds(self):
    import research_demo_app
    self.assertIsNotNone(research_demo_app.build_app())

  def test_authoritative_artifacts_load_and_labels_are_ordered(self):
    data = load_research_data()
    self.assertEqual(data["temperatures"]["labels"], list(LABELS))
    self.assertEqual(data["calibration"]["labels"], list(LABELS))
    self.assertTrue(data["manifest"]["frozen_baseline"])

  def test_missing_artifact_is_reported(self):
    import src.ui_data as ui_data
    with self.assertRaisesRegex(FileNotFoundError, "Research artifact unavailable"):
        ui_data._json(Path("/tmp/absent-research-artifact.json"))

  def test_calibration_transform_matches_frozen_formula(self):
    logits = [0.0, 1.0]
    temps = [1.0, 2.0]
    got = temperature_probabilities(logits, temps)
    np.testing.assert_allclose(got, [0.5, 1 / (1 + np.exp(-0.5))])

  def test_preprocessing_matches_research_evaluation_shape(self):
    from src.dataset import build_transforms
    image = np.zeros((320, 480, 3), dtype=np.uint8)
    x = build_transforms(train=False, image_size=224)(image=image)["image"]
    self.assertEqual(tuple(x.shape), (3, 224, 224))
    self.assertEqual(str(x.dtype), "torch.float32")

  def test_abcd_definitions_and_thresholds_are_frozen_on_validation(self):
    data = load_research_data()
    a, b = data["thresholds_raw"], data["thresholds_cal"]
    self.assertEqual(a["fit_split"]["fixed"], a["fit_split"]["f1_optimal"])
    self.assertEqual(a["fit_split"]["fixed"], "val")
    self.assertEqual(b["fit_split"]["fixed"], b["fit_split"]["f1_optimal"])
    self.assertEqual(b["fit_split"]["fixed"], "val")
    self.assertEqual(a["policies"]["fixed"]["Cardiomegaly"]["threshold"], 0.5)
    self.assertEqual(b["policies"]["fixed"]["Effusion"]["threshold"], 0.5)
    self.assertNotEqual(a["policies"]["f1_optimal"]["Cardiomegaly"]["threshold"], 0.5)

  def test_ui_loading_does_not_mutate_research_artifacts(self):
    paths = [C.OUTPUTS_DIR / "final_research_manifest.json",
             C.OUTPUTS_DIR / "metrics/calibration/temperature_scalers.json",
             C.OUTPUTS_DIR / "final_results/arm_differences.csv"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    load_research_data()
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    self.assertEqual(before, after)

  def test_external_validation_is_explicitly_pending(self):
    status = load_research_data()["external"]
    self.assertTrue(status["no_metrics_were_computed"])
    self.assertIn("PENDING", status["project_status"]["external_validation"])
