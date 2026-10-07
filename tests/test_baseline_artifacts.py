"""Guards on the frozen control and on repository data integrity.

These tests do NOT train or run the model: they check that the artifacts the
project compares against are still the artifacts that were verified in Phase 2,
and that the known data defects are still exactly the ones documented.

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.reproducibility import sha256_file

# The five defective images documented in docs/REPOSITORY_AUDIT.md §2a.
DEFECTIVE_IMAGES = {
    "00029705_000.png": "truncated",   # test split — substituted label
    "00029717_001.png": "truncated",   # train
    "00029718_000.png": "empty",       # train
    "00029719_000.png": "empty",       # train
    "00029720_000.png": "empty",       # train
}


class TestFrozenBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(C.BASELINE_MANIFEST.read_text())

    def test_baseline_checkpoints_exist(self):
        self.assertTrue(C.BASELINE_CHECKPOINT_BEST.exists())
        self.assertTrue(C.BASELINE_CHECKPOINT_LAST.exists())

    def test_best_checkpoint_hash_unchanged(self):
        """The control must never be retrained/overwritten."""
        self.assertEqual(
            sha256_file(C.BASELINE_CHECKPOINT_BEST), self.manifest["sha256"]
        )

    def test_last_checkpoint_hash_unchanged(self):
        self.assertEqual(
            sha256_file(C.BASELINE_CHECKPOINT_LAST),
            self.manifest["last_checkpoint_sha256"],
        )

    def test_root_checkpoints_still_present_for_fusion_and_demo(self):
        """R4: fusion/demo hard-code these paths — they must not disappear."""
        self.assertTrue((C.OUTPUTS_DIR / "checkpoints" / "densenet121_best.pt").exists())
        self.assertTrue((C.OUTPUTS_DIR / "checkpoints" / "densenet121_last.pt").exists())

    def test_manifest_points_at_this_documentation(self):
        self.assertIn("BASELINE.md", self.manifest["source_of_truth"])


class TestBaselineMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agg = json.loads(C.BASELINE_METRICS_JSON.read_text())
        cls.test = json.loads(
            (C.BASELINE_METRICS_DIR / "baseline_metrics_test.json").read_text()
        )
        cls.val = json.loads(
            (C.BASELINE_METRICS_DIR / "baseline_metrics_val.json").read_text()
        )

    def test_both_splits_are_recorded(self):
        self.assertEqual(sorted(self.agg["splits_completed"]), ["test", "val"])

    def test_macro_is_the_mean_of_per_label_values(self):
        for payload in (self.test, self.val):
            per = payload["metrics"]["per_label"]
            labels = payload["labels"]
            self.assertAlmostEqual(
                per["mean"]["auroc"],
                float(np.mean([per[l]["auroc"] for l in labels])),
                places=12,
            )
            self.assertAlmostEqual(
                per["mean"]["auprc"],
                float(np.mean([per[l]["auprc"] for l in labels])),
                places=12,
            )

    def test_confusion_counts_add_up(self):
        for payload in (self.test, self.val):
            n = payload["n_images"]
            for label in payload["labels"]:
                m = payload["metrics"]["per_label"][label]
                self.assertEqual(m["tp"] + m["tn"] + m["fp"] + m["fn"], n)

    def test_threshold_is_the_fixed_baseline_value(self):
        for payload in (self.test, self.val):
            self.assertEqual(payload["threshold"], 0.5)

    def test_expected_test_supports(self):
        """Documented test-set supports (docs/BASELINE.md §4)."""
        per = self.test["metrics"]["per_label"]
        self.assertEqual(per["Cardiomegaly"]["support_pos"], 415)
        self.assertEqual(per["Effusion"]["support_pos"], 1997)  # 1996 true + 1 substituted
        self.assertEqual(self.test["n_images"], 15884)
        self.assertEqual(self.val["n_images"], 16451)

    def test_no_patient_leakage_recorded(self):
        for payload in (self.test, self.val):
            ov = payload["split_statistics"]["patient_overlap"]
            self.assertEqual(ov, {"train_val": 0, "train_test": 0, "val_test": 0})


class TestSplitAndDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = pd.read_csv(C.SPLIT_CACHE_CSV)

    def test_documented_row_counts(self):
        counts = self.split["split"].value_counts().to_dict()
        self.assertEqual(counts["train"], 76977)
        self.assertEqual(counts["val"], 16451)
        self.assertEqual(counts["test"], 15884)
        self.assertEqual(len(self.split), 109312)
        self.assertEqual(self.split["Patient ID"].nunique(), 29720)

    def test_no_patient_appears_in_two_splits(self):
        per_split = self.split.groupby("Patient ID")["split"].nunique()
        self.assertEqual(int((per_split > 1).sum()), 0)

    def test_known_defective_files_are_still_the_documented_set(self):
        """A full structural re-scan of the 109,312 PNGs, in memory-light form.

        Any NEW defect would fail here — which is the point of the guard.
        """
        found = {}
        for d in sorted(C.DATA_RAW_DIR.glob("images_*")):
            for p in d.rglob("*.png"):
                if p.stat().st_size < 8:
                    found[p.name] = "empty"
                    continue
                with p.open("rb") as f:
                    f.seek(-8, 2)
                    tail = f.read(8)
                # a complete PNG ends with the IEND chunk type + its CRC
                if tail[:4] != b"IEND":
                    found[p.name] = "truncated"
        self.assertEqual(found, DEFECTIVE_IMAGES)

    def test_test_split_contains_exactly_one_defective_image(self):
        names = set(self.split.loc[self.split["split"] == "test", "Image Index"])
        self.assertEqual(names & set(DEFECTIVE_IMAGES), {"00029705_000.png"})


class TestCachedPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = {}
        for split in ("test", "val"):
            cls.frames[split] = pd.read_csv(
                C.BASELINE_PREDICTIONS_DIR / f"baseline_{split}_predictions.csv"
            )

    def test_row_counts_match_split(self):
        counts = pd.read_csv(C.SPLIT_CACHE_CSV)["split"].value_counts()
        for split, df in self.frames.items():
            self.assertEqual(len(df), int(counts[split]))

    def test_columns_are_the_contract(self):
        for df in self.frames.values():
            for col in ("Image Index", "Patient ID", "split",
                        "true_Cardiomegaly", "logit_Cardiomegaly", "prob_Cardiomegaly",
                        "true_Effusion", "logit_Effusion", "prob_Effusion"):
                self.assertIn(col, df.columns)

    def test_probabilities_are_sigmoids_of_stored_logits(self):
        for df in self.frames.values():
            for lbl in ("Cardiomegaly", "Effusion"):
                logits = df[f"logit_{lbl}"].to_numpy()
                probs = df[f"prob_{lbl}"].to_numpy()
                expected = 1.0 / (1.0 + np.exp(-logits))
                self.assertLess(np.max(np.abs(probs - expected)), 1e-6)

    def test_raw_cache_is_tied_to_the_checkpoint_hash(self):
        meta = json.loads(
            (C.RAW_PREDICTIONS_DIR /
             "test__densenet121_best__35965f610c8b.meta.json").read_text()
        )
        self.assertEqual(
            meta["checkpoint_sha256"], sha256_file(C.BASELINE_CHECKPOINT_BEST)
        )

    def test_labels_are_binary(self):
        for df in self.frames.values():
            for lbl in ("Cardiomegaly", "Effusion"):
                vals = set(np.unique(df[f"true_{lbl}"].to_numpy()))
                self.assertTrue(vals <= {0, 1}, f"{lbl} has values {vals}")


if __name__ == "__main__":
    unittest.main()
