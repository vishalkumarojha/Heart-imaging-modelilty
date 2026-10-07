"""Consistency tests for the frozen research claim (final phase).

These are *cross-artifact* checks (the brief's "consistency re-run"): every box
that the paper needs to be internally consistent is asserted against the raw
artifacts, so a future edit that drifts figures from data fails loudly.

Run:  python -m unittest discover -s tests -v
"""
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.reproducibility import sha256_file


def load(rel: str) -> dict:
    return json.loads((C.PROJECT_ROOT / rel).read_text())


class TestHeadlineNumbersAreTraceable(unittest.TestCase):
    """README/STATUS headline numbers must equal the artifact values."""

    def test_baseline_auroc(self):
        per = load("outputs/metrics/baseline/baseline_metrics_test.json")["metrics"]["per_label"]
        self.assertAlmostEqual(per["Cardiomegaly"]["auroc"], 0.8972, places=4)
        self.assertAlmostEqual(per["Effusion"]["auroc"], 0.8587, places=4)

    def test_baseline_auprc_precision_f1(self):
        per = load("outputs/metrics/baseline/baseline_metrics_test.json")["metrics"]["per_label"]
        self.assertAlmostEqual(per["Cardiomegaly"]["auprc"], 0.3043, places=4)
        self.assertAlmostEqual(per["Effusion"]["auprc"], 0.4700, places=4)
        self.assertAlmostEqual(per["Cardiomegaly"]["precision"], 0.1787, places=4)
        self.assertAlmostEqual(per["Cardiomegaly"]["f1"], 0.2836, places=4)
        self.assertAlmostEqual(per["Effusion"]["f1"], 0.4676, places=4)

    def test_combined_arm_f1(self):
        e3 = load("outputs/metrics/experiments/exp3_thresholds_calibrated.json")
        c = e3["evaluation"]["test"]["Cardiomegaly"]["f1_optimal"]
        e = e3["evaluation"]["test"]["Effusion"]["f1_optimal"]
        self.assertAlmostEqual(c["f1"], 0.3536, places=4)
        self.assertAlmostEqual(e["f1"], 0.4866, places=4)

    def test_error_analysis_high_confidence_share(self):
        ea = load("outputs/metrics/error_analysis/error_analysis_test.json")
        hc = ea["strata"]["f1_optimal"]["Cardiomegaly"]["high_confidence_errors"]
        self.assertAlmostEqual(hc["share_of_errors"], 0.487, places=3)
        self.assertEqual(hc["count"], 274)

    def test_explainability_headline(self):
        b = load("outputs/gradcam/bbox_localization.json")
        self.assertEqual(b["n_with_box"], 43)
        self.assertEqual(b["pointing_game_hits"], 21)
        self.assertAlmostEqual(b["mean_concentration_ratio"], 2.60, places=2)


class TestExperimentIdsAreUnique(unittest.TestCase):
    def test_master_rows_unique_per_arm_policy_class(self):
        df = pd.read_csv(C.MASTER_RESULTS_CSV)
        dup = df.duplicated(subset=["experiment_id", "threshold_policy", "class"])
        self.assertEqual(int(dup.sum()), 0)
        malformed = df["experiment_id"].astype(str).str.startswith("_").any()
        self.assertFalse(malformed, "internal alias rows leaked into master_results")


class TestSelectionHonesty(unittest.TestCase):
    """Thresholds and temperatures are fit on VALIDATION only — never test."""

    def test_temperature_fit_on_val(self):
        meta = load("outputs/metrics/calibration/temperature_scalers.json")["metadata"]
        self.assertEqual(meta["split"], "val")

    def test_thresholds_fit_on_val(self):
        for fname in ("thresholds_val.json", "thresholds_calibrated_val.json"):
            raw = load(f"outputs/metrics/thresholds/{fname}")
            self.assertTrue(len(raw["fit_split"]) >= 1)
            self.assertEqual(set(raw["fit_split"].values()), {"val"})
            for policy in raw["policies"].values():
                for entry in policy.values():
                    self.assertEqual(entry["split"], "val")

    def test_params_fit_file_declares_val(self):
        for fname in ("exp2_thresholds_raw.json", "exp3_thresholds_calibrated.json"):
            raw = load(f"outputs/metrics/experiments/{fname}")
            self.assertEqual(raw["parameter_fit_split"], "val")

    def test_runtime_guard_blocks_fitting_on_test(self):
        # the parameter-fitting layer must refuse the test (and external) splits
        from src.fit_parameters import fit_all

        with self.assertRaises(ValueError):
            fit_all(split="test")
        with self.assertRaises(ValueError):
            fit_all(split="external")


class TestCheckpointIsFrozen(unittest.TestCase):
    def test_baseline_checkpoint_hash_unchanged(self):
        manifest = load("outputs/checkpoints/baseline/baseline_manifest.json")
        fp = C.BASELINE_CHECKPOINT_BEST
        self.assertTrue(fp.exists())
        self.assertEqual(sha256_file(fp), manifest["sha256"])

    def test_snapshot_reports_same_checkpoint(self):
        snap = load("outputs/research_snapshot.json")
        manifest = load("outputs/checkpoints/baseline/baseline_manifest.json")
        self.assertEqual(snap["checkpoints"]["best"]["sha256"], manifest["sha256"])

    def test_every_recorded_checkpoint_reference_exists(self):
        manifest = load("outputs/checkpoints/baseline/baseline_manifest.json")
        recs = [
            load("outputs/metrics/baseline/baseline_metrics_test.json"),
            load("outputs/metrics/experiments/exp1_calibration.json"),
        ]
        for rec in recs:
            if "checkpoint_sha256" in rec:
                self.assertEqual(rec["checkpoint_sha256"], manifest["sha256"],
                                 msg=f"drifted ckpt hash in {rec.get('experiment')}")
                ckpt = Path(C.PROJECT_ROOT / rec.get("checkpoint", ""))
                if ckpt.exists():
                    self.assertEqual(sha256_file(ckpt), manifest["sha256"])


class TestPercentagesMatchCounts(unittest.TestCase):
    def test_error_analysis_percentages(self):
        df = pd.read_csv(C.ERROR_ANALYSIS_CSV)
        n = 15884
        strata = df[df.confidence_bin == "all"]
        for _, r in strata.iterrows():
            self.assertAlmostEqual(r["percentage"], 100.0 * r["count"] / n, places=2)

    def test_strata_counts_sum_to_cohort(self):
        ea = load("outputs/metrics/error_analysis/error_analysis_test.json")
        for lbl in C.TARGET_LABELS:
            s = ea["strata"]["f1_optimal"][lbl]["strata"]
            total = sum(s[k]["count"] for k in ("TP", "FP", "FN", "TN"))
            self.assertEqual(total, ea["n_images"])
            self.assertEqual(ea["n_images"], 15884)

    def test_high_confidence_count_is_consistent_with_share(self):
        ea = load("outputs/metrics/error_analysis/error_analysis_test.json")
        s = ea["strata"]["f1_optimal"]["Cardiomegaly"]
        n_errors = s["strata"]["FP"]["count"] + s["strata"]["FN"]["count"]
        hc = s["high_confidence_errors"]
        self.assertAlmostEqual(hc["count"] / n_errors, hc["share_of_errors"], places=6)


class TestExternalHonesty(unittest.TestCase):
    def test_no_external_measurement_while_blocked(self):
        status = load("outputs/metrics/external/status.json")
        self.assertIn("BLOCKED", status["status"])
        self.assertTrue(status["no_metrics_were_computed"])
        self.assertEqual(list(C.EXTERNAL_METRICS_DIR.glob("external_eval_*")), [])
        self.assertEqual(list(C.EXTERNAL_PREDICTIONS_DIR.glob("*.csv")), [])
        self.assertEqual(list((C.PLOTS_DIR / "external").glob("*.png")), [])

    def test_project_status_is_precise(self):
        status = load("outputs/metrics/external/status.json")
        ps = status["project_status"]
        self.assertEqual(ps["core_research"], "COMPLETE")
        self.assertEqual(ps["external_validation"], "PENDING")


class TestVerificationReportPasses(unittest.TestCase):
    def test_verification_report_has_no_diffs(self):
        rep = load("outputs/final_results/verification_report.json")
        self.assertEqual(rep["status"], "PASS", msg=rep)
        self.assertEqual(rep["checks_differ"], 0)


class TestECEImplementationIsStandard(unittest.TestCase):
    """Independent recomputation of ECE — written as a manual loop, NOT via the
    library function — to verify the published definition is the standard
    Naeini/Guo formulation: sum_b (n_b/N) |acc_b - conf_b| over equal-width
    bins, empty bins contributing 0.

    This is the Step-4 scientific-correctness check for the calibration
    section of the paper: the numbers the paper reports are re-derived from
    raw data by an independent code path.
    """

    @classmethod
    def setUpClass(cls):
        matches = sorted(C.RAW_PREDICTIONS_DIR.glob("test__*.csv"))
        assert matches, "no frozen test predictions"
        cls.df = pd.read_csv(matches[0]).sort_values("Image Index").reset_index(drop=True)

    def _manual_ece(self, y, p, n_bins=15, lo=0.0, hi=1.0):
        edges = np.linspace(lo, hi, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            idx = (p >= edges[b]) & (p < edges[b + 1])
            if b == n_bins - 1:  # include the right edge in the last bin
                idx = idx | (p == hi)
            n_b = int(idx.sum())
            if n_b == 0:
                continue
            conf_b = float(p[idx].mean())
            acc_b = float(y[idx].mean())
            ece += (n_b / len(p)) * abs(acc_b - conf_b)
        return ece

    def test_manual_ece_matches_published_raw(self):
        exp = load("outputs/metrics/experiments/exp1_calibration.json")
        for lbl in C.TARGET_LABELS:
            y = self.df[f"true_{lbl}"].to_numpy()
            p = self.df[f"prob_{lbl}"].to_numpy()
            manual = self._manual_ece(y, p)
            published = exp["splits"]["test"][lbl]["raw"]["ece"]
            self.assertAlmostEqual(manual, published, places=5,
                                   msg=f"{lbl} raw ECE independent recompute")

    def test_manual_ece_temperature_invariant_definition_is_finite(self):
        # p = sigmoid(logit/T) — recompute calibrated probabilities and check
        # the manual ECE on them equals the published calibrated ECE.
        import json
        temp_path = C.CALIBRATION_METRICS_DIR / "temperature_scalers.json"
        temps = json.loads(temp_path.read_text())["metadata"]["temperatures"]
        exp = load("outputs/metrics/experiments/exp1_calibration.json")
        for lbl in C.TARGET_LABELS:
            logits = self.df[f"logit_{lbl}"].to_numpy()
            y = self.df[f"true_{lbl}"].to_numpy()
            p_cal = 1.0 / (1.0 + np.exp(-logits / float(temps[lbl])))
            manual = self._manual_ece(y, p_cal)
            published = exp["splits"]["test"][lbl]["calibrated"]["ece"]
            self.assertAlmostEqual(manual, published, places=5,
                                   msg=f"{lbl} calibrated ECE independent recompute")

    def test_bins_are_equal_width_and_count_15(self):
        exp = load("outputs/metrics/experiments/exp1_calibration.json")
        for lbl in C.TARGET_LABELS:
            d = exp["splits"]["test"][lbl]["raw"]
            self.assertEqual(d["n_bins"], 15)
            self.assertEqual(d["binning"], "equal_width")


if __name__ == "__main__":
    unittest.main()